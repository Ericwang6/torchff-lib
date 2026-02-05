#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/reduce.cuh"


template <typename scalar_t, int BLOCK_SIZE, bool DO_SHIFT>
__global__ void coulomb_cuda_kernel(
    scalar_t* coords,
    int64_t* pairs, 
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff,
    scalar_t coulomb_constant,
    scalar_t* charges,
    int64_t npairs,
    scalar_t* ene_out, 
    scalar_t* coord_grad, 
    scalar_t* charge_grad
) {

    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];

    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }

    __syncthreads();

    scalar_t ene = static_cast<scalar_t>(0.0);

    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {

        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
        int64_t offset_i = 3 * i;
        int64_t offset_j = 3 * j;
        scalar_t rij[3];
        scalar_t tmp[3];
        diff_vec3(&coords[offset_i], &coords[offset_j], tmp);
        apply_pbc_triclinic(tmp, box, box_inv, rij);

        tmp[0] = norm_vec3(rij);
        
        if (tmp[0] > cutoff) {
            continue;
        }

        scalar_t rinv = static_cast<scalar_t>(1.0) / tmp[0];
        scalar_t rinv2 = rinv * rinv;
        scalar_t rinv3 = rinv2 * rinv;
        scalar_t ci = charges[i];
        scalar_t cj = charges[j];

        scalar_t p = coulomb_constant * ci * cj;

        // Energy 
        scalar_t e_pair;
        if constexpr (DO_SHIFT) {
            e_pair = p * (rinv - static_cast<scalar_t>(1.0) / cutoff);
        } else {
            e_pair = p * rinv;
        }
        ene += e_pair;

        // Charge gradients
        if (charge_grad) {
            atomicAdd(&charge_grad[i], e_pair / ci);
            atomicAdd(&charge_grad[j], e_pair / cj);
        }

        // Coordinate gradients
        if (coord_grad) {
            scalar_t g_coeff = p * rinv3;
            scalar_t g;
            #pragma unroll
            for (int d = 0; d < 3; d++) {
                g = g_coeff * rij[d];
                atomicAdd(&coord_grad[offset_i + d], -g);
                atomicAdd(&coord_grad[offset_j + d],  g);
            }
        }
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class CoulombFunctionCuda: public torch::autograd::Function<CoulombFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& pairs,
        at::Tensor& box,
        at::Tensor& charges,
        at::Scalar coulomb_constant,
        at::Scalar cutoff,
        bool do_shift
    ) {
        int64_t npairs = pairs.size(0);

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int GRID_SIZE = std::min(
            static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->multiProcessorCount * props->maxBlocksPerMultiProcessor
        );

        at::Tensor ene = at::zeros({1}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor charges_grad = at::zeros_like(charges, charges.options());
        at::Tensor box_inv = at::linalg_inv(box);

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_coulomb", ([&] {
            const scalar_t cutoff_val = cutoff.to<scalar_t>();
            const scalar_t coulomb_constant_val = coulomb_constant.to<scalar_t>();

            if (do_shift) {
                coulomb_cuda_kernel<scalar_t, BLOCK_SIZE, true><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    box_inv.data_ptr<scalar_t>(),
                    cutoff_val,
                    coulomb_constant_val,
                    charges.data_ptr<scalar_t>(),
                    npairs,
                    ene.data_ptr<scalar_t>(),
                    coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                    charges.requires_grad() ? charges_grad.data_ptr<scalar_t>() : nullptr
                );
            } else {
                coulomb_cuda_kernel<scalar_t, BLOCK_SIZE, false><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    box_inv.data_ptr<scalar_t>(),
                    cutoff_val,
                    coulomb_constant_val,
                    charges.data_ptr<scalar_t>(),
                    npairs,
                    ene.data_ptr<scalar_t>(),
                    coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                    charges.requires_grad() ? charges_grad.data_ptr<scalar_t>() : nullptr
                );
            }
        }));

        ctx->save_for_backward({coord_grad, charges_grad});
        return ene;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    ){
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[0] * grad_outputs[0], 
            ignore, 
            ignore,
            saved[1] * grad_outputs[0], 
            ignore,
            ignore,
            ignore
        };
    }

};
    
    
at::Tensor compute_coulomb_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& charges,
    at::Scalar coulomb_constant,
    at::Scalar cutoff,
    bool do_shift
) {
    return CoulombFunctionCuda::apply(coords, pairs, box, charges, coulomb_constant, cutoff, do_shift);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_coulomb_energy", compute_coulomb_energy_cuda);
}