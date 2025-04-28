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


template <typename scalar_t>
__global__ void coulomb_cuda_kernel(
    scalar_t* coords,
    int32_t* pairs, 
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff,
    scalar_t prefac,
    scalar_t* charges,
    scalar_t do_shift,  // 1 if shift to rcut else 0
    int32_t npairs,
    scalar_t* ene, 
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

    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= npairs) {
        return;
    }
    int32_t i = pairs[index * 2];
    int32_t j = pairs[index * 2 + 1];
    int32_t offset_i = 3 * i;
    int32_t offset_j = 3 * j;
    scalar_t rij[3];
    scalar_t tmp[3];
    diff_vec3(&coords[offset_i], &coords[offset_j], tmp);
    apply_pbc_triclinic(tmp, box, box_inv, rij);

    tmp[0] = norm_vec3(rij);
    
    if ( tmp[0] > cutoff ) {
        ene[index] = 0.0;
        return;
    }
    else {
        scalar_t rinv = 1 / tmp[0];
        scalar_t rinv3 = pow(rinv, 3);
        scalar_t ci = charges[i];
        scalar_t cj = charges[j];

        scalar_t p = prefac * ci * cj;

        // Energy 
        scalar_t e = p * (rinv - do_shift / cutoff);
        ene[index] = e;

        // Coordinate gradients
        for (int d = 0; d < 3; d++) {
            scalar_t g = p * rinv3 * rij[d];
            atomicAdd(&coord_grad[offset_i + d], -g);
            atomicAdd(&coord_grad[offset_j + d],  g);
        }

        // Charge gradients
        atomicAdd(&charge_grad[i], e / ci);
        atomicAdd(&charge_grad[j], e / cj);
        
        return;
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
        at::Tensor& prefac,
        at::Scalar cutoff,
        bool do_shift
    ) {
        int32_t npairs = pairs.size(0);
        int32_t block_dim = 256;
        int32_t grid_dim = (npairs + block_dim - 1) / block_dim;

        at::Tensor ene = at::empty({npairs}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor charges_grad = at::zeros_like(charges, coords.options());
        at::Tensor box_inv = at::linalg_inv(box);
        at::Scalar p = prefac.item();

        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_coulomb", ([&] {
            scalar_t shift = do_shift ? 1.0 : 0.0;
            coulomb_cuda_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                pairs.data_ptr<int32_t>(),
                box.data_ptr<scalar_t>(),
                box_inv.data_ptr<scalar_t>(),
                cutoff.to<scalar_t>(),
                p.to<scalar_t>(),
                charges.data_ptr<scalar_t>(),
                shift,
                npairs,
                ene.data_ptr<scalar_t>(),
                coord_grad.data_ptr<scalar_t>(),
                charges_grad.data_ptr<scalar_t>()
            );
        }));
        at::Tensor e = at::sum(ene);
        ctx->save_for_backward({coord_grad, charges_grad, e, prefac});
        return e;
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
            saved[2] / saved[3] * grad_outputs[0],
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
    at::Tensor& prefac,
    at::Scalar cutoff,
    bool do_shift
) {
    return CoulombFunctionCuda::apply(coords, pairs, box, charges, prefac, cutoff, do_shift);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_coulomb_energy", compute_coulomb_energy_cuda);
}