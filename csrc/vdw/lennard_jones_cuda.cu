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


template <typename scalar_t, int BLOCK_SIZE = 256>
__global__ void lennard_jones_cuda_kernel(
    scalar_t* coords, 
    int64_t* pairs, 
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff,
    scalar_t* sigma, 
    scalar_t* epsilon,
    int64_t npairs, 
    scalar_t* ene_out, 
    scalar_t* coord_grad, 
    scalar_t* sigma_grad, 
    scalar_t* epsilon_grad
) {
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];

    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }

    __syncthreads();

    scalar_t ene = scalar_t(0.0);

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
        if ( tmp[0] > cutoff ) {
            continue;
        }
        else {
            scalar_t rinv = 1 / tmp[0];

            const scalar_t ei = epsilon[i];
            const scalar_t ej = epsilon[j];
            const scalar_t si = sigma[i];
            const scalar_t sj = sigma[j];
            const scalar_t sij = (si + sj) / 2;
            const scalar_t eij = sqrt_(ei * ej);

            scalar_t s2 = sij * sij;
            scalar_t rinv2 = rinv * rinv;

            scalar_t sij_6 = s2 * s2 * s2;
            scalar_t rinv6 = rinv2 * rinv2 * rinv2;

            // Energy contribution
            scalar_t ene_pair = 4 * eij * sij_6 * rinv6 * (sij_6 * rinv6 - 1);
            ene += ene_pair;

            // Epsilon gradient
            if (epsilon_grad) {
                atomicAdd(&epsilon_grad[i], ene_pair / (2 * ei));
                atomicAdd(&epsilon_grad[j], ene_pair / (2 * ej));
            }

            // Sigma and coordinate gradients
            if (sigma_grad || coord_grad) {
                scalar_t tmp_force = 24 * eij * sij_6 * rinv6 * (2 * sij_6 * rinv6 - 1);

                if (sigma_grad) {
                    scalar_t sigma_deriv = tmp_force / (2 * sij);
                    atomicAdd(&sigma_grad[i], sigma_deriv);
                    atomicAdd(&sigma_grad[j], sigma_deriv);
                }

                if (coord_grad) {
                    scalar_t force_coeff = tmp_force * rinv2;
                    scalar_t g;
                    #pragma unroll
                    for (int d = 0; d < 3; d++) {
                        g = force_coeff * rij[d];
                        atomicAdd(&coord_grad[offset_i + d], -g);
                        atomicAdd(&coord_grad[offset_j + d],  g);
                    }
                }
            }
        }
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
    }
}


class LennardJonesFunctionCuda: public torch::autograd::Function<LennardJonesFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& pairs,
        at::Tensor& box,
        at::Tensor& sigma,
        at::Tensor& epsilon,
        at::Scalar cutoff
    )
    {
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
        at::Tensor sigma_grad = at::zeros_like(sigma, coords.options());
        at::Tensor epsilon_grad = at::zeros_like(epsilon, coords.options());
        at::Tensor box_inv = at::linalg_inv(box);

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_lennard_jones", ([&] {
            lennard_jones_cuda_kernel<scalar_t, BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                pairs.data_ptr<int64_t>(),
                box.data_ptr<scalar_t>(),
                box_inv.data_ptr<scalar_t>(),
                cutoff.to<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                epsilon.data_ptr<scalar_t>(),
                npairs,
                ene.data_ptr<scalar_t>(),
                (coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr),
                (sigma.requires_grad() ? sigma_grad.data_ptr<scalar_t>() : nullptr),
                (epsilon.requires_grad() ? epsilon_grad.data_ptr<scalar_t>() : nullptr)
            );
        }));
        ctx->save_for_backward({coord_grad, sigma_grad, epsilon_grad});
        return ene;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    )
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[0] * grad_outputs[0], 
            ignore, 
            ignore,
            saved[1] * grad_outputs[0], 
            saved[2] * grad_outputs[0],
            ignore
        };
    }

};


at::Tensor compute_lennard_jones_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Scalar cutoff
) {
    return LennardJonesFunctionCuda::apply(coords, pairs, box, sigma, epsilon, cutoff);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_lennard_jones_energy", compute_lennard_jones_energy_cuda);
}