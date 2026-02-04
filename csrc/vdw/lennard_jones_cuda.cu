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
__global__ void lennard_jones_cuda_kernel(
    scalar_t* coords, 
    int32_t* pairs, 
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff,
    scalar_t* sigma, 
    scalar_t* epsilon,
    int32_t npairs, 
    scalar_t* ene, 
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

        tmp[0] = 4 * eij * sij_6 * rinv6 * (sij_6 * rinv6 - 1);
        ene[index] = tmp[0];

        atomicAdd(&epsilon_grad[i], tmp[0] / 2 / ei);
        atomicAdd(&epsilon_grad[j], tmp[0] / 2 / ej);
        
        tmp[0] = 24 * eij * sij_6 * rinv6 * (2 * sij_6 * rinv6 - 1);
        scalar_t sigma_deriv = tmp[0] / 2 / sij;
        atomicAdd(&sigma_grad[i], sigma_deriv);
        atomicAdd(&sigma_grad[j], sigma_deriv);

        tmp[0] = tmp[0] * rinv2;
        scalar_t g;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            g = tmp[0] * rij[i];
            atomicAdd(&coord_grad[offset_i + i], -g);
            atomicAdd(&coord_grad[offset_j + i], g);
        }
        return;
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
        int32_t npairs = pairs.size(0);
        int32_t block_dim = 256;
        int32_t grid_dim = (npairs + block_dim - 1) / block_dim;

        at::Tensor ene = at::empty({npairs}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor sigma_grad = at::zeros_like(sigma, coords.options());
        at::Tensor epsilon_grad = at::zeros_like(epsilon, coords.options());
        at::Tensor box_inv = at::linalg_inv(box);

        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_lennard_jones", ([&] {
            lennard_jones_cuda_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                pairs.data_ptr<int32_t>(),
                box.data_ptr<scalar_t>(),
                box_inv.data_ptr<scalar_t>(),
                cutoff.to<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                epsilon.data_ptr<scalar_t>(),
                npairs,
                ene.data_ptr<scalar_t>(),
                coord_grad.data_ptr<scalar_t>(),
                sigma_grad.data_ptr<scalar_t>(),
                epsilon_grad.data_ptr<scalar_t>()
            );
        }));
        at::Tensor e = at::sum(ene);
        ctx->save_for_backward({coord_grad, sigma_grad, epsilon_grad});
        return e;
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