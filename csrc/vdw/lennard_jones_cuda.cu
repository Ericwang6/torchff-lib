#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"


template <typename scalar_t>
__global__ void lennard_jones_cuda_kernel(
    scalar_t* coords, 
    int64_t* pairs, 
    scalar_t* box,
    scalar_t* box_inv,
    scalar_t* sigma, 
    scalar_t* epsilon, 
    scalar_t* ene, 
    scalar_t* coord_grad, 
    scalar_t* sigma_grad, 
    scalar_t* epsilon_grad,
    int64_t npairs,
    double cutoff
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= npairs) {
        return;
    }
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
        return;
    }
    scalar_t rinv = 1 / tmp[0];

    scalar_t sigma_ij = (sigma[i] + sigma[j]) / 2;
    scalar_t epsilon_ij = sqrt_(epsilon[i] * epsilon[j]);

    scalar_t sigma_ij_6 = pow(sigma_ij, 6);
    scalar_t rinv6 = pow(rinv, 6);

    tmp[0] = 4 * epsilon_ij * sigma_ij_6 * (sigma_ij_6 * rinv6 * rinv6 - rinv6);
    ene[index] = tmp[0];

    atomicAdd(&epsilon_grad[i], tmp[0] / 2 / epsilon[i]);
    atomicAdd(&epsilon_grad[j], tmp[0] / 2 / epsilon[j]);
    
    tmp[0] = 24 * epsilon_ij * sigma_ij_6 * (2 * sigma_ij_6 * rinv6 * rinv6 - rinv6);
    scalar_t sigma_deriv = tmp[0] / 2 / sigma_ij;
    atomicAdd(&sigma[i], sigma_deriv);
    atomicAdd(&sigma[j], sigma_deriv);

    tmp[0] = tmp[0] * rinv * rinv;
    scalar_t g;
    for (int i = 0; i < 3; i++) {
        g = tmp[0] * rij[i];
        atomicAdd(&coord_grad[offset_i + i], -g);
        atomicAdd(&coord_grad[offset_j + i], g);
    }

}


std::vector<at::Tensor> compute_lennard_jones_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    double cutoff
) {
    int64_t npairs = pairs.size(0);

    at::Tensor box_inv = at::linalg_inv(box);

    int block_dim = 512;
    int grid_dim = (npairs + block_dim - 1) / block_dim;

    auto ene = at::zeros({npairs}, coords.options());
    auto coord_grad = at::zeros_like(coords, coords.options());
    auto sigma_grad = at::zeros_like(sigma, sigma.options());
    auto epsilon_grad = at::zeros_like(epsilon, epsilon.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_lennard_jones_cuda", ([&] {
        lennard_jones_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            sigma.data_ptr<scalar_t>(),
            epsilon.data_ptr<scalar_t>(),
            ene.data_ptr<scalar_t>(),
            coord_grad.data_ptr<scalar_t>(),
            sigma_grad.data_ptr<scalar_t>(),
            epsilon_grad.data_ptr<scalar_t>(),
            npairs,
            cutoff
        );
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return {at::sum(ene), coord_grad, sigma_grad, epsilon_grad};
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_lennard_jones", compute_lennard_jones_cuda);
}