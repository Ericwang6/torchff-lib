
#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"



template <typename scalar_t>
__global__ void coulomb_cuda_kernel(
    scalar_t* coords,
    scalar_t* charges, 
    int64_t* pairs, 
    scalar_t* box,
    scalar_t* box_inv,
    scalar_t* epsilon, 
    scalar_t* ene, 
    scalar_t* coord_grad, 
    scalar_t* charge_grad,
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

    constexpr scalar_t PI = 3.14159265358979323846;

    scalar_t rinv = 1 / tmp[0];
    scalar_t rinv2 = pow(rinv, 2);
    scalar_t prefac = 1.0 / (4.0 * PI * (*epsilon));

    ene[index] = prefac * charges[i] * charges[j] * rinv;

    // Epsilon gradient
    atomicAdd(epsilon_grad, -ene[index] / (*epsilon));

    // Charge gradients
    atomicAdd(&charge_grad[i], ene[index] / charges[i]);
    atomicAdd(&charge_grad[j], ene[index] / charges[j]);

    // Coordinate gradients
    for (int d = 0; d < 3; d++) {
        scalar_t g = ene[index] * rij[d] * rinv2;
        atomicAdd(&coord_grad[offset_i + d], -g);
        atomicAdd(&coord_grad[offset_j + d],  g);
    }
}


std::vector<at::Tensor> compute_coulomb_cuda(
    at::Tensor& coords,
    at::Tensor& charges,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& epsilon,
    double cutoff
) {
    int64_t npairs = pairs.size(0);

    at::Tensor box_inv = at::linalg_inv(box);

    int block_dim = 256;
    int grid_dim = (npairs + block_dim - 1) / block_dim;

    auto ene = at::zeros({npairs}, coords.options());
    auto coord_grad = at::zeros_like(coords, coords.options());
    auto charge_grad = at::zeros_like(charges, charges.options());
    auto epsilon_grad = at::zeros_like(epsilon, epsilon.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_coulomb_cuda", ([&] {
        coulomb_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
	    charges.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            epsilon.data_ptr<scalar_t>(),
            ene.data_ptr<scalar_t>(),
            coord_grad.data_ptr<scalar_t>(),
            charge_grad.data_ptr<scalar_t>(),
            epsilon_grad.data_ptr<scalar_t>(),
            npairs,
            cutoff
        );
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return {at::sum(ene), coord_grad, charge_grad, epsilon_grad};
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_coulomb_energy", compute_coulomb_cuda);
}

