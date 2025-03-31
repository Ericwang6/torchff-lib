#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void harmonic_bond_energy_cuda_kernel(scalar_t* coords, int64_t* pairs, scalar_t* b0, scalar_t* k, scalar_t* ene, int64_t nbonds) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nbonds) {
        return;
    }
    int offset = index * 2;
    scalar_t* coords_0 = coords + pairs[offset] * 3;
    scalar_t* coords_1 = coords + (pairs[offset + 1]) * 3;
    scalar_t dx = coords_1[0] - coords_0[0];
    scalar_t dy = coords_1[1] - coords_0[1];
    scalar_t dz = coords_1[2] - coords_0[2];
    scalar_t b = sqrt(dx * dx + dy * dy + dz * dz);
    // atomicAdd(ene, pow(b - b0[index] , 2) * k[index] / 2);
    ene[index] = pow(b - b0[index] , 2) * k[index] / 2;
}


template <typename scalar_t>
__global__ void harmonic_bond_energy_grad_cuda_kernel(
    scalar_t* coords, int64_t* pairs, 
    scalar_t* b0, scalar_t* k, 
    scalar_t* coord_grad, scalar_t* b0_grad, scalar_t* k_grad, 
    int64_t nbonds
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nbonds) {
        return;
    }
    int offset = index * 2;
    int64_t offset_0 = pairs[offset] * 3;
    int64_t offset_1 = pairs[offset + 1] * 3;
    scalar_t* coords_0 = coords + offset_0;
    scalar_t* coords_1 = coords + offset_1;
    scalar_t dx = coords_1[0] - coords_0[0];
    scalar_t dy = coords_1[1] - coords_0[1];
    scalar_t dz = coords_1[2] - coords_0[2];
    scalar_t b = sqrt(dx * dx + dy * dy + dz * dz);
    
    scalar_t k_ = k[index];
    scalar_t db = (b - b0[index]);
    scalar_t prefix = k_ * db / b; 
    scalar_t gx = dx * prefix;
    scalar_t gy = dy * prefix;
    scalar_t gz = dz * prefix;

    atomicAdd(&coord_grad[offset_0],      -gx);
    atomicAdd(&coord_grad[offset_0 + 1],  -gy);
    atomicAdd(&coord_grad[offset_0 + 2],  -gz);

    atomicAdd(&coord_grad[offset_1],      gx);
    atomicAdd(&coord_grad[offset_1 + 1],  gy);
    atomicAdd(&coord_grad[offset_1 + 2],  gz);

    k_grad[index] = db * db / 2;
    b0_grad[index] = -k_ * db;
}



at::Tensor compute_harmonic_bond_energy_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& b0,
    at::Tensor& k
) {
    int64_t nbonds = pairs.size(0);

    int block_dim = 1024;
    int grid_dim = (nbonds + block_dim - 1) / block_dim;

    auto ene = at::zeros({nbonds}, coords.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_bond_energy_cuda", ([&] {
        harmonic_bond_energy_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            b0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            ene.data_ptr<scalar_t>(),
            nbonds
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return at::sum(ene);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_harmonic_bond_energy_grad_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& b0,
    at::Tensor& k
) {
    int64_t nbonds = pairs.size(0);

    int block_dim = 1024;
    int grid_dim = (nbonds + block_dim - 1) / block_dim;

    auto coord_grad = at::zeros_like(coords, coords.options());
    auto b0_grad = at::zeros_like(b0, b0.options());
    auto k_grad = at::zeros_like(k, k.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_bond_energy_grad_cuda", ([&] {
        harmonic_bond_energy_grad_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            b0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            coord_grad.data_ptr<scalar_t>(),
            b0_grad.data_ptr<scalar_t>(),
            k_grad.data_ptr<scalar_t>(),
            nbonds
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return std::make_tuple(coord_grad, b0_grad, k_grad);
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_harmonic_bond_energy", compute_harmonic_bond_energy_cuda);
    m.impl("compute_harmonic_bond_energy_grad", compute_harmonic_bond_energy_grad_cuda);
}