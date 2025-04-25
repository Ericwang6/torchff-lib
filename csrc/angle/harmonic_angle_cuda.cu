#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void harmonic_angle_energy_cuda_kernel(
    scalar_t* coords, int64_t* triplets, 
    scalar_t* theta0, scalar_t* k, 
    scalar_t* ene, int64_t nangles
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nangles) {
        return;
    }
    int offset = index * 3;
    scalar_t* coords_0 = coords + triplets[offset] * 3;
    scalar_t* coords_1 = coords + triplets[offset + 1] * 3;
    scalar_t* coords_2 = coords + triplets[offset + 2] * 3;

    scalar_t v1x = coords_0[0] - coords_1[0];
    scalar_t v1y = coords_0[1] - coords_1[1];
    scalar_t v1z = coords_0[2] - coords_1[2];

    scalar_t v2x = coords_2[0] - coords_1[0];
    scalar_t v2y = coords_2[1] - coords_1[1];
    scalar_t v2z = coords_2[2] - coords_1[2];

    scalar_t v1_norm = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    scalar_t v2_norm = sqrt(v2x * v2x + v2y * v2y + v2z * v2z);

    scalar_t dot_product = v1x * v2x + v1y * v2y + v1z * v2z;
    scalar_t cos_theta = dot_product / (v1_norm * v2_norm);
    scalar_t theta = acos(cos_theta);

    ene[index] = k[index] * pow(theta - theta0[index], 2) / 2;
}

template <typename scalar_t>
__global__ void harmonic_angle_energy_grad_cuda_kernel(
    scalar_t* coords, int64_t* triplets, 
    scalar_t* theta0, scalar_t* k, 
    scalar_t* coord_grad, scalar_t* theta0_grad, scalar_t* k_grad, 
    int64_t nangles
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nangles) {
        return;
    }
    int offset = index * 3;
    int64_t offset_0 = triplets[offset] * 3;
    int64_t offset_1 = triplets[offset + 1] * 3;
    int64_t offset_2 = triplets[offset + 2] * 3;

    scalar_t* coords_0 = coords + offset_0;
    scalar_t* coords_1 = coords + offset_1;
    scalar_t* coords_2 = coords + offset_2;

    scalar_t v1x = coords_0[0] - coords_1[0];
    scalar_t v1y = coords_0[1] - coords_1[1];
    scalar_t v1z = coords_0[2] - coords_1[2];

    scalar_t v2x = coords_2[0] - coords_1[0];
    scalar_t v2y = coords_2[1] - coords_1[1];
    scalar_t v2z = coords_2[2] - coords_1[2];

    scalar_t v1_norm = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    scalar_t v2_norm = sqrt(v2x * v2x + v2y * v2y + v2z * v2z);

    scalar_t dot_product = v1x * v2x + v1y * v2y + v1z * v2z;
    scalar_t cos_theta = dot_product / (v1_norm * v2_norm);
    scalar_t theta = acos(cos_theta);

    scalar_t sin_theta = sqrt(1 - cos_theta * cos_theta);
    scalar_t dtheta_dcos = -1 / sin_theta;

    scalar_t k_ = k[index];
    scalar_t dtheta = theta - theta0[index];
    scalar_t prefix = k_ * dtheta * dtheta_dcos / (v1_norm * v2_norm);

    scalar_t g1x = prefix * (v2x - cos_theta * v1x / v1_norm * v2_norm);
    scalar_t g1y = prefix * (v2y - cos_theta * v1y / v1_norm * v2_norm);
    scalar_t g1z = prefix * (v2z - cos_theta * v1z / v1_norm * v2_norm);

    scalar_t g3x = prefix * (v1x - cos_theta * v2x / v2_norm * v1_norm);
    scalar_t g3y = prefix * (v1y - cos_theta * v2y / v2_norm * v1_norm);
    scalar_t g3z = prefix * (v1z - cos_theta * v2z / v2_norm * v1_norm);

    atomicAdd(&coord_grad[offset_0],     g1x);
    atomicAdd(&coord_grad[offset_0 + 1], g1y);
    atomicAdd(&coord_grad[offset_0 + 2], g1z);

    atomicAdd(&coord_grad[offset_1],     -g1x-g3x);
    atomicAdd(&coord_grad[offset_1 + 1], -g1y-g3y);
    atomicAdd(&coord_grad[offset_1 + 2], -g1z-g3z);

    atomicAdd(&coord_grad[offset_2],     g3x);
    atomicAdd(&coord_grad[offset_2 + 1], g3y);
    atomicAdd(&coord_grad[offset_2 + 2], g3z);

    k_grad[index] = dtheta * dtheta / 2;
    theta0_grad[index] = -k_ * dtheta;
}

at::Tensor compute_harmonic_angle_energy_cuda(
    at::Tensor& coords,
    at::Tensor& triplets,
    at::Tensor& theta0,
    at::Tensor& k
) {
    int64_t nangles = triplets.size(0);

    int block_dim = 1024;
    int grid_dim = (nangles + block_dim - 1) / block_dim;

    auto ene = at::zeros({nangles}, coords.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_angle_energy_cuda", ([&] {
        harmonic_angle_energy_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            triplets.data_ptr<int64_t>(),
            theta0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            ene.data_ptr<scalar_t>(),
            nangles
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return at::sum(ene);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_harmonic_angle_energy_grad_cuda(
    at::Tensor& coords,
    at::Tensor& triplets,
    at::Tensor& theta0,
    at::Tensor& k
) {
    int64_t nangles = triplets.size(0);

    int block_dim = 512;
    int grid_dim = (nangles + block_dim - 1) / block_dim;

    auto coord_grad = at::zeros_like(coords, coords.options());
    auto theta0_grad = at::zeros_like(theta0, theta0.options());
    auto k_grad = at::zeros_like(k, k.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_angle_energy_grad_cuda", ([&] {
        harmonic_angle_energy_grad_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            triplets.data_ptr<int64_t>(),
            theta0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            coord_grad.data_ptr<scalar_t>(),
            theta0_grad.data_ptr<scalar_t>(),
            k_grad.data_ptr<scalar_t>(),
            nangles
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return std::make_tuple(coord_grad, theta0_grad, k_grad);
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_harmonic_angle_energy", compute_harmonic_angle_energy_cuda);
    m.impl("compute_harmonic_angle_energy_grad", compute_harmonic_angle_energy_grad_cuda);
}
