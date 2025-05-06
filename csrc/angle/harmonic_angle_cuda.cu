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

template <typename scalar_t>
__global__ void harmonic_angle_cuda_kernel(
    scalar_t* coords, 
    int32_t* angles, 
    scalar_t* theta0, 
    scalar_t* k, 
    int32_t nangles,
    scalar_t* ene,
    scalar_t* coord_grad, 
    scalar_t* theta0_grad, 
    scalar_t* k_grad,
    scalar_t sign
) {

    // __shared__ scalar_t s_ene;
    // if ( threadIdx.x == 0 && ene ) {
    //     s_ene = 0.0;
    // }
    // __syncthreads();

    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nangles) {
        return;
    }
    int32_t offset = index * 3;
    int32_t offset_0 = angles[offset] * 3;
    int32_t offset_1 = angles[offset + 1] * 3;
    int32_t offset_2 = angles[offset + 2] * 3;

    scalar_t* coords_0 = coords + offset_0;
    scalar_t* coords_1 = coords + offset_1;
    scalar_t* coords_2 = coords + offset_2;

    scalar_t v1x = coords_0[0] - coords_1[0];
    scalar_t v1y = coords_0[1] - coords_1[1];
    scalar_t v1z = coords_0[2] - coords_1[2];

    scalar_t v2x = coords_2[0] - coords_1[0];
    scalar_t v2y = coords_2[1] - coords_1[1];
    scalar_t v2z = coords_2[2] - coords_1[2];

    scalar_t v1_norm = sqrt_(v1x * v1x + v1y * v1y + v1z * v1z);
    scalar_t v2_norm = sqrt_(v2x * v2x + v2y * v2y + v2z * v2z);

    scalar_t dot_product = v1x * v2x + v1y * v2y + v1z * v2z;
    scalar_t cos_theta = dot_product / (v1_norm * v2_norm);
    scalar_t theta = acos_(cos_theta);

    scalar_t sin_theta = sqrt_(1 - cos_theta * cos_theta);
    scalar_t dtheta_dcos = -1 / sin_theta;

    scalar_t k_ = k[index];
    scalar_t dtheta = theta - theta0[index];
    scalar_t prefix = k_ * dtheta * dtheta_dcos / (v1_norm * v2_norm) * sign;

    scalar_t g1x = prefix * (v2x - cos_theta * v1x / v1_norm * v2_norm);
    scalar_t g1y = prefix * (v2y - cos_theta * v1y / v1_norm * v2_norm);
    scalar_t g1z = prefix * (v2z - cos_theta * v1z / v1_norm * v2_norm);

    scalar_t g3x = prefix * (v1x - cos_theta * v2x / v2_norm * v1_norm);
    scalar_t g3y = prefix * (v1y - cos_theta * v2y / v2_norm * v1_norm);
    scalar_t g3z = prefix * (v1z - cos_theta * v2z / v2_norm * v1_norm);

    if ( ene ) {
        // atomicAdd(&s_ene, k_ * dtheta * dtheta / 2);
        ene[index] = k_ * dtheta * dtheta / 2;
    }

    atomicAdd(&coord_grad[offset_0],     g1x);
    atomicAdd(&coord_grad[offset_0 + 1], g1y);
    atomicAdd(&coord_grad[offset_0 + 2], g1z);

    atomicAdd(&coord_grad[offset_1],     -g1x-g3x);
    atomicAdd(&coord_grad[offset_1 + 1], -g1y-g3y);
    atomicAdd(&coord_grad[offset_1 + 2], -g1z-g3z);

    atomicAdd(&coord_grad[offset_2],     g3x);
    atomicAdd(&coord_grad[offset_2 + 1], g3y);
    atomicAdd(&coord_grad[offset_2 + 2], g3z);

    if ( k_grad ) {
        k_grad[index] = dtheta * dtheta / 2;
    }
    if ( theta0_grad ) {
        theta0_grad[index] = -k_ * dtheta;
    }

    // __syncthreads();
    // if ( threadIdx.x == 0 && ene ) {
    //     atomicAdd(ene, s_ene);
    // }
}


class HarmonicAngleFunctionCuda: public torch::autograd::Function<HarmonicAngleFunctionCuda> {

    public: 
        static at::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            at::Tensor& coords,
            at::Tensor& angles,
            at::Tensor& theta0,
            at::Tensor& k
        )
        {
            int32_t nangles = angles.size(0);
            int32_t block_dim = 1024;
            int32_t grid_dim = (nangles + block_dim - 1) / block_dim;
    
            at::Tensor e = at::zeros({nangles}, coords.options());
            at::Tensor coord_grad = at::zeros_like(coords, coords.options());
            at::Tensor theta0_grad = at::zeros_like(theta0, theta0.options());
            at::Tensor k_grad = at::zeros_like(k, k.options());

            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_angle_cuda", ([&] {
                harmonic_angle_cuda_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    angles.data_ptr<int32_t>(),
                    theta0.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    nangles,
                    e.data_ptr<scalar_t>(),
                    coord_grad.data_ptr<scalar_t>(),
                    theta0_grad.data_ptr<scalar_t>(),
                    k_grad.data_ptr<scalar_t>(),
                    static_cast<scalar_t>(1.0)
                );
            }));
            ctx->save_for_backward({coord_grad, theta0_grad, k_grad});
            return at::sum(e);
        }
    
        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* ctx,
            std::vector<at::Tensor> grad_outputs
        )
        {
            auto saved = ctx->get_saved_variables();
            at::Tensor ignore;
            return {saved[0] * grad_outputs[0], ignore, saved[1] * grad_outputs[0], saved[2] * grad_outputs[0]};
        }
    };


at::Tensor compute_harmonic_angle_energy_cuda(
    at::Tensor& coords,
    at::Tensor& angles,
    at::Tensor& theta0,
    at::Tensor& k
) {
    return HarmonicAngleFunctionCuda::apply(coords, angles, theta0, k);
}


void compute_harmonic_angle_forces_cuda(
    at::Tensor& coords,
    at::Tensor& angles,
    at::Tensor& theta0,
    at::Tensor& k,
    at::Tensor& forces
) {

    int32_t nangles = angles.size(0);
    int32_t block_dim = 1024;
    int32_t grid_dim = (nangles + block_dim - 1) / block_dim;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_harmonic_angle_forces_cuda", ([&] {
        harmonic_angle_cuda_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            angles.data_ptr<int32_t>(),
            theta0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            nangles,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            static_cast<scalar_t>(-1.0)
        );
    }));
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_harmonic_angle_energy", compute_harmonic_angle_energy_cuda);
    m.impl("compute_harmonic_angle_forces", compute_harmonic_angle_forces_cuda);
}
