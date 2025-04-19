#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"


template <typename scalar_t>
__global__ void periodic_torsion_cuda_kernel(
    scalar_t* coords, int64_t* torsions, scalar_t* fc, int64_t* per, scalar_t* phase, 
    scalar_t* ene, scalar_t* coord_grad, scalar_t* fc_grad, scalar_t* phase_grad,
    int64_t ntors
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ntors) {
        return;
    }
    int offset = index * 4;
    int64_t offset_i = 3 * torsions[offset];
    int64_t offset_j = 3 * torsions[offset + 1];
    int64_t offset_k = 3 * torsions[offset + 2];
    int64_t offset_l = 3 * torsions[offset + 3];

    scalar_t* coords_i = coords + offset_i;
    scalar_t* coords_j = coords + offset_j;
    scalar_t* coords_k = coords + offset_k;
    scalar_t* coords_l = coords + offset_l;

    scalar_t b1[3];
    scalar_t b2[3];
    scalar_t b3[3];

    scalar_t n1[3];
    scalar_t n2[3];

    diff_vec3(coords_j, coords_i, b1);
    diff_vec3(coords_k, coords_j, b2);
    diff_vec3(coords_l, coords_k, b3);

    cross_vec3(b1, b2, n1);
    cross_vec3(b2, b3, n2);

    scalar_t norm_n1 = norm_vec3(n1);
    scalar_t norm_n2 = norm_vec3(n2);
    scalar_t norm_b2 = norm_vec3(b2);
    scalar_t norm_b2_sqr = norm_b2 * norm_b2;

    scalar_t cosval = dot_vec3(n1, n2) / (norm_n1 * norm_n2);
    cosval = min(scalar_t(1.0), max(scalar_t(-1.0), cosval));

    scalar_t k = fc[index];
    int64_t n = per[index];
    scalar_t phi = acos(cosval);
    phi = dot_vec3(n1, b3) > 0.0 ? phi : -phi;
    phi = n * phi - phase[index];
    
    scalar_t tmp1 = 1 + cos(phi);
    scalar_t tmp2 = k * sin(phi);
    ene[index] = k * tmp1;
    scalar_t prefactor = tmp2 * n;

    scalar_t aux1 = dot_vec3(b1, b2) / norm_b2_sqr;
    scalar_t aux2 = dot_vec3(b2, b3) / norm_b2_sqr;

    scalar_t cgi, cgj, cgk, cgl;
    for (int i = 0; i < 3; i++) {
        cgi = prefactor * norm_b2 / (norm_n1 * norm_n1) * n1[i];
        cgl = -prefactor * norm_b2 / (norm_n2 * norm_n2) * n2[i];
        cgj = (-aux1 - 1) * cgi + aux2 * cgl;
        cgk = -cgi - cgj - cgl;
        atomicAdd(&coord_grad[offset_i + i], cgi);
        atomicAdd(&coord_grad[offset_j + i], cgj);
        atomicAdd(&coord_grad[offset_l + i], cgl);
        atomicAdd(&coord_grad[offset_k + i], cgk);
    }

    fc_grad[index] = tmp1;
    phase_grad[index] = tmp2;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> compute_periodic_torsion_cuda(
    at::Tensor& coords,
    at::Tensor& torsions,
    at::Tensor& fc,
    at::Tensor& per,
    at::Tensor& phase
) {
    int64_t ntors = torsions.size(0);

    int block_dim = 512;
    int grid_dim = (ntors + block_dim - 1) / block_dim;

    auto ene = at::zeros({ntors}, coords.options());
    auto coord_grad = at::zeros_like(coords, coords.options());
    auto fc_grad = at::zeros_like(fc, fc.options());
    auto phase_grad = at::zeros_like(phase, phase.options());

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_periodic_torsion_cuda", ([&] {
        periodic_torsion_cuda_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            torsions.data_ptr<int64_t>(),
            fc.data_ptr<scalar_t>(),
            per.data_ptr<int64_t>(),
            phase.data_ptr<scalar_t>(),
            ene.data_ptr<scalar_t>(),
            coord_grad.data_ptr<scalar_t>(),
            fc_grad.data_ptr<scalar_t>(),
            phase_grad.data_ptr<scalar_t>(),
            ntors
        );
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return std::make_tuple(at::sum(ene), coord_grad, fc_grad, phase_grad);
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_periodic_torsion", compute_periodic_torsion_cuda);
}