#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"


template <typename scalar_t> 
__global__ void build_neighbor_list_nsquared_kernel(
    scalar_t* coords,
    scalar_t* box,
    scalar_t* box_inv,
    scalar_t cutoff2,
    int32_t* pairs,
    int32_t* npairs,
    int32_t natoms,
    int32_t max_npairs
)
{
    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index >= natoms * (natoms - 1) / 2 ) {
        return;
    }
    int32_t i = floor((sqrt(index * 8 + 1) + 1) / 2);
    // if (i * (i - 1) > 2 * index) i--;
    int32_t j = index - (i * (i - 1)) / 2;

    scalar_t drvec[3];
    diff_vec3(&coords[i * 3], &coords[j * 3], drvec);
    apply_pbc_triclinic(drvec, box, box_inv, drvec);

    scalar_t dist2 = drvec[0] * drvec[0] + drvec[1] * drvec[1] + drvec[2] * drvec[2];
    if ( dist2 > cutoff2 ) {
        return;
    }
    int32_t i_pair = atomicAdd(npairs, 1);
    pairs[i_pair * 2] = i;
    pairs[i_pair * 2 + 1] = j;
}


at::Tensor build_neighbor_list_nsquared_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    double cutoff,
    int64_t max_npairs
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);

    int32_t max_npairs_;
    if ( max_npairs == -1 ) {
        max_npairs_ = natoms * (natoms - 1) / 2;
    }
    else {
        max_npairs_ = static_cast<int32_t>(max_npairs);
    }

    int32_t *d_npairs;
    cudaMalloc(&d_npairs, sizeof(int32_t));
    cudaMemset(d_npairs, 0, sizeof(int32_t));

    int block_dim = 128;
    int grid_dim = (natoms * (natoms - 1) / 2 + block_dim - 1) / block_dim;

    at::Tensor pairs = at::empty({max_npairs_, 2}, coords.options().dtype(at::kInt));
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "build_neighbor_list_nsquared_cuda", ([&] {
        build_neighbor_list_nsquared_kernel<scalar_t><<<grid_dim, block_dim>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff * cutoff),
            pairs.data_ptr<int32_t>(),
            d_npairs,
            natoms,
            max_npairs_
        );
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    // check if the number of pairs exceeds the capacity
    int32_t npairs_found = 0;
    cudaMemcpy(&npairs_found, d_npairs, sizeof(int32_t), cudaMemcpyDeviceToHost);
    TORCH_CHECK(npairs_found <= max_npairs_, "Too many neighbor pairs found. Maximum is " + std::to_string(max_npairs_), " but found " + std::to_string(npairs_found));
    cudaFree(d_npairs);

    return pairs.index({at::indexing::Slice(0, npairs_found), at::indexing::Slice()});
}


TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_neighbor_list_nsquared", build_neighbor_list_nsquared_cuda);
}