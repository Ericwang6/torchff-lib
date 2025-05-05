#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"

// #define DEBUG


template <typename scalar_t> 
__global__ void assign_cell_index_shared_kernel(
    scalar_t* coords,
    scalar_t* box_inv,
    scalar_t fcrx, scalar_t fcry, scalar_t fcrz, // cell size in fractional coords
    int32_t ncx, int32_t ncy, int32_t ncz, // number of cells in one dimension
    int32_t natoms,
    scalar_t* f_coords,
    int32_t* cell_indices,
    int32_t* natoms_per_cell
)
{
    __shared__ scalar_t s_box_inv[9];
    if ( threadIdx.x < 9 ) {
        s_box_inv[threadIdx.x] = box_inv[threadIdx.x];
    }
    __syncthreads();

    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index >= natoms ) {
        return;
    }

    scalar_t crd[3];
    crd[0] = coords[index * 3];
    crd[1] = coords[index * 3 + 1];
    crd[2] = coords[index * 3 + 2];

    // compute fractional coords
    scalar_t fx = dot_vec3(s_box_inv, crd);
    scalar_t fy = dot_vec3(s_box_inv+3, crd);
    scalar_t fz = dot_vec3(s_box_inv+6, crd);

    // shift to [0, 1]
    fx -= floor(fx);
    fy -= floor(fy);
    fz -= floor(fz);

    // compute cell index
    int32_t cx = (int32_t)(fx / fcrx) % ncx;
    int32_t cy = (int32_t)(fy / fcry) % ncy;
    int32_t cz = (int32_t)(fz / fcrz) % ncz;
    int32_t c = (cx * ncy + cy) * ncz + cz;

    f_coords[index*3]   = fx;
    f_coords[index*3+1] = fy;
    f_coords[index*3+2] = fz;

    cell_indices[index] = c;
    atomicAdd(&natoms_per_cell[c+1], 1);
}


__global__ void compute_cell_prefix(int32_t* sorted_cell_indices, int32_t natoms, int32_t* cell_prefix) {
    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index >= natoms || index == 0 ) {
        return;
    }

    int32_t prev_c = sorted_cell_indices[index - 1];
    int32_t c = sorted_cell_indices[index];
    if ( prev_c != c ) {
        cell_prefix[c] = index;
    }

}