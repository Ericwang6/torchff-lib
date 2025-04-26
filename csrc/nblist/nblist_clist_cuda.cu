#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"


template <typename scalar_t> 
__global__ void assign_cell_index_kernel(
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


template <typename scalar_t>
__global__ void build_neighbor_list_cell_list_kernel(
    scalar_t* f_coords_sorted, // fractional coordinates sorted by cell index
    scalar_t* box,
    scalar_t cutoff2,
    scalar_t fcrx, scalar_t fcry, scalar_t fcrz,
    int32_t ncx, int32_t ncy, int32_t ncz, // number of cells in each dimension
    int32_t ncr, // number of cells to search in each dimension
    int32_t* sorted_atom_indices, // sorted_atom_indices[i] is the original index of i-th position in f_coords_sorted
    int32_t* cell_prefix,
    int32_t natoms,
    int32_t max_npairs,
    int32_t* pairs,
    int32_t* npairs
)
{
    __shared__ scalar_t s_box[9];
    if ( threadIdx.x < 9 ) {
        s_box[threadIdx.x] = box[threadIdx.x];
    }
    __syncthreads();

    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if ( index >= natoms ) {
        return;
    }

    scalar_t fcrd_i[3];
    fcrd_i[0] = f_coords_sorted[index*3];
    fcrd_i[1] = f_coords_sorted[index*3+1];
    fcrd_i[2] = f_coords_sorted[index*3+2];
    int32_t i = sorted_atom_indices[index];

    int32_t cx = (int32_t)(fcrd_i[0] / fcrx) % ncx;
    int32_t cy = (int32_t)(fcrd_i[1] / fcry) % ncy;
    int32_t cz = (int32_t)(fcrd_i[2] / fcrz) % ncz;

    scalar_t fcrd_j[3];
    scalar_t dfcrd[3];
    scalar_t dcrd[3];

    scalar_t tmp[3];
    int32_t nei_cx, nei_cy, nei_cz, nei_c, i_curr_pair;
    // Loop over neighbor cells
    for (int32_t dcx = -ncr; dcx <= ncr; ++dcx) {
        nei_cx = (cx + dcx + ncx) % ncx;
        for (int32_t dcy = -ncr; dcy <= ncr; ++dcy) {
            nei_cy = (cy + dcy + ncy) % ncy;
            for (int32_t dcz = -ncr; dcz <= ncr; ++dcz) {
                nei_cz = (cz + dcz + ncz) % ncz;
                nei_c = (nei_cx * ncy + nei_cy) * ncz + nei_cz;
                for (int32_t j_sort = cell_prefix[nei_c]; j_sort < cell_prefix[nei_c+1]; ++j_sort) {
                    if ( index > j_sort ) {
                        fcrd_j[0] = f_coords_sorted[j_sort*3];
                        fcrd_j[1] = f_coords_sorted[j_sort*3+1];
                        fcrd_j[2] = f_coords_sorted[j_sort*3+2];
                        diff_vec3(fcrd_i, fcrd_j, dfcrd);
                        dfcrd[0] -= round(dfcrd[0]);
                        dfcrd[1] -= round(dfcrd[1]);
                        dfcrd[2] -= round(dfcrd[2]);

                        dcrd[0] = dot_vec3(dfcrd, s_box);
                        dcrd[1] = dot_vec3(dfcrd, s_box+3);
                        dcrd[2] = dot_vec3(dfcrd, s_box+6);

                        if ( (dcrd[0] * dcrd[0] + dcrd[1] * dcrd[1] + dcrd[2] * dcrd[2]) <= cutoff2 ) {
                            i_curr_pair = atomicAdd(npairs, 1) % max_npairs;
                            pairs[i_curr_pair*2] = i;
                            pairs[i_curr_pair*2+1] = sorted_atom_indices[j_sort];
                        }
                    }
                }
            }
        }
    }
}


at::Tensor build_neighbor_list_cell_list_cuda(
    const at::Tensor& coords,
    const at::Tensor& box,
    const at::Scalar& cutoff,
    const at::Scalar& max_npairs,
    const at::Scalar& cell_size,
    bool padding
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);

    int32_t max_npairs_ = max_npairs.toInt();
    max_npairs_ = ( max_npairs_ == -1 ) ? natoms * (natoms - 1) / 2 : max_npairs_;

    at::Tensor box_cpu = box.to(at::kCPU);
    at::Tensor box_len = at::linalg_norm(box_cpu, 2, 0);
    at::Tensor f_cell_size = cell_size / box_len;
    at::Tensor nc = at::floor(box_len / cell_size).to(at::kInt);

    int32_t ncx = nc[0].item<int32_t>();
    int32_t ncy = nc[1].item<int32_t>();
    int32_t ncz = nc[2].item<int32_t>();
    int32_t ncr = ceilf(cutoff.toFloat() / cell_size.toFloat());

    TORCH_CHECK(ncx > 2 * ncr, "Box is too small in dimension x");
    TORCH_CHECK(ncy > 2 * ncr, "Box is too small in dimension y");
    TORCH_CHECK(ncz > 2 * ncr, "Box is too small in dimension z");


    int block_dim = 128;
    int grid_dim = (natoms + block_dim - 1) / block_dim;

    at::Tensor pairs = at::empty({max_npairs_, 2}, coords.options().dtype(at::kInt));
    at::Tensor npairs = at::zeros({1}, coords.options().dtype(at::kInt));

    at::Tensor f_coords = at::empty_like(coords);
    at::Tensor cell_indices = at::empty({natoms}, pairs.options());
    at::Tensor natoms_per_cell = at::zeros({ncx*ncy*ncz+1}, pairs.options());

    at::Tensor sorted_cell_indices;
    at::Tensor sorted_atom_indices;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Compute fractional coords and assign cell index for each atom
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "assign_cell_index", ([&] {
        scalar_t* fcr = f_cell_size.data_ptr<scalar_t>();
        scalar_t fcrx = fcr[0];
        scalar_t fcry = fcr[1];
        scalar_t fcrz = fcr[2];
        assign_cell_index_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            fcrx, fcry, fcrz,
            ncx, ncy, ncz,
            natoms,
            f_coords.data_ptr<scalar_t>(),
            cell_indices.data_ptr<int32_t>(),
            natoms_per_cell.data_ptr<int32_t>()
        );
    }));

    // Step 2: Sort atoms according to cell indices
    std::tie(sorted_cell_indices, sorted_atom_indices) = at::sort(cell_indices);
    at::Tensor f_coords_sorted = f_coords.index_select(0, sorted_atom_indices);

    // Step 3: Compute prefix (cumsum of number of atoms in each cell)
    at::Tensor cell_prefix = at::cumsum(natoms_per_cell, 0).to(at::kInt);
    // std::cout << "Cell prefix:" << cell_prefix << std::endl; 
    
    // Step 4: Do neighbor list search
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "build_neighbor_list", ([&] {
        scalar_t* fcr = f_cell_size.data_ptr<scalar_t>();
        scalar_t fcrx = fcr[0];
        scalar_t fcry = fcr[1];
        scalar_t fcrz = fcr[2];
        scalar_t cutoff2 = static_cast<scalar_t>(cutoff.toDouble() * cutoff.toDouble());
        build_neighbor_list_cell_list_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            f_coords_sorted.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            cutoff2,
            fcrx, fcry, fcrz,
            ncx, ncy, ncz,
            ncr,
            sorted_atom_indices.to(at::kInt).data_ptr<int32_t>(),
            cell_prefix.data_ptr<int32_t>(),
            natoms,
            max_npairs_,
            pairs.data_ptr<int32_t>(),
            npairs.data_ptr<int32_t>()
        );
    }));
    
    if ( padding ) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        // check if the number of pairs exceeds the capacity
        int32_t npairs_found = npairs[0].item<int32_t>();
        TORCH_CHECK(npairs_found <= max_npairs_, "Too many neighbor pairs found. Maximum is " + std::to_string(max_npairs_), " but found " + std::to_string(npairs_found));
        return pairs.index({at::indexing::Slice(0, npairs_found), at::indexing::Slice()});

    }
    else {
        return pairs;
    }

}


TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_neighbor_list_cell_list", build_neighbor_list_cell_list_cuda);
}