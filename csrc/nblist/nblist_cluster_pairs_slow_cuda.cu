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


__device__ __forceinline__ void unravel_3d(int32_t c, int32_t nx, int32_t ny, int32_t nz, int32_t& x, int32_t& y, int32_t& z) {
    x = c / (ny * nz);
    y = c / nz - x * ny;
    z = c % nz;
}

__device__ __forceinline__ int32_t warp_dist(int32_t a, int32_t b, int32_t n) {
    int32_t d = abs(a - b);
    return min(d, n - d);
}

__device__ __forceinline__ bool is_interact(int32_t start_cidx_i, int32_t end_cidx_i, int32_t start_cidx_j, int32_t end_cidx_j, int32_t ncx, int32_t ncy, int32_t ncz, int32_t ncr) {
    int32_t cx_i, cy_i, cz_i;
    int32_t cx_j, cy_j, cz_j;
    bool interact = false;

    for (int32_t ci = start_cidx_i; ci <= end_cidx_i; ++ci) {
        if ( interact ) { break; }
        unravel_3d(ci, ncx, ncy, ncz, cx_i, cy_i, cz_i);
        for (int32_t cj = start_cidx_j; cj <= end_cidx_j; ++cj) {
            unravel_3d(cj, ncx, ncy, ncz, cx_j, cy_j, cz_j);
            interact = warp_dist(cx_i, cx_j, ncx) <= ncr && warp_dist(cy_i, cy_j, ncy) <= ncr && warp_dist(cz_i, cz_j, ncz) <= ncr;
            if ( interact ) { break; }
        }
    }
    return interact;
}

__device__ __forceinline__ bool in_list(int32_t x, int32_t* list, int32_t n) {
    if ( !list ) {
        return false;
    }
    bool in = false;
    for (int pos = 0; pos < n; ++pos) {
        if ( x == list[pos] ) {
            in = true;
            break;
        }
    }
    return in;
}


template <typename scalar_t> 
__global__ void assign_cell_index_kernel(
    scalar_t* coords,
    scalar_t* box_inv,
    scalar_t fcrx, scalar_t fcry, scalar_t fcrz, // cell size in fractional coords
    int32_t ncx, int32_t ncy, int32_t ncz, // number of cells in one dimension
    // scalar_t* fcr,
    // int32_t* nc,
    int32_t natoms,
    int32_t* cell_indices
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

    // scalar_t fcrx = fcr[0];
    // scalar_t fcry = fcr[1];
    // scalar_t fcrz = fcr[2];

    // int32_t ncx = nc[0];
    // int32_t ncy = nc[1];
    // int32_t ncz = nc[2];

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

    cell_indices[index] = c;
}


template <typename scalar_t>
__global__ void find_interacting_clusters_kernel(
    int32_t* sorted_cell_indices,
    int32_t num_clusters,
    int32_t ncx, int32_t ncy, int32_t ncz,
    int32_t ncr,
    int32_t* interacting_clusters,
    int32_t* num_interacting_clusters,
    int32_t max_num_interacting_clusters
)
{
    int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index >= num_clusters * (num_clusters + 1) / 2 ) {
        return;
    }
    int32_t i = (int32_t)floor((sqrt(((double)index) * 8 + 1) - 1) / 2);
    // if (i * (i - 1) > 2 * index) i--;
    int32_t j = index - (i * (i + 1)) / 2;

    int32_t n;

    if ( i == j ) {
        n = atomicAdd(num_interacting_clusters, 1) % max_num_interacting_clusters;
        interacting_clusters[n * 2] = i;
        interacting_clusters[n * 2 + 1] = j;
    }
    else {
        int32_t start_cidx_i = sorted_cell_indices[i * 32];
        int32_t start_cidx_j = sorted_cell_indices[j * 32];
        int32_t end_cidx_i = (i == num_clusters - 1) ? ncx * ncy * ncz - 1: sorted_cell_indices[i * 32 + 31];
        int32_t end_cidx_j = (j == num_clusters - 1) ? ncx * ncy * ncz - 1: sorted_cell_indices[j * 32 + 31];
        if ( is_interact(start_cidx_i, end_cidx_i, start_cidx_j, end_cidx_j, ncx, ncy, ncz, ncr) ) {
            n = atomicAdd(num_interacting_clusters, 1) % max_num_interacting_clusters;
            interacting_clusters[n * 2] = i;
            interacting_clusters[n * 2 + 1] = j;
        }
    }
}


template <typename scalar_t>
__global__ void set_bitmask_exclusions_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t cutoff2,
    int32_t* sorted_atom_indices,
    int32_t* interacting_clusters,
    int32_t num_interacting_clusters,
    int32_t natoms,
    int32_t* exclusions,
    int32_t max_exclusions_per_atom,
    uint32_t* bitmask_exclusions
)
{
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];

    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }

    __syncthreads();

    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    for (int32_t pos = warpIdx; pos < num_interacting_clusters; pos += totalWarps) {
        int32_t x = interacting_clusters[pos * 2];
        int32_t y = interacting_clusters[pos * 2 + 1];

        if ( x < 0 || y < 0 ) {
            continue;
        }

        int32_t index_i = x * warpSize + idxInWarp;
        int32_t index_j = y * warpSize + idxInWarp;
        int32_t i = -1;
        int32_t j = -1;
        int32_t j_shfl = -1;

        scalar_t crd_i[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j_shfl[3] = {0.0, 0.0, 0.0};

        int32_t* i_excl = nullptr;
        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            i_excl = exclusions + i * max_exclusions_per_atom;
            crd_i[0] = coords[i*3];
            crd_i[1] = coords[i*3+1];
            crd_i[2] = coords[i*3+2];
        }
        if ( index_j < natoms ) {
            j = sorted_atom_indices[index_j];
            crd_j[0] = coords[j*3];
            crd_j[1] = coords[j*3+1];
            crd_j[2] = coords[j*3+2];
        }

        uint32_t excl = 0;
        for ( int srcLane = 0; srcLane < warpSize; ++srcLane ) {
            j_shfl = __shfl_sync(0xFFFFFFFFu, j, srcLane);
            crd_j_shfl[0] = __shfl_sync(0xFFFFFFFFu, crd_j[0], srcLane);
            crd_j_shfl[1] = __shfl_sync(0xFFFFFFFFu, crd_j[1], srcLane);
            crd_j_shfl[2] = __shfl_sync(0xFFFFFFFFu, crd_j[2], srcLane);
            scalar_t rij[3];
            diff_vec3(crd_i, crd_j_shfl, rij);
            apply_pbc_triclinic(rij, box, box_inv, rij);
            scalar_t r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
            if ( i == -1 || j_shfl == -1 || in_list(j_shfl, i_excl, max_exclusions_per_atom) || r2 > cutoff2 ) {
                excl |= (1u << srcLane);
            }
        }

        // for the same cluster, count only i-j pair (not j-i) to avoid double counting
        if ( x == y ) {
            for ( int srcLane = 0; srcLane <= idxInWarp; ++srcLane ) {
                excl |= (1u << srcLane);
            }
        }
        bitmask_exclusions[pos * warpSize + idxInWarp] = excl;

        // if all-excluded, discard this intearcting cluster pair
        int count = __popc(excl);
        bool allexcl = __reduce_add_sync(0xFFFFFFFFu,count) == 1024;
        if ( allexcl && idxInWarp == 0 ) {
            interacting_clusters[pos * 2] = -1;
            interacting_clusters[pos * 2 + 1] = -1;
        }
    }
}


__global__ void decode_cluster_pairs_kernel(
    int32_t* sorted_atom_indices,
    int32_t* interacting_clusters,
    int32_t num_interacting_clusters,
    uint32_t* exclusions,
    int32_t natoms,
    int32_t max_npairs,
    int32_t* pairs,
    int32_t* npairs
)
{

    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    for (int32_t pos = warpIdx; pos < num_interacting_clusters; pos += totalWarps) {
        int32_t x = interacting_clusters[pos * 2];
        int32_t y = interacting_clusters[pos * 2 + 1];
        if ( x < 0 || y < 0 ) {
            continue;
        }
        int32_t index_i = x * warpSize + idxInWarp;
        int32_t index_j = y * warpSize + idxInWarp;
        int32_t i = -1;
        int32_t j = -1;

        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
        }
        if ( index_j < natoms ) {
            j = sorted_atom_indices[index_j];
        }

        uint32_t excl = exclusions[pos * warpSize + idxInWarp];
        for (int32_t srcLane = 0; srcLane < warpSize; ++srcLane) {
            int32_t j_shfl = __shfl_sync(0xFFFFFFFFu, j, srcLane);
            if ( !(excl & 0x1) ) {
                int32_t i_curr_pair = atomicAdd(npairs, 1) % max_npairs;
                pairs[i_curr_pair * 2] = i;
                pairs[i_curr_pair * 2 + 1] = j_shfl;
            }
            excl >>= 1;
        }
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Scalar cutoff,
    at::Tensor& exclusions,
    at::Scalar cell_size,
    at::Scalar max_num_interacting_clusters
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);
    
    at::Tensor box_cpu = box.to(at::kCPU);
    at::Tensor box_len = at::linalg_norm(box_cpu, 2, 0);
    at::Tensor f_cell_size = cell_size / box_len;
    at::Tensor nc = at::floor(box_len / cell_size).to(at::kInt);

    int32_t ncx = nc[0].item<int32_t>();
    int32_t ncy = nc[1].item<int32_t>();
    int32_t ncz = nc[2].item<int32_t>();
    int32_t ncr = (int32_t)ceilf(cutoff.toFloat() / cell_size.toFloat());

    at::Tensor cell_indices = at::empty({natoms}, coords.options().dtype(at::kInt));
    
    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Assign cell index for each atom
    int block_dim = 128;
    int grid_dim = (natoms + block_dim - 1) / block_dim;
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
            cell_indices.data_ptr<int32_t>()
        );
    }));

    // Step 2: Sort atoms according to cell indices
    at::Tensor sorted_cell_indices;
    at::Tensor sorted_atom_indices_long;
    std::tie(sorted_cell_indices, sorted_atom_indices_long) = at::sort(cell_indices);
    at::Tensor sorted_atom_indices = sorted_atom_indices_long.to(at::kInt);

    // Step 3: Find interacting cluster pairs, each cluster contains 32 atoms
    int32_t num_clusters = (natoms + 31) / 32;
    grid_dim = (num_clusters * (num_clusters + 1) / 2 + block_dim - 1) / block_dim;
    at::Tensor num_interacting_clusters = at::zeros({1}, coords.options().dtype(at::kInt));
    
    int32_t max_num_interacting_clusters_ = num_clusters * (num_clusters + 1) / 2;
    if ( max_num_interacting_clusters.toInt() > 0 ) {
        max_num_interacting_clusters_ = std::min(max_num_interacting_clusters_, max_num_interacting_clusters.toInt());
    }
    at::Tensor interacting_clusters = at::full({max_num_interacting_clusters_, 2}, -1, coords.options().dtype(at::kInt));
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "find_interacting_clusters", ([&] {
        find_interacting_clusters_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            sorted_cell_indices.data_ptr<int32_t>(),
            num_clusters,
            ncx, ncy, ncz,
            ncr,
            interacting_clusters.data_ptr<int32_t>(),
            num_interacting_clusters.data_ptr<int32_t>(),
            max_num_interacting_clusters_
        );
    }));

    // Step 4: Set bitmask exclusions for interacting cluster pairs
    int32_t max_exclusions_per_atom = exclusions.size(1);
    at::Tensor bitmask_exclusions = at::empty({max_num_interacting_clusters_, 32}, coords.options().dtype(at::kUInt32));
    block_dim = 32;
    grid_dim = max_num_interacting_clusters_;
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "set_bitmask_exclusions", ([&] {
        set_bitmask_exclusions_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff.toDouble()*cutoff.toDouble()),
            sorted_atom_indices.data_ptr<int32_t>(),
            interacting_clusters.data_ptr<int32_t>(),
            num_interacting_clusters.item().toInt(),
            natoms,            
            exclusions.data_ptr<int32_t>(),
            max_exclusions_per_atom,
            bitmask_exclusions.data_ptr<uint32_t>()
        );
    }));
    return std::make_tuple(
        sorted_atom_indices,
        interacting_clusters,
        bitmask_exclusions,
        num_interacting_clusters
    );
}


std::tuple<at::Tensor, at::Tensor> decode_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& sorted_atom_indices,
    at::Tensor& interacting_clusters,
    at::Tensor& bitmask_exclusions,
    at::Scalar cutoff,
    at::Scalar max_npairs,
    at::Scalar num_interacting_clusters,
    bool padding
)
{
    at::Tensor box_inv = at::linalg_inv(box);
    int32_t natoms = coords.size(0);
    int32_t max_npairs_ = max_npairs.toInt();
    max_npairs_ = ( max_npairs_ == -1 ) ? natoms * (natoms - 1) / 2 : max_npairs_;
    at::Tensor pairs = at::empty({max_npairs_, 2}, coords.options().dtype(at::kInt));
    at::Tensor npairs = at::zeros({1}, coords.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream();
    int block_dim = 32; // This has to be 32, which is the warp size
    int grid_dim = (num_interacting_clusters.toInt() > 0) ? num_interacting_clusters.toInt(): interacting_clusters.size(0);
    
    decode_cluster_pairs_kernel<<<grid_dim, block_dim, 0, stream>>>(
        sorted_atom_indices.data_ptr<int32_t>(),
        interacting_clusters.data_ptr<int32_t>(),
        num_interacting_clusters.toInt(),
        bitmask_exclusions.data_ptr<uint32_t>(),
        natoms,
        max_npairs_,
        pairs.data_ptr<int32_t>(),
        npairs.data_ptr<int32_t>()
    );

    if ( !padding ) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        // check if the number of pairs exceeds the capacity
        int32_t npairs_found = npairs[0].item<int32_t>();
        TORCH_CHECK(npairs_found <= max_npairs_, "Too many neighbor pairs found. Maximum is " + std::to_string(max_npairs_), " but found " + std::to_string(npairs_found));
        return std::make_tuple(pairs.index({at::indexing::Slice(0, npairs_found), at::indexing::Slice()}), npairs);

    }
    else {
        return std::make_tuple(pairs, npairs);
    }
}

TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("build_cluster_pairs", build_cluster_pairs_cuda);
    m.impl("decode_cluster_pairs", decode_cluster_pairs_cuda);
}