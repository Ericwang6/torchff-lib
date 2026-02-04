#pragma once

#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_sum(T v, T* __restrict__ out) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be >= 32");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ T warp_sums[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);

    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        T v = (lane < NUM_WARPS) ? warp_sums[lane] : T(0);
        v = warp_reduce_sum(v);
        if (lane == 0) { atomicAdd(out, v); }
    }
}
