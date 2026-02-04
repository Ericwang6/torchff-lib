#include "common/constants.cuh"
#include "common/reduce.cuh"

template <typename T, int BLOCK_SIZE = 256>
__global__ void compute_self_contribution_kernel_rank_0(
    const T* __restrict__ q, // (N)
    int64_t N,
    const T alpha,
    T* __restrict__ energy,
    T* __restrict__ epot
) {
    constexpr T INV_ROOT_PI = inv_root_pi<T>();
    const T a_over_rpi = alpha * INV_ROOT_PI;
    T e = 0;
    for (int index = threadIdx.x+blockIdx.x*blockDim.x; index < N; index += blockDim.x*gridDim.x) {
        const T qi = q[index];
        e -= a_over_rpi * qi * qi;
        epot[index] -= a_over_rpi * qi * 2.0;
    }
    // reduce sum
    block_reduce_sum<T, BLOCK_SIZE>(e, energy);
}

template <typename T, int BLOCK_SIZE = 256>
__global__ void compute_self_contribution_kernel_rank_1(
    const T* __restrict__ q, // (N)
    const T* __restrict__ p, // (N,3)
    int64_t N,
    const T alpha,
    T* __restrict__ energy,
    T* __restrict__ epot,
    T* __restrict__ efield
) {
    constexpr T INV_ROOT_PI = inv_root_pi<T>();
    const T a_over_rpi = alpha * INV_ROOT_PI;
    const T alpha2 = alpha * alpha;
    const T pref_fld = a_over_rpi * (4.0 * alpha2 / 3.0);
    
    T e = 0;
    for (int index = threadIdx.x+blockIdx.x*blockDim.x; index < N; index += blockDim.x*gridDim.x) {
        const T qi = q[index];
        const T px = p[3*index + 0];
        const T py = p[3*index + 1];
        const T pz = p[3*index + 2];

        // Self-energy: -a_over_rpi * (q^2 + 2*alpha^2/3 * p^2)
        e -= a_over_rpi * qi*qi + pref_fld * (px*px + py*py + pz*pz) / 2;
        epot[index] -= a_over_rpi * qi * 2.0;
        efield[3*index + 0] += pref_fld * px;
        efield[3*index + 1] += pref_fld * py;
        efield[3*index + 2] += pref_fld * pz;
    }
    // reduce sum
    block_reduce_sum<T, BLOCK_SIZE>(e, energy);
}


template <typename T, int BLOCK_SIZE = 256>
__global__ void compute_self_contribution_kernel_rank_2(
    const T* __restrict__ q, // (N)
    const T* __restrict__ p, // (N,3)
    const T* __restrict__ Q, // (N,9)
    int64_t N,
    const T alpha,
    T* __restrict__ energy,
    T* __restrict__ epot,
    T* __restrict__ efield,
    T* __restrict__ efield_grad
) {
    constexpr T INV_ROOT_PI = inv_root_pi<T>();
    const T a_over_rpi = alpha * INV_ROOT_PI;
    const T alpha2 = alpha * alpha;
    const T pref_fld = a_over_rpi * (4.0 * alpha2 / 3.0);
    const T pref_fg = a_over_rpi * (16.0 * alpha2 * alpha2 / 15.0);
    
    T e = 0;
    for (int index = threadIdx.x+blockIdx.x*BLOCK_SIZE; index < N; index += BLOCK_SIZE*gridDim.x) {
        const T qi = q[index];
        const T px = p[3*index + 0];
        const T py = p[3*index + 1];
        const T pz = p[3*index + 2];
        const T p2 = px*px + py*py + pz*pz;
        
        const T* Qi = &Q[9*index];
        // Q is stored as [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        // Trace of Q^2: sum of squared elements
        const T Qxx = Qi[0], Qxy = Qi[1], Qxz = Qi[2];
        const T Qyx = Qi[3], Qyy = Qi[4], Qyz = Qi[5];
        const T Qzx = Qi[6], Qzy = Qi[7], Qzz = Qi[8];
        const T Q2 = Qxx*Qxx + Qxy*Qxy + Qxz*Qxz +
                          Qyx*Qyx + Qyy*Qyy + Qyz*Qyz +
                          Qzx*Qzx + Qzy*Qzy + Qzz*Qzz;
        
        // Self-energy: -a_over_rpi * (q^2 + 2*alpha^2/3 * p^2 + 8*alpha^4/45 * trace(Q^2))
        e -= a_over_rpi * qi*qi + pref_fld * p2 / T(2.0) + pref_fg * Q2 / T(6.0);
        
        epot[index] -= a_over_rpi * qi * 2.0;
        efield[3*index + 0] += pref_fld * px;
        efield[3*index + 1] += pref_fld * py;
        efield[3*index + 2] += pref_fld * pz;
        efield_grad[9*index + 0] += pref_fg * Qxx;
        efield_grad[9*index + 1] += pref_fg * Qxy;
        efield_grad[9*index + 2] += pref_fg * Qxz;
        efield_grad[9*index + 3] += pref_fg * Qyx;
        efield_grad[9*index + 4] += pref_fg * Qyy;
        efield_grad[9*index + 5] += pref_fg * Qyz;
        efield_grad[9*index + 6] += pref_fg * Qzx;
        efield_grad[9*index + 7] += pref_fg * Qzy;
        efield_grad[9*index + 8] += pref_fg * Qzz;
    }
    // reduce sum
    block_reduce_sum<T, BLOCK_SIZE>(e, energy);
}