#pragma once
#include <cuda_runtime.h>
#include "common/vec3.cuh"
#include "common/constants.cuh"
#include "common/pbc.cuh"
#include "self_contribution.cuh"
#include "smem.cuh"


// ============================================================================
// KERNEL 1: PREPARE K-CONSTANTS (HKL + KROW_OPS)
// ============================================================================
template <typename T>
__global__ void prepare_k_constants_kernel(
    int64_t kmax, T alpha,
    const T* __restrict__ box, // (3,3)
    T* __restrict__ kvecs         // (4,M1) out (kx,ky,kz,weight)
)
{
    __shared__ T recip[9];

    if (threadIdx.x == 0) {
        // Compute reciprocal box matrix once per block
        invert_box_3x3(box, recip);
    }
    __syncthreads();

    const int64_t L = 2 * kmax + 1;
    const int64_t M1 = kmax * L * L;
    const int64_t M2 = kmax * L;
    const int64_t M  = M1 + M2 + kmax;

    int64_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    int64_t kx, ky, kz;

    if (m < M1) {
        int64_t t = m;
        kx = t / (L * L) + 1;
        int64_t r = t - (kx - 1) * (L * L);
        ky = r / L - kmax;
        kz = r - (r / L) * L - kmax;
    } else if (m < M1 + M2) {
        int64_t t = m - M1;
        kx = 0;
        ky = t / L + 1;
        kz = t - (t / L) * L - kmax;
    } else {
        int64_t t = m - (M1 + M2);
        kx = 0; ky = 0; kz = t + 1;
    }

    // 3. Compute K-Vector (k = recip * hkl)
    const T kvec_x = recip[0]*kx + recip[3]*ky + recip[6]*kz;
    const T kvec_y = recip[1]*kx + recip[4]*ky + recip[7]*kz;
    const T kvec_z = recip[2]*kx + recip[5]*ky + recip[8]*kz;

    // 4. Compute Gaussian and Symmetry weights;
    const T k2v = kvec_x*kvec_x + kvec_y*kvec_y + kvec_z*kvec_z;
    const T gau = exp_(-(pi_squared<T>() * k2v) / (alpha*alpha)) / k2v;

    // 5. Store k-vector and weights;
    kvecs[0*M + m] = kvec_x;
    kvecs[1*M + m] = kvec_y;
    kvecs[2*M + m] = kvec_z;
    kvecs[3*M + m] = gau * T(2.0);
}


// Ewald forward kernel: compute structure factors (used for backward and energy)
template <typename T, int64_t RANK = 0, int64_t BLOCK_SIZE = 256>
__global__ void ewald_forward_kernel(
    const T* __restrict__ kvec,                // (4, M)
    const T* __restrict__ coords,              // (N, 3)
    const T* __restrict__ q,                   // (N)
    const T* __restrict__ p,                   // (N, 3) or null
    const T* __restrict__ t,                   // (N, 9) or null
    int64_t M, int64_t N,
    const T* __restrict__ box,                 // (3,3)
    T* __restrict__ Sreal,                     // (M) Out
    T* __restrict__ Simag,                     // (M) Out
    T* __restrict__ energy                       // (1) Out
)                    
{

    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "BLOCK_SIZE must be a power of two.");
    static_assert(BLOCK_SIZE <= 1024,
        "BLOCK_SIZE must be <= 1024 for CUDA thread blocks.");
    static_assert(RANK >= 0 && RANK <= 2,
        "RANK must be 0, 1, or 2.");

    // Compute box volume once per block
    __shared__ T s_volume;
    if (threadIdx.x == 0) {
        T tmp_inv[9];
        invert_box_3x3(box, tmp_inv, &s_volume);
    }
    __syncthreads();
    const T V = s_volume;

    const int k = blockIdx.x; // One block per k-vector

    const T kx = kvec[0*M + k];
    const T ky = kvec[1*M + k];
    const T kz = kvec[2*M + k];

    constexpr T TWOPI = two_pi<T>();
    
    T Sr = T(0);
    T Si = T(0);

    // Grid-stride loop over atoms (if N > blockDim)
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        
        // 1. Phase Calculation
        const T rx = coords[n*3+0];
        const T ry = coords[n*3+1];
        const T rz = coords[n*3+2];
        const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
        T s, c;
        sincos_(theta, &s, &c);

        // 2. Multipole Charge L_n(k)
        T Lr = q[n];
        
	   // Dipole term
        if constexpr ( RANK >= 1 ) {
            const T px = p[3*n + 0], py = p[3*n + 1], pz = p[3*n + 2];
            T Li = TWOPI * (kx*px + ky*py + kz*pz);
            // Quadrupole term
            if ( RANK >= 2 ) {
                const T* tn = &t[n*9];
                constexpr T TWOPI2O3 = TWOPI * TWOPI / T(3.0);
                Lr -= TWOPI2O3 * (tn[0]*kx*kx + (tn[1]+tn[3])*kx*ky + (tn[6]+tn[2])*kx*kz + tn[4]*ky*ky + (tn[7]+tn[5])*ky*kz + tn[8]*kz*kz);
            }
            // Complex Multiply: (Lr + iLi) * (c + is)
            Sr += (Lr*c - Li*s);
            Si += (Lr*s + Li*c);
        }
        else {
            Sr += Lr * c;
            Si += Lr * s;
        }
    }

    // Warp-level reduction to reduce synchronization overhead
    // Assumes full warp participation
    for (int offset = 16; offset > 0; offset >>= 1) {
        Sr += __shfl_down_sync(0xffffffff, Sr, offset);
        Si += __shfl_down_sync(0xffffffff, Si, offset);
    }

    // One partial sum per warp written to shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE >> 5;
    __shared__ T warp_r[NUM_WARPS];
    __shared__ T warp_i[NUM_WARPS];
    const int lane = threadIdx.x & 31;          // thread index within warp
    const int warp = threadIdx.x >> 5;          // warp index within block

    if (lane == 0) {
        warp_r[warp] = Sr;
        warp_i[warp] = Si;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        Sr = (lane < NUM_WARPS) ? warp_r[lane] : T(0);
        Si = (lane < NUM_WARPS) ? warp_i[lane] : T(0);

        for (int64_t offset = 16; offset > 0; offset >>= 1) {
            Sr += __shfl_down_sync(0xffffffff, Sr, offset);
            Si += __shfl_down_sync(0xffffffff, Si, offset);
        }

        if (lane == 0) {
            Sreal[k] = Sr; Simag[k] = Si;
            atomicAdd(energy, (Sr*Sr+Si*Si) * kvec[3*M+k] / TWOPI / V);
        }
        
    }
}


// ============================================================================
// KERNEL 3: MAIN
// ============================================================================
// Merges: Accumulation, Operator Forces, Energy, and Self-Terms.

template <typename T>
__device__ __forceinline__ void accum_rank_0(
    T q, 
    T rx, T ry, T rz, 
    T kx, T ky, T kz, T w,
    T Sr, T Si,
    T& ep, T& fx, T& fy, T& fz
)
{
    constexpr T TWOPI = two_pi<T>();
    const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
    T si, ci;
    sincos_(theta, &si, &ci);

    // Define eS = S e^{-i theta}
    // Re[eS] = Re[S e^{-i theta}] = Sr*c + Si*s
    // Im[eS] = Im[S e^{-i theta}] = -Sr*s + Si*c
    const T eSr = Sr*ci + Si*si;
    const T eSi = -Sr*si + Si*ci;

    // Potential
    ep += w * eSr;

    // Forces: L_i(k) = q (monopole only)
    const T Lr = q;
    // Formula: Im{ e^{-i theta} * conj(L) * S }
    // For rank 0: conj(L) = Lr, so eLSi = Im[(ci - i*si) * Lr * (Sr + i*Si)]
    const T eLSi = ci * Lr * Si - si * Lr * Sr;

    fx += w * eLSi * kx;
    fy += w * eLSi * ky;
    fz += w * eLSi * kz;
}


template <typename T>
__device__ __forceinline__ void accum_rank_1(
    T q,
    T px, T py, T pz,
    T rx, T ry, T rz,
    T kx, T ky, T kz, T w,
    T Sr, T Si,
    T& ep, T& efx, T& efy, T& efz, T& fx, T& fy, T& fz
)
{
    constexpr T TWOPI = two_pi<T>();
    const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
    T si, ci;
    sincos_(theta, &si, &ci);

    // Define eS = S e^{-i theta}
    // Re[eS] = Re[S e^{-i theta}] = Sr*c + Si*s
    // Im[eS] = Im[S e^{-i theta}] = -Sr*s + Si*c
    const T eSr = Sr*ci + Si*si;
    const T eSi = -Sr*si + Si*ci;

    // Potential
    ep += w * eSr;

    // Field
    efx += w * eSi * kx;
    efy += w * eSi * ky;
    efz += w * eSi * kz;

    // Forces: L_i(k) = q + i * TWOPI * (k·p)
    const T Lr = q;
    const T Li = TWOPI * (kx*px + ky*py + kz*pz);
    // Formula: Im{ e^{-i theta} * conj(L) * S }
    // conj(L) = Lr - i*Li
    // Term = (ci - i*si) * (Lr - i*Li) * (Sr + i*Si)
    const T eLSi = ci * (Lr*Si - Li*Sr) - si * (Lr*Sr + Li*Si);

    fx += w * eLSi * kx;
    fy += w * eLSi * ky;
    fz += w * eLSi * kz;
}


template <typename T>
__device__ __forceinline__ void accum_rank_2(
    T q,
    T px, T py, T pz,
    T qxx, T qxy, T qxz, T qyy, T qyz, T qzz,
    T rx, T ry, T rz,
    T kx, T ky, T kz, T w,
    T Sr, T Si,
    T& ep, T& efx, T& efy, T& efz,
    T& egxx, T& egxy, T& egxz, T& egyy, T& egyz, T& egzz,
    T& fx, T& fy, T& fz
)
{
    constexpr T TWOPI = two_pi<T>();
    constexpr T TWOPI2O3  = TWOPI * TWOPI / T(3.0);
    const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
    T si, ci;
    sincos_(theta, &si, &ci);

    // Define eS = S e^{-i theta}
    // Re[eS] = Re[S e^{-i theta}] = Sr*c + Si*s
    // Im[eS] = Im[S e^{-i theta}] = -Sr*s + Si*c
    const T eSr = Sr*ci + Si*si;
    const T eSi = -Sr*si + Si*ci;

    // Potential
    ep += w * eSr;

    // Field
    efx += w * eSi * kx;
    efy += w * eSi * ky;
    efz += w * eSi * kz;

    // Gradient
    const T rr = w * eSr;
    egxx += rr*kx*kx; egxy += rr*kx*ky; egxz += rr*kx*kz;
    egyy += rr*ky*ky; egyz += rr*ky*kz;
    egzz += rr*kz*kz;

    // Forces: L_i(k) = q + i * TWOPI * (k·p) - twopi^2/3 * (k^T Q k)
    const T Li = TWOPI * (kx*px + ky*py + kz*pz);
    
    // Quadrupole term: k^T Q k
    const T tkx = qxx*kx + qxy*ky + qxz*kz;
    const T tky = qxy*kx + qyy*ky + qyz*kz;  // qxy is symmetric
    const T tkz = qxz*kx + qyz*ky + qzz*kz;  // qxz, qyz are symmetric
    const T Lr = q - TWOPI2O3  * (kx*tkx + ky*tky + kz*tkz);
    
    // Formula: Im{ e^{-i theta} * conj(L) * S }
    // conj(L) = Lr - i*Li
    // Term = (ci - i*si) * (Lr - i*Li) * (Sr + i*Si)
    const T eLSi = ci * (Lr*Si - Li*Sr) - si * (Lr*Sr + Li*Si);

    fx += w * eLSi * kx;
    fy += w * eLSi * ky;
    fz += w * eLSi * kz;
}


template <typename T, int NUM_WARPS>
__device__ __forceinline__ void reduce_rank_0(
    T alpha, T V, T qi,
    T& ep, T& fx, T& fy, T& fz,
    Smem<T,0,NUM_WARPS>& smem,
    T* __restrict__ epot_out, T* __restrict__ forces_out
) {
    const int64_t lane = threadIdx.x & 31;  // thread index within warp
    const int64_t warp = threadIdx.x >> 5;  // warp index within block
    
    // Warp-level reduction
    for (int64_t offset = 16; offset > 0; offset >>= 1) {
        ep += __shfl_down_sync(0xffffffff, ep, offset);
        fx += __shfl_down_sync(0xffffffff, fx, offset);
        fy += __shfl_down_sync(0xffffffff, fy, offset);
        fz += __shfl_down_sync(0xffffffff, fz, offset);
    }

    // One partial sum per warp written to shared memory
    if (lane == 0) {
        smem.ep[warp] = ep;
        smem.fx[warp] = fx;
        smem.fy[warp] = fy;
        smem.fz[warp] = fz;   
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        T sum_ep = (lane < NUM_WARPS) ? smem.ep[lane] : T(0);
        T sum_fx = (lane < NUM_WARPS) ? smem.fx[lane] : T(0);
        T sum_fy = (lane < NUM_WARPS) ? smem.fy[lane] : T(0);
        T sum_fz = (lane < NUM_WARPS) ? smem.fz[lane] : T(0);

        unsigned mask = __activemask();
        for (int64_t offset = 16; offset > 0; offset >>= 1) {
            sum_ep += __shfl_down_sync(mask, sum_ep, offset);
            sum_fx += __shfl_down_sync(mask, sum_fx, offset);
            sum_fy += __shfl_down_sync(mask, sum_fy, offset);
            sum_fz += __shfl_down_sync(mask, sum_fz, offset);
        }

        if (lane == 0) {
            const T pref_ep = T(1) / V / pi<T>();
            const T pref_f = T(-2.0) / V;
            epot_out[0] += sum_ep * pref_ep;
            forces_out[0] += sum_fx * pref_f;
            forces_out[1] += sum_fy * pref_f;
            forces_out[2] += sum_fz * pref_f;
        }
    }
}


template <typename T, int NUM_WARPS>
__device__ __forceinline__ void reduce_rank_1(
    T alpha, T V, T qi,
    T px, T py, T pz,
    T& ep, T& efx, T& efy, T& efz, T& fx, T& fy, T& fz,
    Smem<T,1,NUM_WARPS>& smem,
    T* __restrict__ epot_out, T* __restrict__ efield_out, T* __restrict__ forces_out
) {
    const int64_t lane = threadIdx.x & 31;  // thread index within warp
    const int64_t warp = threadIdx.x >> 5;  // warp index within block
    
    // Warp-level reduction
    for (int64_t offset = 16; offset > 0; offset >>= 1) {
        ep += __shfl_down_sync(0xffffffff, ep, offset);
        efx += __shfl_down_sync(0xffffffff, efx, offset);
        efy += __shfl_down_sync(0xffffffff, efy, offset);
        efz += __shfl_down_sync(0xffffffff, efz, offset);
        fx += __shfl_down_sync(0xffffffff, fx, offset);
        fy += __shfl_down_sync(0xffffffff, fy, offset);
        fz += __shfl_down_sync(0xffffffff, fz, offset);
    }

    // One partial sum per warp written to shared memory
    if (lane == 0) {
        smem.ep[warp] = ep;
        smem.efx[warp] = efx;
        smem.efy[warp] = efy;
        smem.efz[warp] = efz;
        smem.fx[warp] = fx;
        smem.fy[warp] = fy;
        smem.fz[warp] = fz;   
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        T sum_ep = (lane < NUM_WARPS) ? smem.ep[lane] : T(0);
        T sum_efx = (lane < NUM_WARPS) ? smem.efx[lane] : T(0);
        T sum_efy = (lane < NUM_WARPS) ? smem.efy[lane] : T(0);
        T sum_efz = (lane < NUM_WARPS) ? smem.efz[lane] : T(0);
        T sum_fx = (lane < NUM_WARPS) ? smem.fx[lane] : T(0);
        T sum_fy = (lane < NUM_WARPS) ? smem.fy[lane] : T(0);
        T sum_fz = (lane < NUM_WARPS) ? smem.fz[lane] : T(0);

        unsigned mask = __activemask();
        for (int64_t offset = 16; offset > 0; offset >>= 1) {
            sum_ep += __shfl_down_sync(mask, sum_ep, offset);
            sum_efx += __shfl_down_sync(mask, sum_efx, offset);
            sum_efy += __shfl_down_sync(mask, sum_efy, offset);
            sum_efz += __shfl_down_sync(mask, sum_efz, offset);
            sum_fx += __shfl_down_sync(mask, sum_fx, offset);
            sum_fy += __shfl_down_sync(mask, sum_fy, offset);
            sum_fz += __shfl_down_sync(mask, sum_fz, offset);
        }

        if (lane == 0) {
            const T pref_ep = T(1) / V / pi<T>();
            const T pref_ef = -T(2) / V;
            const T pref_f = T(-2.0) / V;
            
            // Potential
            epot_out[0] += sum_ep * pref_ep;
            
            // Field
            efield_out[0] += sum_efx * pref_ef;
            efield_out[1] += sum_efy * pref_ef;
            efield_out[2] += sum_efz * pref_ef;
            
            // Forces
            forces_out[0] += sum_fx * pref_f;
            forces_out[1] += sum_fy * pref_f;
            forces_out[2] += sum_fz * pref_f;
        }
    }
}


template <typename T, int NUM_WARPS>
__device__ __forceinline__ void reduce_rank_2(
    T alpha, T V, T qi,
    T px, T py, T pz,
    T qxx, T qxy, T qxz, T qyy, T qyz, T qzz,
    T& ep, T& efx, T& efy, T& efz,
    T& egxx, T& egxy, T& egxz, T& egyy, T& egyz, T& egzz,
    T& fx, T& fy, T& fz,
    Smem<T,2,NUM_WARPS>& smem,
    T* __restrict__ epot_out, T* __restrict__ efield_out,
    T* __restrict__ efield_grad_out, T* __restrict__ forces_out
) {
    const int64_t lane = threadIdx.x & 31;  // thread index within warp
    const int64_t warp = threadIdx.x >> 5;  // warp index within block
    
    // Warp-level reduction
    for (int64_t offset = 16; offset > 0; offset >>= 1) {
        ep += __shfl_down_sync(0xffffffff, ep, offset);
        efx += __shfl_down_sync(0xffffffff, efx, offset);
        efy += __shfl_down_sync(0xffffffff, efy, offset);
        efz += __shfl_down_sync(0xffffffff, efz, offset);
        egxx += __shfl_down_sync(0xffffffff, egxx, offset);
        egxy += __shfl_down_sync(0xffffffff, egxy, offset);
        egxz += __shfl_down_sync(0xffffffff, egxz, offset);
        egyy += __shfl_down_sync(0xffffffff, egyy, offset);
        egyz += __shfl_down_sync(0xffffffff, egyz, offset);
        egzz += __shfl_down_sync(0xffffffff, egzz, offset);
        fx += __shfl_down_sync(0xffffffff, fx, offset);
        fy += __shfl_down_sync(0xffffffff, fy, offset);
        fz += __shfl_down_sync(0xffffffff, fz, offset);
    }

    // One partial sum per warp written to shared memory
    if (lane == 0) {
        smem.ep[warp] = ep;
        smem.efx[warp] = efx;
        smem.efy[warp] = efy;
        smem.efz[warp] = efz;
        smem.egxx[warp] = egxx;
        smem.egxy[warp] = egxy;
        smem.egxz[warp] = egxz;
        smem.egyy[warp] = egyy;
        smem.egyz[warp] = egyz;
        smem.egzz[warp] = egzz;
        smem.fx[warp] = fx;
        smem.fy[warp] = fy;
        smem.fz[warp] = fz;   
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        T sum_ep = (lane < NUM_WARPS) ? smem.ep[lane] : T(0);
        T sum_efx = (lane < NUM_WARPS) ? smem.efx[lane] : T(0);
        T sum_efy = (lane < NUM_WARPS) ? smem.efy[lane] : T(0);
        T sum_efz = (lane < NUM_WARPS) ? smem.efz[lane] : T(0);
        T sum_egxx = (lane < NUM_WARPS) ? smem.egxx[lane] : T(0);
        T sum_egxy = (lane < NUM_WARPS) ? smem.egxy[lane] : T(0);
        T sum_egxz = (lane < NUM_WARPS) ? smem.egxz[lane] : T(0);
        T sum_egyy = (lane < NUM_WARPS) ? smem.egyy[lane] : T(0);
        T sum_egyz = (lane < NUM_WARPS) ? smem.egyz[lane] : T(0);
        T sum_egzz = (lane < NUM_WARPS) ? smem.egzz[lane] : T(0);
        T sum_fx = (lane < NUM_WARPS) ? smem.fx[lane] : T(0);
        T sum_fy = (lane < NUM_WARPS) ? smem.fy[lane] : T(0);
        T sum_fz = (lane < NUM_WARPS) ? smem.fz[lane] : T(0);

        unsigned mask = __activemask();
        for (int64_t offset = 16; offset > 0; offset >>= 1) {
            sum_ep += __shfl_down_sync(mask, sum_ep, offset);
            sum_efx += __shfl_down_sync(mask, sum_efx, offset);
            sum_efy += __shfl_down_sync(mask, sum_efy, offset);
            sum_efz += __shfl_down_sync(mask, sum_efz, offset);
            sum_egxx += __shfl_down_sync(mask, sum_egxx, offset);
            sum_egxy += __shfl_down_sync(mask, sum_egxy, offset);
            sum_egxz += __shfl_down_sync(mask, sum_egxz, offset);
            sum_egyy += __shfl_down_sync(mask, sum_egyy, offset);
            sum_egyz += __shfl_down_sync(mask, sum_egyz, offset);
            sum_egzz += __shfl_down_sync(mask, sum_egzz, offset);
            sum_fx += __shfl_down_sync(mask, sum_fx, offset);
            sum_fy += __shfl_down_sync(mask, sum_fy, offset);
            sum_fz += __shfl_down_sync(mask, sum_fz, offset);
        }

        if (lane == 0) {
            const T pref_ep = T(1) / V / pi<T>();
            const T pref_ef = -T(2) / V;
            const T pref_eg = T(4) * pi<T>() / V;
            const T pref_f = T(-2.0) / V;
            
            // Potential
            epot_out[0] += sum_ep * pref_ep;
            
            // Field
            efield_out[0] += sum_efx * pref_ef;
            efield_out[1] += sum_efy * pref_ef;
            efield_out[2] += sum_efz * pref_ef;
            
            // Field Gradient
            efield_grad_out[0] += sum_egxx * pref_eg;
            efield_grad_out[1] += sum_egxy * pref_eg;
            efield_grad_out[2] += sum_egxz * pref_eg;
            efield_grad_out[3] += sum_egxy * pref_eg;
            efield_grad_out[4] += sum_egyy * pref_eg;
            efield_grad_out[5] += sum_egyz * pref_eg;
            efield_grad_out[6] += sum_egxz * pref_eg;
            efield_grad_out[7] += sum_egyz * pref_eg;
            efield_grad_out[8] += sum_egzz * pref_eg;
            
            // Forces
            forces_out[0] += sum_fx * pref_f;
            forces_out[1] += sum_fy * pref_f;
            forces_out[2] += sum_fz * pref_f;
        }
    }
}



template <typename T, int64_t RANK = 0, int64_t BLOCK_SIZE = 256>
__global__ void ewald_backward_kernel(
    const T* __restrict__ kvec,                   // (4,M1)
    const T* __restrict__ Sreal,                  // (M1)
    const T* __restrict__ Simag,                  // (M1)
    const T* __restrict__ coords,                 // (N,3)
    const T* __restrict__ q,                      // (N)
    const T* __restrict__ p,                      // (N,3)
    const T* __restrict__ Q,                      // (N,9)
    int64_t M, int64_t N, const T* __restrict__ box, T alpha,
    T* __restrict__ epot_out,                      // (N)
    T* __restrict__ efield_out,                    // (N,3)
    T* __restrict__ efield_grad_out,                     // (N,9)
    T* __restrict__ forces_out                    // (N,3)
)
{

    // TODO: add static assert to check things
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "BLOCK_SIZE must be a power of two.");
    static_assert(BLOCK_SIZE <= 1024,
        "BLOCK_SIZE must be <= 1024 for CUDA thread blocks.");
    static_assert(RANK >= 0 && RANK <= 2,
        "RANK must be 0, 1, or 2.");

    // Compute box volume once per block
    __shared__ T s_volume;
    if (threadIdx.x == 0) {
        T tmp_inv[9];
        invert_box_3x3(box, tmp_inv, &s_volume);
    }
    __syncthreads();
    const T V = s_volume;

    constexpr int NUM_WARPS = BLOCK_SIZE >> 5;
    __shared__ Smem<T, RANK, NUM_WARPS> smem;

    const int64_t i = blockIdx.x;
    
    // Load atom i
    const T ri_x = coords[i*3+0];
    const T ri_y = coords[i*3+1];
    const T ri_z = coords[i*3+2];
    T fx = 0, fy = 0, fz = 0;

    MultipoleAccumWithGrad<T, RANK> mp{};
    mp.c0 = q[i];
    if constexpr (RANK >= 1) { 
        mp.dx = p[3*i+0]; mp.dy = p[3*i+1]; mp.dz = p[3*i+2]; 
    }
    if constexpr (RANK >= 2) {
        mp.qxx = Q[9*i+0]; mp.qxy = Q[9*i+1]; mp.qxz = Q[9*i+2];
        mp.qyy = Q[9*i+4]; mp.qyz = Q[9*i+5]; mp.qzz = Q[9*i+8];
    }

    // --- K-VECTOR LOOP ---
    for (int64_t k = threadIdx.x; k < M; k+=blockDim.x) {
        const T kx = kvec[0*M + k]; const T ky = kvec[1*M + k]; const T kz = kvec[2*M + k];
        const T w  = kvec[3*M + k];
        
        const T Sr = Sreal[k]; const T Si = Simag[k];

        // Potential and Forces using accum_rank_X functions
        if constexpr (RANK == 0) {
            accum_rank_0<T>(
                mp.c0, ri_x, ri_y, ri_z,
                kx, ky, kz, w, Sr, Si,
                mp.ep, fx, fy, fz
            );
        } else if constexpr (RANK == 1) {
            accum_rank_1<T>(
                mp.c0, mp.dx, mp.dy, mp.dz, ri_x, ri_y, ri_z,
                kx, ky, kz, w, Sr, Si,
                mp.ep, mp.efx, mp.efy, mp.efz, fx, fy, fz
            );
        } else if constexpr (RANK == 2) {
            accum_rank_2<T>(
                mp.c0, mp.dx, mp.dy, mp.dz, mp.qxx, mp.qxy, mp.qxz, mp.qyy, mp.qyz, mp.qzz,
                ri_x, ri_y, ri_z,
                kx, ky, kz, w, Sr, Si,
                mp.ep, mp.efx, mp.efy, mp.efz, mp.egxx, mp.egxy, mp.egxz, mp.egyy, mp.egyz, mp.egzz,
                fx, fy, fz
            );
        }
    }

    // Reduction and output using reduce_rank_X functions
    if constexpr (RANK == 0) {
        reduce_rank_0<T, NUM_WARPS>(
            alpha, V, mp.c0,
            mp.ep, fx, fy, fz,
            smem,
            epot_out + i, forces_out + i*3
        );
    } else if constexpr (RANK == 1) {
        reduce_rank_1<T, NUM_WARPS>(
            alpha, V, mp.c0, mp.dx, mp.dy, mp.dz,
            mp.ep, mp.efx, mp.efy, mp.efz, fx, fy, fz,
            smem,
            epot_out + i, efield_out + i*3, forces_out + i*3
        );
    } else if constexpr (RANK == 2) {
        reduce_rank_2<T, NUM_WARPS>(
            alpha, V, mp.c0, mp.dx, mp.dy, mp.dz, mp.qxx, mp.qxy, mp.qxz, mp.qyy, mp.qyz, mp.qzz,
            mp.ep, mp.efx, mp.efy, mp.efz, mp.egxx, mp.egxy, mp.egxz, mp.egyy, mp.egyz, mp.egzz, fx, fy, fz,
            smem,
            epot_out + i, efield_out + i*3, efield_grad_out + i*9, forces_out + i*3
        );
    }
}


template <typename T, int64_t RANK = 0, int64_t BLOCK_SIZE = 256>
__global__ void ewald_fourier_gradient_kernel(
    const T* __restrict__ kvec,                
    const T* __restrict__ coords,
    const T* __restrict__ pot_grad,
    const T* __restrict__ field_grad,
    int64_t M, int64_t N, const T* __restrict__ box,
    T* __restrict__ Fdr, T* __restrict__ Fdi 
)                    
{

    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "BLOCK_SIZE must be a power of two.");
    static_assert(BLOCK_SIZE <= 1024,
        "BLOCK_SIZE must be <= 1024 for CUDA thread blocks.");
    static_assert(RANK >= 0 && RANK <= 2,
        "RANK must be 0, 1, or 2.");

    // Compute box volume once per block
    __shared__ T s_volume;
    if (threadIdx.x == 0) {
        T tmp_inv[9];
        invert_box_3x3(box, tmp_inv, &s_volume);
    }
    __syncthreads();
    const T boxV = s_volume;

    const int k = blockIdx.x; // One block per k-vector

    const T kx = kvec[0*M + k];
    const T ky = kvec[1*M + k];
    const T kz = kvec[2*M + k];

    constexpr T TWOPI = two_pi<T>();

    const T pref_dV = T(1) / boxV / pi<T>();
    const T pref_dE = T(2) / boxV;
    
    T Fdr_k = T(0);
    T Fdi_k = T(0);

    // Grid-stride loop over atoms (if N > blockDim)
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        
        // 1. Phase Calculation
        const T rx = coords[n*3+0];
        const T ry = coords[n*3+1];
        const T rz = coords[n*3+2];
        const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
        T s, c;
        sincos_(theta, &s, &c);

        T re = pref_dV * pot_grad[n];
        T im = pref_dE * (kx*field_grad[n*3+0] + ky*field_grad[n*3+1] + kz*field_grad[n*3+2]);

        // (re + i * im) * (c - i * s)
        Fdr_k += re * c + im * s;
        Fdi_k += im * c - re * s;
    }

    // Warp-level reduction to reduce synchronization overhead
    // Assumes full warp participation
    for (int offset = 16; offset > 0; offset >>= 1) {
        Fdr_k += __shfl_down_sync(0xffffffff, Fdr_k, offset);
        Fdi_k += __shfl_down_sync(0xffffffff, Fdi_k, offset);
    }

    // One partial sum per warp written to shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE >> 5;
    __shared__ T Fdr_warp[NUM_WARPS];
    __shared__ T Fdi_warp[NUM_WARPS];
    const int lane = threadIdx.x & 31;          // thread index within warp
    const int warp = threadIdx.x >> 5;          // warp index within block

    if (lane == 0) {
        Fdr_warp[warp] = Fdr_k;
        Fdi_warp[warp] = Fdi_k;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        Fdr_k = (lane < NUM_WARPS) ? Fdr_warp[lane] : T(0);
        Fdi_k = (lane < NUM_WARPS) ? Fdi_warp[lane] : T(0);
    
        for (int offset = 16; offset > 0; offset >>= 1) {
            Fdr_k += __shfl_down_sync(0xffffffff, Fdr_k, offset);
            Fdi_k += __shfl_down_sync(0xffffffff, Fdi_k, offset);
        }
    
        if (lane == 0) {
            Fdr[k] = Fdr_k;
            Fdi[k] = Fdi_k;
        }
    }
}



template <typename T, int64_t RANK = 0, int64_t BLOCK_SIZE = 256>
__global__ void ewald_backward_with_fields_kernel(
    const T* __restrict__ kvec,                   // (4,M1)
    const T* __restrict__ Fdr,                  // (M1)
    const T* __restrict__ Fdi,                  // (M1)
    const T* __restrict__ coords,                 // (N,3)
    const T* __restrict__ q,                      // (N)
    const T* __restrict__ p,                      // (N,3)
    const T* __restrict__ Q,                      // (N,9)
    const T* __restrict__ dV,
    const T* __restrict__ dE,
    const T* __restrict__ Sreal,
    const T* __restrict__ Simag,
    int64_t M, T alpha, const T* __restrict__ box,
    T* __restrict__ coords_grad_out,
    T* __restrict__ q_grad_out,                      // (N)
    T* __restrict__ p_grad_out,                    // (N,3)
    T* __restrict__ Q_grad_out                     // (N,9)
)
{

    // TODO: add static assert to check things
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "BLOCK_SIZE must be a power of two.");
    static_assert(BLOCK_SIZE <= 1024,
        "BLOCK_SIZE must be <= 1024 for CUDA thread blocks.");
    static_assert(RANK >= 0 && RANK <= 2,
        "RANK must be 0, 1, or 2.");

    constexpr T TWOPI = two_pi<T>();
    constexpr T TWOPI2O3 = TWOPI * TWOPI / T(3.0);
    constexpr int NUM_WARPS = BLOCK_SIZE >> 5;

    // Compute box volume once per block
    __shared__ T s_volume;
    if (threadIdx.x == 0) {
        T tmp_inv[9];
        invert_box_3x3(box, tmp_inv, &s_volume);
    }
    __syncthreads();
    const T V = s_volume;

    const int64_t i = blockIdx.x;
    
    // Load atom i
    const T rx = coords[i*3+0];
    const T ry = coords[i*3+1];
    const T rz = coords[i*3+2];

    MultipoleAccumWithGrad<T, RANK> mp{};
    mp.c0 = q[i];
    if constexpr (RANK >= 1) {
        mp.dx = p[i*3+0]; mp.dy = p[i*3+1]; mp.dz = p[i*3+2];
    }
    if constexpr (RANK >= 2) {
        mp.qxx = Q[i*9+0]; mp.qxy = Q[i*9+1]; mp.qxz = Q[i*9+2];
        mp.qyy = Q[i*9+4]; mp.qyz = Q[i*9+5]; mp.qzz = Q[i*9+8];
    }
    
    // Multipole gradients
    T drx = T(0); T dry = T(0); T drz = T(0);

    const T pref_dV = T(1) / V / pi<T>();
    const T pref_dE = T(2) / V;

    // --- K-VECTOR LOOP ---
    for (int64_t k = threadIdx.x; k < M; k+=blockDim.x) {
        const T kx = kvec[0*M + k]; const T ky = kvec[1*M + k]; const T kz = kvec[2*M + k];
        const T w  = kvec[3*M + k];
        
        const T Fdr_k = Fdr[k]; const T Fdi_k = Fdi[k];

        const T theta = TWOPI * (kx*rx + ky*ry + kz*rz);
        T s, c;
        sincos_(theta, &s, &c);

        // Define eFd = Fd e^{i theta}
        // Re[eFd] = Re[Fd e^{i theta}] = Fdr*c - Fdi*s
        // Im[eFd] = Im[Fd e^{i theta}] = Fdr*s + Fdi*c
        const T eFdr = Fdr_k*c - Fdi_k*s;
        const T eFdi = Fdr_k*s + Fdi_k*c;

        // Part 1: dr = Re[2pi * i * L * eFd]
        T Lr = mp.c0;
        T Li = T(0);
        if constexpr (RANK == 2) {
            Lr -= TWOPI2O3 * (mp.qxx*kx*kx + 2*mp.qxy*kx*ky + 2*mp.qxz*kx*kz + mp.qyy*ky*ky + 2*mp.qyz*ky*kz + mp.qzz*kz*kz);
        }
        if constexpr (RANK >= 1) {
            Li = TWOPI * (mp.dx*kx + mp.dy*ky + mp.dz*kz);
        }
        T dr = -TWOPI * (Li*eFdr + Lr*eFdi);
         
        // Part 2: dr = Re[i * 2pi * -S * e^{-i theta} *eFd]

        // 2pi * i * (dV/pi/V + 2i*dE.k/V)
        // im = 2pi * dV / pi / V;
        // re = -2pi * (dE.k) * (2/V) 
        const T re = -TWOPI * (dE[i*3+0] * kx + dE[i*3+1] * ky + dE[i*3+2] * kz) * pref_dE;
        const T im = TWOPI * dV[i] * pref_dV;

        // -S * e^{-i theta} = -(Sr + i Si) * (c - i s)
        const T Sr = Sreal[k]; const T Si = Simag[k];
        const T eSr = -(Sr * c + Si * s);
        const T eSi = -(Si * c - Sr * s);

        dr += re * eSr - im * eSi;

       // Coord grad
        drx += w * dr * kx;
        dry += w * dr * ky;
        drz += w * dr * kz;

        // Potential
        mp.ep += w * eFdr;

        // Field
        if constexpr (RANK >= 1) {
            T tmp = w * TWOPI * eFdi;
            mp.efx -= tmp * kx;
            mp.efy -= tmp * ky;
            mp.efz -= tmp * kz;
        }

        // Gradient
        if constexpr (RANK == 2) {
            const T tmp = -(w * TWOPI2O3 * eFdr);
            mp.egxx += tmp * kx*kx; mp.egxy += tmp * kx*ky; mp.egxz += tmp * kx*kz;
            mp.egyy += tmp * ky*ky; mp.egyz += tmp * ky*kz;
            mp.egzz += tmp * kz*kz;
        }
    }

    // Warp-level reduction to reduce synchronization overhead
    // Assumes full warp participation

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        mp.ep += __shfl_down_sync(0xffffffff, mp.ep, offset);
        drx += __shfl_down_sync(0xffffffff, drx, offset);
        dry += __shfl_down_sync(0xffffffff, dry, offset);
        drz += __shfl_down_sync(0xffffffff, drz, offset);

        if constexpr (RANK >= 1) {
            mp.efx += __shfl_down_sync(0xffffffff, mp.efx, offset);
            mp.efy += __shfl_down_sync(0xffffffff, mp.efy, offset);
            mp.efz += __shfl_down_sync(0xffffffff, mp.efz, offset);
        }
        if constexpr (RANK == 2) {
            mp.egxx += __shfl_down_sync(0xffffffff, mp.egxx, offset);
            mp.egxy += __shfl_down_sync(0xffffffff, mp.egxy, offset);
            mp.egxz += __shfl_down_sync(0xffffffff, mp.egxz, offset);
            mp.egyy += __shfl_down_sync(0xffffffff, mp.egyy, offset);
            mp.egyz += __shfl_down_sync(0xffffffff, mp.egyz, offset);
            mp.egzz += __shfl_down_sync(0xffffffff, mp.egzz, offset);
        }
    }
    
    const int lane = threadIdx.x & 31;          // thread index within warp
    const int warp = threadIdx.x >> 5;          // warp index within block
    __shared__ Smem<T, RANK, NUM_WARPS> smem;

    if (lane == 0) {
        smem.ep[warp] = mp.ep;
        smem.fx[warp] = drx;
        smem.fy[warp] = dry;
        smem.fz[warp] = drz;

        if constexpr (RANK >= 1) {
            smem.efx[warp] = mp.efx;
            smem.efy[warp] = mp.efy;
            smem.efz[warp] = mp.efz;
        }
        if constexpr (RANK == 2) {
            smem.egxx[warp] = mp.egxx;
            smem.egxy[warp] = mp.egxy;
            smem.egxz[warp] = mp.egxz;
            smem.egyy[warp] = mp.egyy;
            smem.egyz[warp] = mp.egyz;
            smem.egzz[warp] = mp.egzz;
        }
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp == 0) {
        
        mp.ep = (lane < NUM_WARPS) ? smem.ep[lane] : T(0);
        drx = (lane < NUM_WARPS) ? smem.fx[lane] : T(0);
        dry = (lane < NUM_WARPS) ? smem.fy[lane] : T(0);
        drz = (lane < NUM_WARPS) ? smem.fz[lane] : T(0);

        if constexpr (RANK >= 1) {
            mp.efx = (lane < NUM_WARPS) ? smem.efx[lane] : T(0);
            mp.efy = (lane < NUM_WARPS) ? smem.efy[lane] : T(0);
            mp.efz = (lane < NUM_WARPS) ? smem.efz[lane] : T(0);
        }
        if constexpr (RANK == 2) {
            mp.egxx = (lane < NUM_WARPS) ? smem.egxx[lane] : T(0);
            mp.egxy = (lane < NUM_WARPS) ? smem.egxy[lane] : T(0);
            mp.egxz = (lane < NUM_WARPS) ? smem.egxz[lane] : T(0);
            mp.egyy = (lane < NUM_WARPS) ? smem.egyy[lane] : T(0);
            mp.egyz = (lane < NUM_WARPS) ? smem.egyz[lane] : T(0);
            mp.egzz = (lane < NUM_WARPS) ? smem.egzz[lane] : T(0);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            mp.ep += __shfl_down_sync(0xffffffff, mp.ep, offset);
            drx += __shfl_down_sync(0xffffffff, drx, offset);
            dry += __shfl_down_sync(0xffffffff, dry, offset);
            drz += __shfl_down_sync(0xffffffff, drz, offset);
    
            if constexpr (RANK >= 1) {
                mp.efx += __shfl_down_sync(0xffffffff, mp.efx, offset);
                mp.efy += __shfl_down_sync(0xffffffff, mp.efy, offset);
                mp.efz += __shfl_down_sync(0xffffffff, mp.efz, offset);
            }
            if constexpr (RANK == 2) {
                mp.egxx += __shfl_down_sync(0xffffffff, mp.egxx, offset);
                mp.egxy += __shfl_down_sync(0xffffffff, mp.egxy, offset);
                mp.egxz += __shfl_down_sync(0xffffffff, mp.egxz, offset);
                mp.egyy += __shfl_down_sync(0xffffffff, mp.egyy, offset);
                mp.egyz += __shfl_down_sync(0xffffffff, mp.egyz, offset);
                mp.egzz += __shfl_down_sync(0xffffffff, mp.egzz, offset);
            }
        }
        
        if (lane == 0) {
            constexpr T INV_ROOT_PI = inv_root_pi<T>();
            const T a_over_rpi = alpha * INV_ROOT_PI;
            
            q_grad_out[i] += mp.ep - T(2.0) * a_over_rpi * dV[i];
            coords_grad_out[i*3+0] += drx;
            coords_grad_out[i*3+1] += dry;
            coords_grad_out[i*3+2] += drz;
            if constexpr (RANK >= 1) {
                const T self_p_grad = a_over_rpi * (4.0 * alpha * alpha / 3.0);
                p_grad_out[i*3+0] += mp.efx + self_p_grad * dE[i*3+0];
                p_grad_out[i*3+1] += mp.efy + self_p_grad * dE[i*3+1];
                p_grad_out[i*3+2] += mp.efz + self_p_grad * dE[i*3+2];
            }
            if constexpr (RANK == 2) {
                Q_grad_out[i*9+0] += mp.egxx;
                Q_grad_out[i*9+1] += mp.egxy;
                Q_grad_out[i*9+2] += mp.egxz;
                Q_grad_out[i*9+3] += mp.egxy;
                Q_grad_out[i*9+4] += mp.egyy;
                Q_grad_out[i*9+5] += mp.egyz;
                Q_grad_out[i*9+6] += mp.egxz;
                Q_grad_out[i*9+7] += mp.egyz;
                Q_grad_out[i*9+8] += mp.egzz;
            }
        }
    }
    
}
