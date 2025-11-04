// Ewald PME (reciprocal space) for multipoles up to quadrupoles.
//   twopi = 2*pi
//   L_j(k) = q_j + i*twopi*(p_j.k) - (twopi^2/3)*(k^T Q_j k)   [Q is 3x3 cartesian]
//   S(k)   = sum_j L_j(k) * exp(+i*twopi*k.r_j)
//   theta_ik = twopi * k.r_i
// Channels accumulated per atom i from all k (excluding origin):
//   Re_part = Re{ S(k) * exp(-i*theta_ik) }
//   Im_part = Im{ S(k) * exp(-i*theta_ik) }
// Accumulations and post-factors:
//   phi_i   += (1/(pi*V)) * sum_k w(k) * Re_part
//   E_i     += (-2/V)     * sum_k w(k) * Im_part * k
//   gradE_i += (4*pi/V)   * sum_k w(k) * Re_part * (k k^T)
// where w(k) = exp( -pi^2 |k|^2 / alpha^2 ) / |k|^2 * sym(k), sym = 2 if h>0 else 1.
// Operator-form forces (no curvature needed):
//   F_i = (-2/V) * sum_k w(k) * k * Im{ e^{-i*theta_ik} * conj(L_i(k)) * S(k) }
// with conj(L_i(k)) = q_i - i*twopi*(p_i.k) - (twopi^2/3)*(k^T Q_i k)   [quad term is real]
// Self terms (added after accumulation):
//   phi_self = -2*alpha/sqrt(pi) * q
//   E_self   =  (4/3)*alpha^3/sqrt(pi) * p
//   grad_self= (16/15)*alpha^5/sqrt(pi) * Q
// Q convention: full Cartesian quadrupole matrix (no 1/2 factor, not traceless-scaled).
#include <torch/extension.h>
#include <torch/autograd.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdio>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

// ========================= DEBUG CONTROLS =========================
#ifndef EWALD_DBG
#define EWALD_DBG 0   // 1 = enable; 0 = disable
#endif
#ifndef DBG_KMAX
#define DBG_KMAX 6    // print k = 0..DBG_KMAX-1
#endif
#ifndef DBG_NIDX
#define DBG_NIDX 1    // atom index to trace in accumulation kernel
#endif
#define DBG_PRINT_S(k) ((k) < DBG_KMAX)
#define DBG_PRINT_N(n) ((n) == DBG_NIDX)
// =================================================================

#define INV_ROOT_PI 0.5641895835477563  // 1/sqrt(pi)

// Device-safe numeric helpers
__device__ __forceinline__ double d_twopi()       { return 2.0 * CUDART_PI; }
__device__ __forceinline__ double d_twopi2()      { double t = d_twopi(); return t*t; }
__device__ __forceinline__ double d_twopi2_over3(){ return d_twopi2() / 3.0; }

// ============================================================================
// small host helper
// ============================================================================
inline double det3x3_host_rowmajor(const double m[9]) {
  return m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
}

// ============================================================================
// kernels
// ============================================================================

// reciprocal_box_kernel
// Purpose: compute reciprocal box from box and volume.
// Inputs:
//   box: (3,3) row-major on device
//   V: scalar volume (host-computed)
// Output:
//   recip: (3,3) on device
template <typename T>
__device__ __forceinline__ void reciprocal_box_dev(const T* box, T* recip, T V) {
  const T b1x=box[0], b1y=box[3], b1z=box[6];
  const T b2x=box[1], b2y=box[4], b2z=box[7];
  const T b3x=box[2], b3y=box[5], b3z=box[8];
  const T invV = T(1)/V;

  const T c23x = b2y*b3z - b2z*b3y;
  const T c23y = b2z*b3x - b2x*b3z;
  const T c23z = b2x*b3y - b2y*b3x;

  const T c31x = b3y*b1z - b3z*b1y;
  const T c31y = b3z*b1x - b3x*b1z;
  const T c31z = b3x*b1y - b3y*b1x;

  const T c12x = b1y*b2z - b1z*b2y;
  const T c12y = b1z*b2x - b1x*b2z;
  const T c12z = b1x*b2y - b1y*b2x;

  recip[0]=c23x*invV; recip[3]=c31x*invV; recip[6]=c12x*invV;
  recip[1]=c23y*invV; recip[4]=c31y*invV; recip[7]=c12y*invV;
  recip[2]=c23z*invV; recip[5]=c31z*invV; recip[8]=c12z*invV;
}

template <typename T>
__global__ void reciprocal_box_kernel(const T* box, T* recip, T V) {
  if (blockIdx.x==0 && threadIdx.x==0) reciprocal_box_dev<T>(box, recip, V);
}

// make_hkl_kernel
// Purpose: generate HKL grid (3 x M1) for K-shells, excluding origin (0,0,0).
// Inputs:
//   K: scalar, half-width in h; h in [0..K], k,l in [-K..K]
// Outputs:
//   hkl: (3, M1) with origin removed; ld = M1
template <typename T>
__global__ void make_hkl_kernel(int K, T* __restrict__ hkl, size_t ld) {
  const int R = 2*K + 1;
  const size_t M  = (size_t)(K+1)*R*R;  // include origin
  const size_t i0 = (size_t)K*R + K;    // origin index when h=0 plane

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M || idx == i0) return;

  const int lIdx =  idx            % R;
  const int kIdx = (idx / R)       % R;
  const int hIdx =  idx / (R*R);

  const int h  = hIdx;       // h >= 0
  const int kk = kIdx - K;
  const int kl = lIdx - K;
  if (h==0 && kk==0 && kl==0) return;

  const size_t j = (idx < i0) ? idx : (idx - 1);  // 0..M1-1
  hkl[0*ld + j] = (T)h;
  hkl[1*ld + j] = (T)kk;
  hkl[2*ld + j] = (T)kl;
}

// krow_ops_kernel
// Purpose: per-k constants: exp(-pi^2 |k|^2 / alpha^2)/|k|^2 and symmetry factor.
// Inputs:
//   kvec: (3, M1)
//   hkl : (3, M1) [h used for symmetry choice]
// Outputs:
//   gaussian(M1), sym(M1)
template <typename T>
__global__ void krow_ops_kernel(
    size_t M1, T alpha,
    const T* __restrict__ kvec, int ld_kvec,  // (3, M1)
    const T* __restrict__ hkl,  int ld_hkl,   // (3, M1)
    T* __restrict__ gaussian, T* __restrict__ sym) {

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=M1) return;
  const T kx = kvec[0*ld_kvec + i];
  const T ky = kvec[1*ld_kvec + i];
  const T kz = kvec[2*ld_kvec + i];
  const T k2v = kx*kx + ky*ky + kz*kz;

  const T num = -(CUDART_PI * CUDART_PI * k2v) / (alpha*alpha);
  gaussian[i] = exp(num) / k2v;

  const int hi = (int)llrint((double)hkl[0*ld_hkl + i]); // h index
  sym[i] = (hi > 0) ? T(2) : T(1);
}

// sin_cos_kernel
// Purpose: cos/sin(theta) for theta = twopi * k.r
// Inputs:
//   k_dot_r: (M1, N), ldk is stride(0)
// Outputs:
//   cos_kr: (M1, N)
//   sin_kr: (M1, N)
template <typename T>
__global__ void sin_cos_kernel(
    int M, int N,
    const T* __restrict__ k_dot_r, int ldk,
    T* __restrict__ cos_kr,
    T* __restrict__ sin_kr) {
  int k = blockIdx.x;
  int n = blockIdx.y * blockDim.x + threadIdx.x;
  if (k>=M || n>=N) return;
  const T x = T(2) * T(CUDART_PI) * k_dot_r[k*ldk + n];
  cos_kr[k*ldk + n] = cos(x);
  sin_kr[k*ldk + n] = sin(x);

#if EWALD_DBG
  if (k==0 && n==0) {
    printf("[PHASE] ldk=%d  cos(0,0)=% .8e  sin(0,0)=% .8e\n",
           ldk, (double)cos_kr[0], (double)sin_kr[0]);
  }
#endif
}

// structure_factor_kernel
// Purpose: build S(k) = sum_n L_n(k) * exp(+i*theta_nk)
// where L_n(k) = q_n + i*twopi*(p_n.k) - (twopi^2/3)*(k^T Q_n k)
// and theta_nk = twopi * k.r_n
// Dimensions:
//   kvec: (3, M1), ld_kvec = M1
//   q: (N), p: (N,3), t: (N,9 row-major)
//   coskr, sinkr: (M1, N), leading dim ldk_kr*
//   outputs Sreal, Simag: (M1)
template <typename T>
__global__ void structure_factor_kernel(
    int64_t ldk_kr, int64_t N,
    const T* __restrict__ kvec, int ld_kvec,   // (3,M1)
    const T* __restrict__ q,                   // (N)
    const T* __restrict__ p,                   // (N,3) or nullptr
    const T* __restrict__ t,                   // (N,9) row-major or nullptr
    const T* __restrict__ coskr, int ldk_kr1,  // (M1,N)
    const T* __restrict__ sinkr, int ldk_kr2,  // (M1,N)
    int rank,
    T* __restrict__ Sreal,                     // (M1)
    T* __restrict__ Simag) {                   // (M1)

  const int k = blockIdx.x;
  if (k >= ld_kvec) return;

  const T kx = kvec[0*ld_kvec + k];
  const T ky = kvec[1*ld_kvec + k];
  const T kz = kvec[2*ld_kvec + k];

  const T twopi    = (T)d_twopi();
  const T twopi2o3 = (T)d_twopi2_over3();

  T acc_r = T(0), acc_i = T(0);

  for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
    const T c = coskr[k*ldk_kr1 + n];
    const T s = sinkr[k*ldk_kr2 + n];

    // F_real and F_imag
    T F_r = q[n];
    if (rank >= 2 && t) {
      const T* tn = &t[n*9];
      const T tkx = tn[0]*kx + tn[1]*ky + tn[2]*kz;
      const T tky = tn[3]*kx + tn[4]*ky + tn[5]*kz;
      const T tkz = tn[6]*kx + tn[7]*ky + tn[8]*kz;
      const T ktk = kx*tkx + ky*tky + kz*tkz;
      F_r -= twopi2o3 * ktk; // quadrupole contribution (real)
    }

    T F_i = T(0);
    if (rank >= 1 && p) {
      const T px = p[3*n + 0], py = p[3*n + 1], pz = p[3*n + 2];
      F_i = twopi * (kx*px + ky*py + kz*pz);
    }

    // (F_r + i F_i) e^{+i theta}
    const T real_add = F_r*c - F_i*s;
    const T imag_add = F_r*s + F_i*c;

#if EWALD_DBG
    if (k < DBG_KMAX && N <= 8) {
      printf("[SF.DBG] k=%d tid=%d n=%lld  c=% .8e s=% .8e  Fr=% .8e Fi=% .8e  +re=% .8e +im=% .8e  k=(% .8e,% .8e,% .8e)\n",
             k, (int)threadIdx.x, (long long)n,
             (double)c, (double)s, (double)F_r, (double)F_i,
             (double)real_add, (double)imag_add,
             (double)kx, (double)ky, (double)kz);
    }
#endif

    acc_r += real_add;
    acc_i += imag_add;
  }

  __shared__ T s_r[256], s_i[256];
  const int tid = threadIdx.x;
  s_r[tid] = acc_r; s_i[tid] = acc_i;
  __syncthreads();
  for (int stride = blockDim.x>>1; stride>0; stride>>=1) {
    if (tid < stride) { s_r[tid]+=s_r[tid+stride]; s_i[tid]+=s_i[tid+stride]; }
    __syncthreads();
  }
  if (tid==0) {
    Sreal[k] = s_r[0];
    Simag[k] = s_i[0];
#if EWALD_DBG
    if (DBG_PRINT_S(k)) {
      printf("[SF] k=%d  Sr=% .8e  Si=% .8e\n", k, (double)Sreal[k], (double)Simag[k]);
    }
#endif
  }
}

// accumulate_atoms_kernel
// Purpose: accumulate atom-wise potential, field, gradient from S(k).
// For atom i:
//   Re_part = Re{ S e^{-i theta_i} }  with theta_i = twopi*k.r_i
//   Im_part = Im{ S e^{-i theta_i} }
// Accumulated (pre-scaling):
//   pot_sum += w * Re_part
//   fld_sum += w * Im_part * k
//   grd_sum += w * Re_part * (k k^T)
// Post scalings:
//   phi_i   += (1/(pi*V)) * pot_sum
//   E_i     += (-2/V)     * fld_sum
//   gradE_i += (4*pi/V)   * grd_sum
// Inputs:
//   kvec(3,M1), coskr(M1,N), sinkr(M1,N), gaussian(M1), sym(M1), Sreal(M1), Simag(M1)
// Outputs:
//   pot(N), field(N,3), grad(N,9)
template <typename T>
__global__ void accumulate_atoms_kernel(
    int64_t M1, int64_t N,
    const T* __restrict__ kvec, int ld_kvec,      // (3,M1)
    const T* __restrict__ coskr, int ldk_kr1,     // (M1,N)
    const T* __restrict__ sinkr, int ldk_kr2,     // (M1,N)
    const T* __restrict__ gaussian,               // (M1)
    const T* __restrict__ sym,                    // (M1)
    const T* __restrict__ Sreal,                  // (M1)
    const T* __restrict__ Simag,                  // (M1)
    T V,
    int with_field, int with_grad,
    T* __restrict__ pot,                          // (N)
    T* __restrict__ field,                        // (N,3)
    T* __restrict__ grad)                         // (N,9)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  const T invV     = T(1) / V;
  const T pref_pot = invV / T(CUDART_PI);            // 1/(pi*V)
  const T pref_fld = -T(2) * invV;                   // -2/V
  const T pref_grd = T(4) * T(CUDART_PI) * invV;     // 4*pi/V

  T acc_p = T(0);
  T acc_fx=T(0), acc_fy=T(0), acc_fz=T(0);
  T g00=0,g01=0,g02=0, g10=0,g11=0,g12=0, g20=0,g21=0,g22=0;

  for (int64_t k = 0; k < M1; ++k) {
    const T ckr  = coskr[k*ldk_kr1 + n];
    const T skr  = sinkr[k*ldk_kr2 + n];
    const T Sr   = Sreal[k];
    const T Si   = Simag[k];
    const T w    = gaussian[k] * sym[k];

    // Re[S e^{-i theta}] and Im[S e^{-i theta}]
    const T realp = Sr*ckr + Si*skr;     // potential term
    const T imagp = -Sr*skr + Si*ckr;    // field term

    // potential
    acc_p += w * realp;

    // k components
    const T kx = kvec[0*ld_kvec + k];
    const T ky = kvec[1*ld_kvec + k];
    const T kz = kvec[2*ld_kvec + k];

    // field
    if (with_field) {
      acc_fx += w * imagp * kx;
      acc_fy += w * imagp * ky;
      acc_fz += w * imagp * kz;
    }
    // field grad
    if (with_grad) {
      const T rr = w * realp;
      g00 += rr*kx*kx; g01 += rr*kx*ky; g02 += rr*kx*kz;
      g10 += rr*ky*kx; g11 += rr*ky*ky; g12 += rr*ky*kz;
      g20 += rr*kz*kx; g21 += rr*kz*ky; g22 += rr*kz*kz;
    }

#if EWALD_DBG
    if (DBG_PRINT_N(n) && DBG_PRINT_S(k)) {
      printf("[ACC] n=%d k=%d  Sr=% .8e Si=% .8e  c=% .8e s=% .8e  re=% .8e im=% .8e  w=% .8e sym=%g  k=(% .8e,% .8e,% .8e)\n",
             n, (int)k,
             (double)Sr, (double)Si, (double)ckr, (double)skr,
             (double)realp, (double)imagp, (double)w, (double)sym[k],
             (double)kx, (double)ky, (double)kz);
    }
#endif
  }

#if EWALD_DBG
  if (DBG_PRINT_N(n)) {
    printf("[ACC] n=%d totals (pre-prefactor): pot_sum=% .12e  Fx=% .12e  Fy=% .12e  Fz=% .12e\n",
           n, (double)acc_p, (double)acc_fx, (double)acc_fy, (double)acc_fz);
  }
#endif

  // Final scalings
  pot[n] += pref_pot * acc_p;

  if (with_field) {
    field[3*n+0] += pref_fld * acc_fx;
    field[3*n+1] += pref_fld * acc_fy;
    field[3*n+2] += pref_fld * acc_fz;
  }
  if (with_grad) {
    T* G = &grad[9*n];
    G[0]+=pref_grd*g00; G[1]+=pref_grd*g01; G[2]+=pref_grd*g02;
    G[3]+=pref_grd*g10; G[4]+=pref_grd*g11; G[5]+=pref_grd*g12;
    G[6]+=pref_grd*g20; G[7]+=pref_grd*g21; G[8]+=pref_grd*g22;
  }
}

// forces_from_operator_kernel
// From Celeste Sagui; Lee G. Pedersen; Thomas A. Darden J. Chem. Phys. 120, 73–87 (2004)
// Purpose: operator-form forces for atom i:
//   F_i = (-2/V) * sum_k w(k) * k * Im{ e^{-i*theta_ik} * conj(L_i(k)) * S(k) }
// with
//   conj(L_i(k)) = q_i - i*twopi*(p_i.k) - (twopi^2/3)*(k^T Q_i k)   [quad term is real]
// Dimensions:
//   kvec(3,M1), coskr(M1,N), sinkr(M1,N), gaussian(M1), sym(M1)
//   Sreal(M1), Simag(M1), q(N), p(N,3), Q(N,9)
// Output:
//   F(N,3)  (atomic adds)
template <typename T>
__global__ void forces_from_operator_kernel(
    int64_t M1, int64_t N, T V,
    const T* __restrict__ kvec, int ld_kvec,
    const T* __restrict__ gaussian, const T* __restrict__ sym,
    const T* __restrict__ coskr, int ldk, const T* __restrict__ sinkr, int ldk2,
    const T* __restrict__ q,    // (N)
    const T* __restrict__ p,    // (N,3) or nullptr
    const T* __restrict__ Q,    // (N,9) or nullptr
    const T* __restrict__ Sreal,// (M1)
    const T* __restrict__ Simag,// (M1)
    T* __restrict__ F,          // (N,3)  +=
    int rank)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  T Fx=0, Fy=0, Fz=0;
  for (int64_t k=0; k<M1; ++k) {
    const T kx = kvec[0*ld_kvec + k];
    const T ky = kvec[1*ld_kvec + k];
    const T kz = kvec[2*ld_kvec + k];
    const T w  = gaussian[k] * sym[k];
    const T ci = coskr[k*ldk  + i];
    const T si = sinkr[k*ldk2 + i];
    const T Sr = Sreal[k], Si = Simag[k];

    // conj(L_i(k)) = Lr + i*Li
    T Lr = q[i];
    T Li = T(0);

    if (rank >= 1 && p) {
      const T px = p[3*i+0], py = p[3*i+1], pz = p[3*i+2];
      Li -= (T)d_twopi() * (kx*px + ky*py + kz*pz); // minus for conjugate
    }
    if (rank >= 2 && Q) {
      const T* Qi = &Q[9*i];
      const T tkx = Qi[0]*kx + Qi[1]*ky + Qi[2]*kz;
      const T tky = Qi[3]*kx + Qi[4]*ky + Qi[5]*kz;
      const T tkz = Qi[6]*kx + Qi[7]*ky + Qi[8]*kz;
      Lr -= (T)d_twopi2_over3() * (kx*tkx + ky*tky + kz*tkz);
    }

    // Im{ (ci - i si) * (Lr + i Li) * (Sr + i Si) }
    const T real_LS = Lr*Sr - Li*Si;
    const T imag_LS = Lr*Si + Li*Sr;
    const T Im_eL_S = ci * imag_LS - si * real_LS;

    // F += (-2/V) * w * k * Im{...}
    const T pref = T(-2.0) / V;
    const T scale = pref * w * Im_eL_S;
    Fx += scale * kx;
    Fy += scale * ky;
    Fz += scale * kz;
  }

  atomicAdd(&F[3*i+0], Fx);
  atomicAdd(&F[3*i+1], Fy);
  atomicAdd(&F[3*i+2], Fz);
}

// energy_kernel
// Purpose: energy from channels (pot, field, grad)..
// Energy per atom n:
//   E_local = 0.5 * ( q*phi - p.E - (1/3) * Q : gradE )
// Inputs:
//   q(N), p(N,3) or null if rank<1, t(N,9) or null if rank<2
//   pot(N), field(N,3), grad(N,9)
// Output:
//   energy (scalar) via block reduction
template <typename T>
__global__ void energy_kernel(
    int64_t N, int rank,
    const T* __restrict__ q,        // (N)
    const T* __restrict__ p,        // (N,3) or nullptr
    const T* __restrict__ t,        // (N,9) or nullptr
    const T* __restrict__ pot,      // (N)
    const T* __restrict__ field,    // (N,3)
    const T* __restrict__ grad,     // (N,9) or nullptr
    T*       __restrict__ energy    // scalar (size 1), init 0
){
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  T e_local = T(0);

  if (n < N) {
    const T qn   = q[n];
    const T potn = pot[n];

    const T Ex = field[3*n + 0];
    const T Ey = field[3*n + 1];
    const T Ez = field[3*n + 2];

    T field_ene = T(0);
    if (rank >= 1 && p) {
      const T px = p[3*n + 0], py = p[3*n + 1], pz = p[3*n + 2];
      field_ene = px*Ex + py*Ey + pz*Ez; // p dot E
    }

    T grad_ene = T(0);
    if (rank >= 2 && t && grad) {
      const T* tn = &t[9*n];
      const T* gn = &grad[9*n]; // row-major
      grad_ene =
        tn[0]*gn[0] + tn[1]*gn[1] + tn[2]*gn[2] +
        tn[3]*gn[3] + tn[4]*gn[4] + tn[5]*gn[5] +
        tn[6]*gn[6] + tn[7]*gn[7] + tn[8]*gn[8];
    }

    e_local = T(0.5) * ( qn*potn - field_ene - grad_ene / T(3) );
  }

  __shared__ T ssum[256];
  const int tid = threadIdx.x;
  ssum[tid] = e_local;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) ssum[tid] += ssum[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(energy, ssum[0]); // OK for double on CC>=6
  }
}

// ============================================================================
// Host-side functions
// ============================================================================

// ewald_prepare_intermediates_cuda
// Purpose: prepare reciprocal-space intermediates and phases.
// Inputs:
//   coords(N,3), box(3,3), K, alpha
// Outputs (8 tensors):
//   0: recip(3,3)
//   1: hkl(3,M1)
//   2: kvec(3,M1)
//   3: gauss(M1)
//   4: sym(M1)
//   5: kdotr(M1,N)
//   6: coskr(M1,N)
//   7: sinkr(M1,N)
std::vector<at::Tensor> ewald_prepare_intermediates_cuda(
    const at::Tensor& coords, const at::Tensor& box, int64_t K, double alpha) {
  TORCH_CHECK(coords.is_cuda() && box.is_cuda(), "coords & box must be CUDA");
  TORCH_CHECK(coords.scalar_type()==at::kDouble && box.scalar_type()==at::kDouble, "use float64");
  TORCH_CHECK(coords.dim()==2 && coords.size(1)==3, "coords must be (N,3)");
  TORCH_CHECK(box.sizes()==at::IntArrayRef({3,3}), "box must be (3,3)");

  at::cuda::CUDAGuard guard(coords.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t N = coords.size(0);
  auto opts = coords.options();

  // Volume on host (sync anyway due to .cpu())
  double h_box[9];
  {
    auto box_c = box.contiguous().cpu();
    auto* b = box_c.data_ptr<double>();
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) h_box[r*3+c] = b[r*3+c];
  }
  const double V = det3x3_host_rowmajor(h_box);

  // Reciprocal box on device
  auto recip = at::empty({3,3}, opts);
  reciprocal_box_kernel<double><<<1,1,0,stream>>>(
      box.data_ptr<double>(), recip.data_ptr<double>(), V);

  // HKL (3 x M1)
  const int R  = 2*int(K) + 1;
  const int64_t M  = (int64_t)(K+1) * R * R;  // include origin
  const int64_t M1 = M - 1;                   // exclude origin
  auto hkl = at::empty({3, M1}, opts);
  {
    dim3 block(256), grid((unsigned)((M + block.x - 1)/block.x));
    make_hkl_kernel<double><<<grid, block, 0, stream>>>(
        int(K), hkl.data_ptr<double>(), (size_t)M1);
  }

  // kvec = recip @ hkl  (3x3)*(3xM1) = (3xM1)
  auto kvec = at::matmul(recip, hkl).contiguous();
#if EWALD_DBG
  std::cout << "[DBG] kvec sizes " << kvec.sizes()
            << " strides " << kvec.strides() << "\n";
#endif

  // gauss, sym
  auto gauss = at::empty({M1}, opts);
  auto sym   = at::empty({M1}, opts);
  {
    dim3 block(256), grid((unsigned)((M1 + block.x - 1)/block.x));
    krow_ops_kernel<double><<<grid, block, 0, stream>>>(
        (size_t)M1, (double)alpha, kvec.data_ptr<double>(), (int)M1,
        hkl.data_ptr<double>(), (int)M1,
        gauss.data_ptr<double>(), sym.data_ptr<double>());
  }

  // kdotr = (kvec^T) * coords^T  -> (M1 x N)
  auto kdotr = at::matmul(kvec.transpose(0,1), coords.transpose(0,1)).contiguous();

  // cos/sin(2pi*k.r)
  auto coskr = at::empty_like(kdotr);
  auto sinkr = at::empty_like(kdotr);
  const int ldk = (int)kdotr.stride(0);
  {
    dim3 block(256,1,1);
    dim3 grid((int)M1, (int)((N + block.x - 1)/block.x), 1);
    sin_cos_kernel<double><<<grid, block, 0, stream>>>(
        (int)M1, (int)N, kdotr.data_ptr<double>(), ldk,
        coskr.data_ptr<double>(), sinkr.data_ptr<double>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {recip, hkl, kvec, gauss, sym, kdotr, coskr, sinkr};
}

// ewald_long_range_cuda
// Purpose: compute channels (phi, field, grad) and S(k), then add self terms.
// Returns (6 tensors):
//   potential(N), field(N,3), fieldgrad(N,3,3), Sreal(M1), Simag(M1), dummy (empty)
// Note: dummy kept only for compatibility of tie(...) patterns; not necessary otherwise.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ewald_long_range_cuda(
    const at::Tensor& coords, const at::Tensor& box,
    const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
    const at::Tensor& kvec, const at::Tensor& gaussian, const at::Tensor& sym,
    const at::Tensor& coskr, const at::Tensor& sinkr,
    int rank, double alpha) {

  TORCH_CHECK(coords.is_cuda() && q.is_cuda() && kvec.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(coords.scalar_type()==at::kDouble, "double tensors expected");

  at::cuda::CUDAGuard guard(coords.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t N  = coords.size(0);
  const int64_t M1 = gaussian.size(0);

#if EWALD_DBG
  std::cout << "Number of k-vectors (M1): " << M1 << std::endl;
  const double V_dbg = at::det(box).item<double>();
  std::cout << std::setprecision(10)
          << "Volume V = " << V_dbg
          << "   1/(pi*V)=" << (1.0 / (CUDART_PI * V_dbg))
          << "   -2/V="   << (-2.0 / V_dbg) << "\n";
  std::cout << std::setprecision(10)
            << "[CUDA] sum(sym) = " << sym.sum().item<double>() << "\n";
  for (int k = 0; k < std::min<int64_t>(DBG_KMAX, M1); ++k) {
    auto kx = kvec.index({0,k}).item<double>();
    auto ky = kvec.index({1,k}).item<double>();
    auto kz = kvec.index({2,k}).item<double>();
    auto g  = gaussian[k].item<double>();
    auto s  = sym[k].item<double>();
    std::cout << "[CUDA] k["<<k<<"] = ("<<kx<<","<<ky<<","<<kz<<")  g="<<g<<"  sym="<<s<<"\n";
  }
  {
    const int dbg_n = std::min<int64_t>(DBG_NIDX, N-1);
    std::cout << "--- cos/sin for n="<<dbg_n<<" (first " << std::min<int64_t>(DBG_KMAX, M1) << " k) ---\n";
    for (int k = 0; k < std::min<int64_t>(DBG_KMAX, M1); ++k) {
      double c = coskr.index({k, dbg_n}).item<double>();
      double s = sinkr.index({k, dbg_n}).item<double>();
      std::cout << "k="<<k<<"  cos="<<c<<"  sin="<<s<<"\n";
    }
  }
#endif

  // Build S(k)
  auto Sreal = at::zeros({M1}, coords.options());
  auto Simag = at::zeros({M1}, coords.options());
  const int ldk_kr = (int)coskr.stride(0);
  {
    dim3 grid((unsigned)M1), block(256);
    structure_factor_kernel<double><<<grid, block, 0, stream>>>(
      M1, N,
      kvec.data_ptr<double>(), (int)M1,
      q.data_ptr<double>(),
      (rank>=1 && p.defined()) ? p.data_ptr<double>() : nullptr,
      (rank>=2 && t.defined()) ? t.reshape({N,9}).data_ptr<double>() : nullptr,
      coskr.data_ptr<double>(), ldk_kr,
      sinkr.data_ptr<double>(), ldk_kr,
      rank,
      Sreal.data_ptr<double>(),
      Simag.data_ptr<double>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#if EWALD_DBG
  std::cout << "--- S(k) (first " << std::min<int64_t>(DBG_KMAX,M1) << ") ---\n";
  for (int k = 0; k < std::min<int64_t>(DBG_KMAX, M1); ++k) {
    std::cout << "[CUDA] S["<<k<<"] = ("
              << Sreal[k].item<double>() << ", "
              << Simag[k].item<double>() << ")\n";
  }
#endif

  // per-atom accum
  auto potential = at::zeros({N}, coords.options());
  auto field     = at::zeros({N,3}, coords.options());
  auto fieldgrad = at::zeros({N,3,3}, coords.options());

  const double V = at::det(box).item<double>();
  {
    const int with_field = 1;
    const int with_grad  = (rank >= 1);
    const int ldk = (int)coskr.stride(0);
    dim3 block(256), grid((unsigned)((N + block.x - 1)/block.x));
    accumulate_atoms_kernel<double><<<grid, block, 0, stream>>>(
      M1, N,
      kvec.data_ptr<double>(), (int)M1,
      coskr.data_ptr<double>(), ldk,
      sinkr.data_ptr<double>(), ldk,
      gaussian.data_ptr<double>(),
      sym.data_ptr<double>(),
      Sreal.data_ptr<double>(),
      Simag.data_ptr<double>(),
      (double)V,
      with_field, with_grad,
      potential.data_ptr<double>(),
      field.data_ptr<double>(),
      fieldgrad.reshape({N,9}).data_ptr<double>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // self terms
  const double a_over_rpi = alpha * INV_ROOT_PI;
  potential.add_(q, -2.0 * a_over_rpi);
  if (rank >= 1 && p.defined()) field.add_(p, a_over_rpi * (4.0*alpha*alpha/3.0));
  if (rank >= 2 && t.defined()) fieldgrad.add_(t, a_over_rpi * (16.0*alpha*alpha*alpha*alpha/15.0));

#if EWALD_DBG
  if (N >= 2) {
    std::cout << std::setprecision(12)
              << "CUDA potential[0..1]: "
              << potential[0].item<double>() << " "
              << potential[1].item<double>() << "\n";
    std::cout << "CUDA field[1]: ("
              << field[1][0].item<double>() << ", "
              << field[1][1].item<double>() << ", "
              << field[1][2].item<double>() << ")\n";
  }
#endif

  return {potential, field, fieldgrad, Sreal, Simag, at::Tensor() /*unused*/};
}

// ewald_energy_and_forces_cuda
// Purpose: energy (channel-based) and forces (operator-form).
// Inputs include precomputed Sreal/Simag; we do NOT rebuild S.
// Returns (energy scalar, forces(N,3)).
std::tuple<at::Tensor, at::Tensor>
ewald_energy_and_forces_cuda(
    const at::Tensor& coords, const at::Tensor& box,
    const at::Tensor& kvec,  const at::Tensor& gaussian,
    const at::Tensor& sym,   const at::Tensor& coskr,
    const at::Tensor& sinkr,
    const at::Tensor& Sreal, const at::Tensor& Simag,
    const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
    const at::Tensor& potential, const at::Tensor& field, const at::Tensor& grad,
    int rank)
{
  TORCH_CHECK(q.is_cuda() && potential.is_cuda() && field.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(q.scalar_type()==at::kDouble && potential.scalar_type()==at::kDouble &&
              field.scalar_type()==at::kDouble, "double tensors expected");

  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto V = at::det(box).item<double>();
  const int64_t M1 = gaussian.size(0);
  const int64_t N = q.size(0);
  auto opts = q.options();

  // Energy
  auto energy = at::zeros({}, opts);
  {
    dim3 block(256), grid((unsigned)((N + block.x - 1)/block.x));
    energy_kernel<double><<<grid, block, 0, stream>>>(
        N, rank,
        q.data_ptr<double>(),
        (rank>=1 && p.defined()) ? p.data_ptr<double>() : nullptr,
        (rank>=2 && t.defined()) ? t.reshape({N,9}).data_ptr<double>() : nullptr,
        potential.data_ptr<double>(),
        field.data_ptr<double>(),
        (rank>=1) ? grad.reshape({N,9}).data_ptr<double>() : nullptr,
        energy.data_ptr<double>());
  }

  // Forces
  auto forces = at::zeros({N,3}, opts);
  {
    dim3 block(256), grid((unsigned)((N + block.x - 1)/block.x));
    forces_from_operator_kernel<double><<<grid,block,0,stream>>>(
      M1, N, (double)V,
      kvec.data_ptr<double>(), (int)M1,
      gaussian.data_ptr<double>(), sym.data_ptr<double>(),
      coskr.data_ptr<double>(), (int)coskr.stride(0),
      sinkr.data_ptr<double>(), (int)sinkr.stride(0),
      q.data_ptr<double>(),
      (rank>=1 && p.defined()) ? p.data_ptr<double>() : nullptr,
      (rank>=2 && t.defined()) ? t.reshape({N,9}).data_ptr<double>() : nullptr,
      Sreal.data_ptr<double>(), Simag.data_ptr<double>(),
      forces.data_ptr<double>(), rank);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {energy, forces};
}

// One-shot op: returns (pot, field, grad, energy, forces)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ewald_long_range_all_cuda(
    const at::Tensor& coords, const at::Tensor& box,
    const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
    int64_t K, int rank, double alpha) {

  at::cuda::CUDAGuard guard(coords.device());

  // prepare (8 tensors)
  auto prep = ewald_prepare_intermediates_cuda(coords, box, K, alpha);
  const at::Tensor& kvec  = prep[2];
  const at::Tensor& gauss = prep[3];
  const at::Tensor& sym   = prep[4];
  const at::Tensor& coskr = prep[6];
  const at::Tensor& sinkr = prep[7];

  // channels + S(k)
  at::Tensor pot, field, grad, Sreal, Simag, unused;
  std::tie(pot, field, grad, Sreal, Simag, unused) =
      ewald_long_range_cuda(coords, box, q, p, t, kvec, gauss, sym, coskr, sinkr, rank, alpha);

  // energy and forces
  at::Tensor energy, forces;
  std::tie(energy, forces) = ewald_energy_and_forces_cuda(
      coords, box,
      kvec, gauss, sym, coskr, sinkr,
      Sreal, Simag,
      q, p, t,
      pot, field, grad,
      rank);

  return {pot, field, grad, energy, forces};
}

// ============================================================================
// Autograd wrapper: custom backward (coords only via energy)
// ============================================================================
struct EwaldLongRangeAllFunctionCuda
    : public torch::autograd::Function<EwaldLongRangeAllFunctionCuda> {

  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor coords,
      at::Tensor box,
      at::Tensor q,
      at::Tensor p,
      at::Tensor t,
      at::Tensor K_t,     // 0-d long tensor
      at::Tensor rank_t,  // 0-d long tensor
      at::Tensor alpha_t  // 0-d double tensor
  ) {
    const int64_t K    = K_t.item<int64_t>();
    const int      rk  = static_cast<int>(rank_t.item<int64_t>());
    const double   alp = alpha_t.item<double>();

    // prepare
    auto prep = ewald_prepare_intermediates_cuda(coords, box, K, alp);
    const at::Tensor& kvec  = prep[2];
    const at::Tensor& gauss = prep[3];
    const at::Tensor& sym   = prep[4];
    const at::Tensor& coskr = prep[6];
    const at::Tensor& sinkr = prep[7];

#if EWALD_DBG
    const int64_t N = coords.size(0);
    const int64_t M1 = gauss.size(0);
    TORCH_CHECK(kvec.dim()==2, "kvec must be 2D, got ", kvec.sizes());
    TORCH_CHECK(gauss.dim()==1 && gauss.size(0)==M1, "gauss must be (M1,), got ", gauss.sizes());
    TORCH_CHECK(sym.dim()==1 && sym.size(0)==M1, "sym must be (M1,), got ", sym.sizes());
    TORCH_CHECK(coskr.dim()==2 && coskr.size(0)==M1 && coskr.size(1)==N, "coskr must be (M1,N), got ", coskr.sizes());
    TORCH_CHECK(sinkr.dim()==2 && sinkr.size(0)==M1 && sinkr.size(1)==N, "sinkr must be (M1,N), got ", sinkr.sizes());
    std::cout << "[AUTO] N="<<N<<" M1="<<M1<<" sum(sym)="<<sym.sum().item<double>()<<"\n";
#endif

    // channels + S(k)
    at::Tensor potential, field, grad, Sreal, Simag, unused;
    std::tie(potential, field, grad, Sreal, Simag, unused) =
        ewald_long_range_cuda(coords, box, q, p, t,
                              kvec, gauss, sym, coskr, sinkr,
                              rk, alp);

    // energy & forces
    at::Tensor energy, forces;
    std::tie(energy, forces) = ewald_energy_and_forces_cuda(
      coords, box,
      kvec, gauss, sym, coskr, sinkr,
      Sreal, Simag,
      q, p, t,
      potential, field, grad,
      rk);

    // save forces for backward: dL/dcoords = -gE * F
    ctx->save_for_backward({forces});

#if EWALD_DBG
    if (potential.numel()>=3) {
      std::cout << "[AUTO] pot[0..2]="
                << potential[0].item<double>() << " "
                << potential[1].item<double>() << " "
                << potential[2].item<double>() << "\n";
    }
#endif

    torch::autograd::variable_list out(5);
    out[0] = potential;
    out[1] = field;
    out[2] = grad;
    out[3] = energy;
    out[4] = forces;
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {

    // grad_outputs: [g_potential, g_field, g_grad, g_energy, g_forces]
    const at::Tensor& gE = grad_outputs[3]; // scalar

    auto saved = ctx->get_saved_variables();
    const at::Tensor& F = saved[0]; // (N,3)

    at::Tensor dcoords;
    if (gE.defined()) {
      // dL/dcoords = -gE * forces
      auto scale = (gE.dim()==0) ? gE.view({1,1}) : gE;
      dcoords = -(scale) * F;
      if (!dcoords.sizes().equals(F.sizes())) {
        dcoords = dcoords.expand_as(F).contiguous();
      }
    } else {
      dcoords = at::zeros_like(F);
    }

    // We passed 8 inputs to forward; return 8 grads in order.
    torch::autograd::variable_list grads(8);
    grads[0] = dcoords;      // d/d coords
    grads[1] = at::Tensor(); // d/d box   (not implemented)
    grads[2] = at::Tensor(); // d/d q     (not implemented)
    grads[3] = at::Tensor(); // d/d p     (not implemented)
    grads[4] = at::Tensor(); // d/d t     (not implemented)
    grads[5] = at::Tensor(); // d/d K_t   (int scalar -> no grad)
    grads[6] = at::Tensor(); // d/d rank_t
    grads[7] = at::Tensor(); // d/d alpha_t
    return grads;
  }
};

// ============================================================================
// Registration
// ============================================================================

TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
  // prepare: 8-tensor return
  m.impl("ewald_prepare_intermediates",
         [](const at::Tensor& coords, const at::Tensor& box, int64_t K, double alpha) {
           auto outs = ewald_prepare_intermediates_cuda(coords, box, K, alpha);
           // return 8 tensors
           return std::make_tuple(outs[0],outs[1],outs[2],outs[3],outs[4],outs[5],outs[6],outs[7]);
         });

  // finish long-range: returns (pot, field, grad, Sreal, Simag, unused)
  m.impl("ewald_finish_long_range",
         [](const at::Tensor& coords, const at::Tensor& box,
            const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
            const at::Tensor& kvec, const at::Tensor& gaussian, const at::Tensor& sym,
            const at::Tensor& coskr, const at::Tensor& sinkr, int64_t rank, double alpha) {
           return ewald_long_range_cuda(coords, box, q, p, t,
                                        kvec, gaussian, sym, coskr, sinkr,
                                        (int)rank, alpha);
         });

  // energy + forces using precomputed S(k)
  m.impl("ewald_energy_forces",
         [](const at::Tensor& coords, const at::Tensor& box,
            const at::Tensor& kvec,  const at::Tensor& gaussian,
            const at::Tensor& sym,   const at::Tensor& coskr,
            const at::Tensor& sinkr,
            const at::Tensor& Sreal, const at::Tensor& Simag,
            const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
            const at::Tensor& potential, const at::Tensor& field, const at::Tensor& grad,
            int64_t rank) {
           return ewald_energy_and_forces_cuda(
             coords, box,
             kvec, gaussian, sym, coskr, sinkr,
             Sreal, Simag,
             q, p, t,
             potential, field, grad, (int)rank);
         });
}

// One-shot op with custom autograd.
TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
  m.impl("ewald_long_range",
         [](const at::Tensor& coords, const at::Tensor& box,
            const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
            int64_t K, int64_t rank, double alpha) {

           auto K_t     = at::scalar_tensor(K,     at::TensorOptions().dtype(at::kLong));
           auto rank_t  = at::scalar_tensor(rank,  at::TensorOptions().dtype(at::kLong));
           auto alpha_t = at::scalar_tensor(alpha, at::TensorOptions().dtype(at::kDouble));

           auto outs = EwaldLongRangeAllFunctionCuda::apply(
               coords, box, q, p, t, K_t, rank_t, alpha_t);

           return std::make_tuple(outs[0], outs[1], outs[2], outs[3], outs[4]);
         });
}

