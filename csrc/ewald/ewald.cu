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
#include <tuple>
#include <iostream>
#include <iomanip>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

// ========================= DEBUG CONTROLS =========================
#ifndef EWALD_DBG
#define EWALD_DBG 0
#endif

// Device-safe numeric helpers
__device__ __forceinline__ double d_twopi()       { return 2.0 * CUDART_PI; }
__device__ __forceinline__ double d_twopi2()      { double t = d_twopi(); return t*t; }
__device__ __forceinline__ double d_twopi2_over3(){ return d_twopi2() / 3.0; }
#define INV_ROOT_PI 0.56418958354

// ============================================================================
// KERNEL 1: PREPARE K-CONSTANTS (HKL + KROW_OPS)
// ============================================================================
template <typename T>
__global__ void prepare_k_constants_kernel(
    int K, T alpha, size_t M1,
    const T* __restrict__ recip, // (3,3)
    T* __restrict__ kvec,        // (3,M1) out
    T* __restrict__ gaussian,    // (M1) out
    T* __restrict__ sym)         // (M1) out
{
    const int R = 2*K + 1;
    const size_t M  = (size_t)(K+1)*R*R;
    const size_t i0 = (size_t)K*R + K;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= M || idx == i0) return;

    // 1. Decode Linear Index to H,K,L
    const int lIdx =  idx % R;
    const int kIdx = (idx / R) % R;
    const int hIdx =  idx / (R*R);
    
    const int h  = hIdx;
    const int kk = kIdx - K;
    const int kl = lIdx - K;
    
    // Safety check for (0,0,0) although idx==i0 usually catches it
    if (h==0 && kk==0 && kl==0) return;

    // 2. Map to output index j (0..M1-1)
    const size_t j = (idx < i0) ? idx : (idx - 1);

    // 3. Compute K-Vector (k = recip * hkl)
    // hkl vector is (h, kk, kl)
    // kvec_x = recip[0]*h + recip[1]*kk + recip[2]*kl
    T kx = recip[0]*h + recip[1]*kk + recip[2]*kl;
    T ky = recip[3]*h + recip[4]*kk + recip[5]*kl;
    T kz = recip[6]*h + recip[7]*kk + recip[8]*kl;

    kvec[0*M1 + j] = kx;
    kvec[1*M1 + j] = ky;
    kvec[2*M1 + j] = kz;

    // 4. Compute Gaussian and Sym
    const T k2v = kx*kx + ky*ky + kz*kz;
    const T num = -(CUDART_PI * CUDART_PI * k2v) / (alpha*alpha);
    gaussian[j] = exp(num) / k2v;
    sym[j] = (h > 0) ? T(2) : T(1);
}

// Small helper for reciprocal box
template <typename T>
__global__ void reciprocal_box_kernel(const T* box, T* recip, T V) {
  if (threadIdx.x == 0) {
    const T b1x=box[0], b1y=box[3], b1z=box[6];
    const T b2x=box[1], b2y=box[4], b2z=box[7];
    const T b3x=box[2], b3y=box[5], b3z=box[8];
    const T invV = T(1)/V;
    recip[0]=(b2y*b3z - b2z*b3y)*invV; recip[3]=(b3y*b1z - b3z*b1y)*invV; recip[6]=(b1y*b2z - b1z*b2y)*invV;
    recip[1]=(b2z*b3x - b2x*b3z)*invV; recip[4]=(b3z*b1x - b3x*b1z)*invV; recip[7]=(b1z*b2x - b1x*b2z)*invV;
    recip[2]=(b2x*b3y - b2y*b3x)*invV; recip[5]=(b3x*b1y - b3y*b1x)*invV; recip[8]=(b1x*b2y - b1y*b2x)*invV;
  }
}

// ============================================================================
// KERNEL 2: STRUCTURE FACTOR
// ============================================================================
// Merges: Phase calculation + S(k) accumulation.
template <typename T>
__global__ void compute_structure_factor_fused(
    int64_t M1, int64_t N,
    const T* __restrict__ kvec,                // (3, M1)
    const T* __restrict__ coords,              // (N, 3)
    const T* __restrict__ q,                   // (N)
    const T* __restrict__ p,                   // (N, 3) or null
    const T* __restrict__ t,                   // (N, 9) or null
    int rank,
    T* __restrict__ Sreal,                     // (M1) Out
    T* __restrict__ Simag)                     // (M1) Out
{
    const int k = blockIdx.x; // One block per k-vector
    if (k >= M1) return;

    const T kx = kvec[0*M1 + k];
    const T ky = kvec[1*M1 + k];
    const T kz = kvec[2*M1 + k];

    const T twopi    = (T)d_twopi();
    const T twopi2o3 = (T)d_twopi2_over3();

    T acc_r = T(0);
    T acc_i = T(0);

    // Grid-stride loop over atoms (if N > blockDim)
    for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
        
        // 1. Phase Calculation
        const T rx = coords[n*3+0];
        const T ry = coords[n*3+1];
        const T rz = coords[n*3+2];
        const T theta = twopi * (kx*rx + ky*ry + kz*rz);
        T s, c;
        sincos(theta, &s, &c);

        // 2. Multipole Charge L_n(k)
        T F_r = q[n];
        T F_i = T(0);
	   // Dipole term
        if (rank >= 1 && p) {
            const T px = p[3*n + 0], py = p[3*n + 1], pz = p[3*n + 2];
            F_i = twopi * (kx*px + ky*py + kz*pz);
        }
        if (rank >= 2 && t) {
            const T* tn = &t[n*9];
            // Quadrupole term
            const T tkx = tn[0]*kx + tn[1]*ky + tn[2]*kz;
            const T tky = tn[3]*kx + tn[4]*ky + tn[5]*kz;
            const T tkz = tn[6]*kx + tn[7]*ky + tn[8]*kz;
            const T ktk = kx*tkx + ky*tky + kz*tkz;
            F_r -= twopi2o3 * ktk; 
        }

        // 3. Complex Multiply: (Fr + iFi) * (c + is)
        acc_r += (F_r*c - F_i*s);
        acc_i += (F_r*s + F_i*c);
    }

    __shared__ T s_r[256], s_i[256];
    const int tid = threadIdx.x;
    s_r[tid] = acc_r; 
    s_i[tid] = acc_i;
    __syncthreads();

    for (int stride = blockDim.x>>1; stride>0; stride>>=1) {
        if (tid < stride) { 
            s_r[tid]+=s_r[tid+stride]; 
            s_i[tid]+=s_i[tid+stride]; 
        }
        __syncthreads();
    }

    if (tid==0) {
        Sreal[k] = s_r[0];
        Simag[k] = s_i[0];
    }
}

// ============================================================================
// KERNEL 3: MAIN
// ============================================================================
// Merges: Accumulation, Operator Forces, Energy, and Self-Terms.
template <typename T>
__global__ void compute_ewald_dynamics_fused(
    int64_t M1, int64_t N, T V, double alpha,
    const T* __restrict__ kvec,                   // (3,M1)
    const T* __restrict__ gaussian,               // (M1)
    const T* __restrict__ sym,                    // (M1)
    const T* __restrict__ Sreal,                  // (M1)
    const T* __restrict__ Simag,                  // (M1)
    const T* __restrict__ coords,                 // (N,3)
    const T* __restrict__ q,                      // (N)
    const T* __restrict__ p,                      // (N,3)
    const T* __restrict__ Q,                      // (N,9)
    int rank,
    T* __restrict__ pot_out,                      // (N)
    T* __restrict__ field_out,                    // (N,3)
    T* __restrict__ grad_out,                     // (N,9)
    T* __restrict__ force_out,                    // (N,3)
    T* __restrict__ total_energy_out)             // (1)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    T local_energy_contribution = T(0);

    if (i < N) {
        // --- CONSTANTS ---
        const T invV     = T(1) / V;
        const T pref_pot = invV / T(CUDART_PI);        
        const T pref_fld = -T(2) * invV;               
        const T pref_grd = T(4) * T(CUDART_PI) * invV; 
        const T force_pref = T(-2.0) / V;
        const T twopi = (T)d_twopi();
        const T twopi2o3 = (T)d_twopi2_over3();

        // --- LOAD ATOM i ---
        const T ri_x = coords[i*3+0];
        const T ri_y = coords[i*3+1];
        const T ri_z = coords[i*3+2];
        const T qi   = q[i];
        
        T p_x=0, p_y=0, p_z=0;
        if (rank >= 1 && p) { p_x=p[3*i+0]; p_y=p[3*i+1]; p_z=p[3*i+2]; }
        
        // --- ACCUMULATORS ---
        T sum_pot = 0;
        T sum_fx = 0, sum_fy = 0, sum_fz = 0; // Field
        // Gradient accumulators
        T g00=0,g01=0,g02=0, g10=0,g11=0,g12=0, g20=0,g21=0,g22=0;
        // Force accumulators
        T force_x = 0, force_y = 0, force_z = 0;

        // --- K-VECTOR LOOP ---
        for (int64_t k = 0; k < M1; ++k) {
            const T kx = kvec[0*M1 + k];
            const T ky = kvec[1*M1 + k];
            const T kz = kvec[2*M1 + k];
            const T w  = gaussian[k] * sym[k];
            
            // Phase
            const T theta = twopi * (kx*ri_x + ky*ri_y + kz*ri_z);
            T si, ci; 
            sincos(theta, &si, &ci);

            const T Sr = Sreal[k];
            const T Si = Simag[k];

            // 1. RE/IM PARTS for Pot/Field
            // Re[S e^{-i theta}] = Sr*c + Si*s
            // Im[S e^{-i theta}] = -Sr*s + Si*c
            const T realp = Sr*ci + Si*si;     
            const T imagp = -Sr*si + Si*ci;    

            sum_pot += w * realp;
            
            // Field
            sum_fx += w * imagp * kx;
            sum_fy += w * imagp * ky;
            sum_fz += w * imagp * kz;

            // Gradient)
            if (rank >= 1) { 
                 const T rr = w * realp;
                 g00 += rr*kx*kx; g01 += rr*kx*ky; g02 += rr*kx*kz;
                 g10 += rr*ky*kx; g11 += rr*ky*ky; g12 += rr*ky*kz;
                 g20 += rr*kz*kx; g21 += rr*kz*ky; g22 += rr*kz*kz;
            }

            // 2. Forces
            // L_i(k) terms
            T Lr = qi;
            T Li = T(0);
            if (rank >= 1) Li -= twopi * (kx*p_x + ky*p_y + kz*p_z);
            if (rank >= 2 && Q) {
                const T* Qi = &Q[9*i];
                const T tkx = Qi[0]*kx + Qi[1]*ky + Qi[2]*kz;
                const T tky = Qi[3]*kx + Qi[4]*ky + Qi[5]*kz;
                const T tkz = Qi[6]*kx + Qi[7]*ky + Qi[8]*kz;
                Lr -= twopi2o3 * (kx*tkx + ky*tky + kz*tkz);
            }

            // Formula: Im{ e^{-i theta} * conj(L) * S }
            // conj(L) = Lr - i*Li 
            // Term = (ci - i*si) * (Lr - i*Li) * (Sr + i*Si)
            const T real_LS = Lr*Sr - Li*Si;
            const T imag_LS = Lr*Si + Li*Sr;
            const T Im_eL_S = ci * imag_LS - si * real_LS;

            const T scale = force_pref * w * Im_eL_S;
            force_x += scale * kx;
            force_y += scale * ky;
            force_z += scale * kz;
        }

        const T a_over_rpi = alpha * INV_ROOT_PI;
        
        // Potential
        T final_pot = pref_pot * sum_pot;
        final_pot += (-2.0 * a_over_rpi * qi); // Self
        pot_out[i] = final_pot;

        // Field
        T final_Ex = pref_fld * sum_fx;
        T final_Ey = pref_fld * sum_fy;
        T final_Ez = pref_fld * sum_fz;
        
        if (rank >= 1 && p) {
            const T self_fld = a_over_rpi * (4.0*alpha*alpha/3.0);
            final_Ex += p_x * self_fld;
            final_Ey += p_y * self_fld;
            final_Ez += p_z * self_fld;
        }
        field_out[3*i+0] = final_Ex;
        field_out[3*i+1] = final_Ey;
        field_out[3*i+2] = final_Ez;

        // Gradient
        T G00=pref_grd*g00, G01=pref_grd*g01, G02=pref_grd*g02;
        T G10=pref_grd*g10, G11=pref_grd*g11, G12=pref_grd*g12;
        T G20=pref_grd*g20, G21=pref_grd*g21, G22=pref_grd*g22;
        
        if (rank >= 2 && Q) {
            const T self_grd = a_over_rpi * (16.0*alpha*alpha*alpha*alpha/15.0);
            const T* Qi = &Q[9*i];
            G00 += Qi[0]*self_grd; G01 += Qi[1]*self_grd; G02 += Qi[2]*self_grd;
            G10 += Qi[3]*self_grd; G11 += Qi[4]*self_grd; G12 += Qi[5]*self_grd;
            G20 += Qi[6]*self_grd; G21 += Qi[7]*self_grd; G22 += Qi[8]*self_grd;
        }
        
        T* G_out = &grad_out[9*i];
        G_out[0]=G00; G_out[1]=G01; G_out[2]=G02;
        G_out[3]=G10; G_out[4]=G11; G_out[5]=G12;
        G_out[6]=G20; G_out[7]=G21; G_out[8]=G22;

        // Forces
        force_out[3*i+0] = force_x;
        force_out[3*i+1] = force_y;
        force_out[3*i+2] = force_z;

        // Local Energy Calculation
        // E_local = 0.5 * ( q*phi - p.E - (1/3) * Q : gradE )
        T term_q = qi * final_pot;
        T term_p = 0; 
        if (rank >= 1) term_p = p_x*final_Ex + p_y*final_Ey + p_z*final_Ez;
        T term_Q = 0;
        if (rank >= 2 && Q) {
            const T* Qi = &Q[9*i];
            term_Q = Qi[0]*G00 + Qi[1]*G01 + Qi[2]*G02 +
                     Qi[3]*G10 + Qi[4]*G11 + Qi[5]*G12 +
                     Qi[6]*G20 + Qi[7]*G21 + Qi[8]*G22;
        }
        local_energy_contribution = 0.5 * (term_q - term_p - term_Q/3.0);
    }

    __shared__ T ssum[256];
    const int tid = threadIdx.x;
    ssum[tid] = local_energy_contribution;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(total_energy_out, ssum[0]);
    }
}

// ============================================================================
// HOST ENTRY POINT
// ============================================================================

struct EwaldLongRangeAllFunctionCuda : public torch::autograd::Function<EwaldLongRangeAllFunctionCuda> {
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor coords, at::Tensor box, at::Tensor q, at::Tensor p, at::Tensor t,
      at::Tensor K_t, at::Tensor rank_t, at::Tensor alpha_t) {
    
    int64_t K   = K_t.item<int64_t>();
    int64_t rank= rank_t.item<int64_t>();
    double alpha= alpha_t.item<double>();

    at::cuda::CUDAGuard guard(coords.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = coords.options();
    const int64_t N = coords.size(0);

    // 1. Host side volume and box copy
    double h_box[9];
    auto box_c = box.contiguous().cpu();
    memcpy(h_box, box_c.data_ptr<double>(), 9*sizeof(double));
    double V = h_box[0]*(h_box[4]*h_box[8]-h_box[5]*h_box[7]) - h_box[1]*(h_box[3]*h_box[8]-h_box[5]*h_box[6]) + h_box[2]*(h_box[3]*h_box[7]-h_box[4]*h_box[6]);

    // 2. Prepare Constants)
    auto recip = at::empty({3,3}, opts);
    reciprocal_box_kernel<double><<<1,1,0,stream>>>(box.data_ptr<double>(), recip.data_ptr<double>(), V);

    const int R  = 2*int(K) + 1;
    const int64_t M  = (int64_t)(K+1) * R * R;
    const int64_t M1 = M - 1;

    auto kvec = at::empty({3, M1}, opts);
    auto gauss = at::empty({M1}, opts);
    auto sym   = at::empty({M1}, opts);
    
    {
        dim3 block(256), grid((unsigned)((M + block.x - 1)/block.x));
        prepare_k_constants_kernel<double><<<grid, block, 0, stream>>>(
            (int)K, (double)alpha, (size_t)M1,
            recip.data_ptr<double>(),
            kvec.data_ptr<double>(),
            gauss.data_ptr<double>(),
            sym.data_ptr<double>()
        );
    }

    // 3. Structure Factor
    auto Sreal = at::zeros({M1}, opts);
    auto Simag = at::zeros({M1}, opts);
    {
        dim3 grid((unsigned)M1); // Parallelize over K
        dim3 block(256);         // Threads sum over Atoms
        compute_structure_factor_fused<double><<<grid, block, 0, stream>>>(
            M1, N,
            kvec.data_ptr<double>(),
            coords.data_ptr<double>(),
            q.data_ptr<double>(),
            (rank>=1 && p.defined()) ? p.data_ptr<double>() : nullptr,
            (rank>=2 && t.defined()) ? t.reshape({N,9}).data_ptr<double>() : nullptr,
            (int)rank,
            Sreal.data_ptr<double>(),
            Simag.data_ptr<double>()
        );
    }

    // 4. (Pot++, Forces, Energy)
    auto pot = at::zeros({N}, opts);
    auto field = at::zeros({N,3}, opts);
    auto grad = at::zeros({N,9}, opts);
    auto forces = at::zeros({N,3}, opts);
    auto energy = at::zeros({}, opts);

    {
        dim3 block(256);
        dim3 grid((unsigned)((N + block.x - 1)/block.x));
        compute_ewald_dynamics_fused<double><<<grid, block, 0, stream>>>(
            M1, N, V, alpha,
            kvec.data_ptr<double>(),
            gauss.data_ptr<double>(), sym.data_ptr<double>(),
            Sreal.data_ptr<double>(), Simag.data_ptr<double>(),
            coords.data_ptr<double>(),
            q.data_ptr<double>(),
            (rank>=1 && p.defined()) ? p.data_ptr<double>() : nullptr,
            (rank>=2 && t.defined()) ? t.reshape({N,9}).data_ptr<double>() : nullptr,
            (int)rank,
            pot.data_ptr<double>(),
            field.data_ptr<double>(),
            grad.data_ptr<double>(),
            forces.data_ptr<double>(),
            energy.data_ptr<double>()
        );
    }
    
    // Reshape grad to (N,3,3)
    at::Tensor grad_reshaped = grad.reshape({N,3,3});

#if EWALD_DBG
    if (pot.numel()>=3) {
      std::cout << "[AUTO] pot[0..2]="
                << pot[0].item<double>() << " "
                << pot[1].item<double>() << " "
                << pot[2].item<double>() << "\n";
    }
#endif

    // Save forces for backward
    ctx->save_for_backward({forces});

    // Prepare output list
    torch::autograd::variable_list out(5);
    out[0] = pot;
    out[1] = field;
    out[2] = grad_reshaped;
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
