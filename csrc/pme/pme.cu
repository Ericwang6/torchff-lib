#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <cmath>
#include <stdio.h> 
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include "bsplines.cuh"
#include "common/constants.cuh"

// ============================================================================
// 2. Spread Kernel
// ============================================================================
template <typename T, int RANK>
__global__ void spread_q_kernel(
    const T* __restrict__ coords,
    const T* __restrict__ Q,
    const T* __restrict__ recip_vecs,
    T* __restrict__ grid,
    int N_atoms,
    int K1, int K2, int K3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;

    T n_star[3][3];
    #pragma unroll
    for(int i=0; i<3; i++)
        #pragma unroll
        for(int j=0; j<3; j++)
            n_star[i][j] = recip_vecs[i*3 + j];

    T r[3];
    r[0] = coords[idx * 3 + 0]; r[1] = coords[idx * 3 + 1]; r[2] = coords[idx * 3 + 2];

    int   m_u0[3];
    T u_frac[3];
    T grid_K[3] = {(T)K1, (T)K2, (T)K3};

    #pragma unroll
    for(int i=0; i<3; i++) {
        T val = (n_star[i][0]*r[0] + n_star[i][1]*r[1] + n_star[i][2]*r[2]) * grid_K[i];
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ( (T)m_u0[i] - val) + (T)3.0;
    }

    T M[3][6], dM[3][6], d2M[3][6];
    // Calculate B-Spline and derivatives
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            T u_eval = u_frac[d] + (T)(k - 3);
            eval_b6_and_derivs<T>(u_eval, &M[d][k], &dM[d][k], &d2M[d][k]);
        }
    }
    // Initialize monopoles
    T q_val = Q[idx * 10 + 0];
    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        T Mz = M[2][iz];
        T dMz = (RANK >= 1) ? dM[2][iz] : (T)0.0;
        T d2Mz = (RANK >= 2) ? d2M[2][iz] : (T)0.0;

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            T My = M[1][iy];
            T dMy = (RANK >= 1) ? dM[1][iy] : (T)0.0;
            T d2My = (RANK >= 2) ? d2M[1][iy] : (T)0.0;

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                T Mx = M[0][ix];
                T dMx = (RANK >= 1) ? dM[0][ix] : (T)0.0;
                T d2Mx = (RANK >= 2) ? d2M[0][ix] : (T)0.0;

                T theta = Mx * My * Mz;
                T term = q_val * theta;

                if (RANK >= 1) {
                    // Initialize dipoles
                    T px = Q[idx * 10 + 1]; T py = Q[idx * 10 + 2]; T pz = Q[idx * 10 + 3];
                    // Initialize b-spline derivative
                    T dt_du[3];
                    dt_du[0] = dMx * My * Mz;
                    dt_du[1] = Mx * dMy * Mz;
                    dt_du[2] = Mx * My * dMz;

                    T dt_dr[3] = {(T)0,(T)0,(T)0};
                    #pragma unroll
                    for(int i=0; i<3; i++) {
                        dt_dr[0] += n_star[i][0] * grid_K[i] * dt_du[i];
                        dt_dr[1] += n_star[i][1] * grid_K[i] * dt_du[i];
                        dt_dr[2] += n_star[i][2] * grid_K[i] * dt_du[i];
                    }
                    term -= px * dt_dr[0] + py * dt_dr[1] + pz * dt_dr[2];
                    if (RANK >= 2) {
                        // 1. Load Cartesian Quadrupoles
                        T Qxx = Q[idx * 10 + 4];
                        T Qxy = Q[idx * 10 + 5];
                        T Qxz = Q[idx * 10 + 6];
                        T Qyy = Q[idx * 10 + 7];
                        T Qyz = Q[idx * 10 + 8];
                        T Qzz = Q[idx * 10 + 9];

                        // 2. Compute Transformation Matrix A = K * n_star
                        // A maps Cartesian coords (r) to Grid coords (u)
                        T A[3][3];
                        #pragma unroll
                        for(int i=0; i<3; i++)
                            for(int j=0; j<3; j++)
                                A[i][j] = grid_K[i] * n_star[i][j];

                        // 3. Transform Q_cart to Q_lat
                        // Q_lat = A * Q_cart * A^T
                        T Q_cart[3][3];
                        Q_cart[0][0]=Qxx; Q_cart[0][1]=Qxy; Q_cart[0][2]=Qxz;
                        Q_cart[1][0]=Qxy; Q_cart[1][1]=Qyy; Q_cart[1][2]=Qyz;
                        Q_cart[2][0]=Qxz; Q_cart[2][1]=Qyz; Q_cart[2][2]=Qzz;

                        T Q_temp[3][3] = {(T)0};
                        #pragma unroll
                        for(int i=0; i<3; i++)
                            for(int j=0; j<3; j++)
                                for(int k=0; k<3; k++)
                                    Q_temp[i][j] += Q_cart[i][k] * A[j][k]; // Q * A^T (symmetric)

                        T Q_lat[3][3] = {(T)0};
                        #pragma unroll
                        for(int i=0; i<3; i++)
                            for(int j=0; j<3; j++)
                                for(int k=0; k<3; k++)
                                    Q_lat[i][j] += A[i][k] * Q_temp[k][j]; // A * (Q * A^T)

                        T Q_lat_xx = Q_lat[0][0]; T Q_lat_yy = Q_lat[1][1]; T Q_lat_zz = Q_lat[2][2];
                        T Q_lat_xy = Q_lat[0][1]; T Q_lat_xz = Q_lat[0][2]; T Q_lat_yz = Q_lat[1][2];

                        // 4. Compute B-Spline Lattice Hessians (Hu_vals)
                        T d2t_du2[3];
                        d2t_du2[0] = d2Mx * My * Mz;
                        d2t_du2[1] = Mx * d2My * Mz;
                        d2t_du2[2] = Mx * My * d2Mz;
                        T d2t_du_mix[3];
                        d2t_du_mix[0] = dMx * dMy * Mz;
                        d2t_du_mix[1] = dMx * My * dMz;
                        d2t_du_mix[2] = Mx * dMy * dMz;

                        T Hu_vals[3][3];
                        Hu_vals[0][0] = d2t_du2[0]; Hu_vals[1][1] = d2t_du2[1]; Hu_vals[2][2] = d2t_du2[2];
                        Hu_vals[0][1] = d2t_du_mix[0];
                        Hu_vals[0][2] = d2t_du_mix[1];
                        Hu_vals[1][2] = d2t_du_mix[2];

                        // 5. Contract Lattice Quadrupole with Lattice Hessian
                        T interaction = Q_lat_xx * Hu_vals[0][0] +
                                         Q_lat_yy * Hu_vals[1][1] +
                                         Q_lat_zz * Hu_vals[2][2] +
                                         (T)2.0 * (Q_lat_xy * Hu_vals[0][1] +
                                         Q_lat_xz * Hu_vals[0][2] +
                                         Q_lat_yz * Hu_vals[1][2]);

                        interaction *= (T)0.5;
                        term += interaction;
                    }
                }

                int grid_ptr = gx * K2 * K3 + gy * K3 + gz;
                atomicAdd(&grid[grid_ptr], term);
            }
        }
    }
}
// ============================================================================
// 3. Interpolate back to atoms potentials, fields, field gradients, and compute forces
// ============================================================================
template <typename T, int RANK>
__global__ void interpolate_kernel(
    const T* __restrict__ grid,
    const T* __restrict__ coords,
    const T* __restrict__ recip_vecs,
    const T* __restrict__ Q,
    T* __restrict__ phi_atoms,
    T* __restrict__ E_atoms,
    T* __restrict__ EG_atoms,
    T* __restrict__ force_atoms,
    T alpha,
    int N_atoms,
    int K1, int K2, int K3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;

    // --- 1. Geometry Setup ---
    T n_star[3][3];
    #pragma unroll
    for(int i=0; i<3; i++)
        #pragma unroll
        for(int j=0; j<3; j++)
            n_star[i][j] = recip_vecs[i*3 + j];

    T r[3] = {coords[idx*3+0], coords[idx*3+1], coords[idx*3+2]};
    T grid_K[3] = {(T)K1, (T)K2, (T)K3};

    int m_u0[3];
    T u_frac[3];

    #pragma unroll
    for(int i=0; i<3; i++) {
        T val = (n_star[i][0]*r[0] + n_star[i][1]*r[1] + n_star[i][2]*r[2]) * grid_K[i];
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ((T)m_u0[i] - val) + (T)3.0;
    }

    // --- 2. B-Spline Evaluation ---
    T M[3][6], dM[3][6], d2M[3][6], d3M[3][6];
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            T u_eval = u_frac[d] + (T)(k - 3);
            eval_b6_and_derivs<T>(u_eval, &M[d][k], &dM[d][k], &d2M[d][k], &d3M[d][k]);
        }
    }

    // --- 3. Multipole Setup ---
    T q_val = Q[idx * 10 + 0];

    // Dipoles
    T p_lat[3] = {(T)0.0};
    if (RANK >= 1) {
        T px = Q[idx * 10 + 1];
        T py = Q[idx * 10 + 2];
        T pz = Q[idx * 10 + 3];
        #pragma unroll
        for(int i=0; i<3; i++)
            p_lat[i] = (px * n_star[i][0] + py * n_star[i][1] + pz * n_star[i][2]) * grid_K[i];
    }

    // Quadrupoles
    T Q_lat[6] = {(T)0.0};
    if (RANK >= 2) {
        // Initialize quadrupoles
        T Qxx = Q[idx * 10 + 4];
        T Qxy = Q[idx * 10 + 5];
        T Qxz = Q[idx * 10 + 6];
        T Qyy = Q[idx * 10 + 7];
        T Qyz = Q[idx * 10 + 8];
        T Qzz = Q[idx * 10 + 9];

        T Q_cart[3][3];
        Q_cart[0][0]=Qxx; Q_cart[0][1]=Qxy; Q_cart[0][2]=Qxz;
        Q_cart[1][0]=Qxy; Q_cart[1][1]=Qyy; Q_cart[1][2]=Qyz;
        Q_cart[2][0]=Qxz; Q_cart[2][1]=Qyz; Q_cart[2][2]=Qzz;

        T A[3][3];
        #pragma unroll
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                A[i][j] = grid_K[i] * n_star[i][j];

        T Q_temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_temp[i][j] += Q_cart[i][k] * A[j][k];

        T Q_L[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_L[i][j] += A[i][k] * Q_temp[k][j];

        Q_lat[0]=Q_L[0][0]; Q_lat[1]=Q_L[1][1]; Q_lat[2]=Q_L[2][2];
        Q_lat[3]=Q_L[0][1]; Q_lat[4]=Q_L[0][2]; Q_lat[5]=Q_L[1][2];
    }

    // --- 4. Grid Accumulation ---
    T phi_acc = (T)0.0;
    T grad_lat[3] = {(T)0.0};
    T hess_p_lat[3] = {(T)0.0};
    T grad_Q_lat[3] = {(T)0.0};
    T hess_lat[6] = {(T)0.0};

    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        T Mz = M[2][iz]; T dMz = dM[2][iz];
        T d2Mz = (RANK>=1)?d2M[2][iz]:(T)0; T d3Mz = (RANK>=2)?d3M[2][iz]:(T)0;

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            T My = M[1][iy]; T dMy = dM[1][iy];
            T d2My = (RANK>=1)?d2M[1][iy]:(T)0; T d3My = (RANK>=2)?d3M[1][iy]:(T)0;

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                T Mx = M[0][ix]; T dMx = dM[0][ix];
                T d2Mx = (RANK>=1)?d2M[0][ix]:(T)0; T d3Mx = (RANK>=2)?d3M[0][ix]:(T)0;

                T val = grid[gx * K2 * K3 + gy * K3 + gz];

                phi_acc      += (Mx * My * Mz) * val;
                grad_lat[0]  += (dMx * My * Mz) * val;
                grad_lat[1]  += (Mx * dMy * Mz) * val;
                grad_lat[2]  += (Mx * My * dMz) * val;

                if (RANK >= 1) {
                    T H_uu = d2Mx * My * Mz;
                    T H_vv = Mx * d2My * Mz;
                    T H_ww = Mx * My * d2Mz;
                    T H_uv = dMx * dMy * Mz;
                    T H_uw = dMx * My * dMz;
                    T H_vw = Mx * dMy * dMz;

                    hess_lat[0] += H_uu * val; hess_lat[1] += H_vv * val; hess_lat[2] += H_ww * val;
                    hess_lat[3] += H_uv * val; hess_lat[4] += H_uw * val; hess_lat[5] += H_vw * val;

                    hess_p_lat[0] += (p_lat[0]*H_uu + p_lat[1]*H_uv + p_lat[2]*H_uw) * val;
                    hess_p_lat[1] += (p_lat[0]*H_uv + p_lat[1]*H_vv + p_lat[2]*H_vw) * val;
                    hess_p_lat[2] += (p_lat[0]*H_uw + p_lat[1]*H_vw + p_lat[2]*H_ww) * val;

                    if (RANK >= 2) {
                        T T_uuu = d3Mx * My * Mz;
                        T T_vvv = Mx * d3My * Mz;
                        T T_www = Mx * My * d3Mz;
                        T T_uuv = d2Mx * dMy * Mz;
                        T T_uuw = d2Mx * My * dMz;
                        T T_uvv = dMx * d2My * Mz;
                        T T_vvw = Mx * d2My * dMz;
                        T T_uww = dMx * My * d2Mz;
                        T T_vww = Mx * dMy * d2Mz;
                        T T_uvw = dMx * dMy * dMz;

                        T dE_du = (T)0.5 * (Q_lat[0]*T_uuu + Q_lat[1]*T_uvv + Q_lat[2]*T_uww +
                                              (T)2.0*(Q_lat[3]*T_uuv + Q_lat[4]*T_uuw + Q_lat[5]*T_uvw));
                        T dE_dv = (T)0.5 * (Q_lat[0]*T_uuv + Q_lat[1]*T_vvv + Q_lat[2]*T_vww +
                                              (T)2.0*(Q_lat[3]*T_uvv + Q_lat[4]*T_uvw + Q_lat[5]*T_vvw));
                        T dE_dw = (T)0.5 * (Q_lat[0]*T_uuw + Q_lat[1]*T_vvw + Q_lat[2]*T_www +
                                              (T)2.0*(Q_lat[3]*T_uvw + Q_lat[4]*T_uww + Q_lat[5]*T_vww));

                        grad_Q_lat[0] += dE_du * val;
                        grad_Q_lat[1] += dE_dv * val;
                        grad_Q_lat[2] += dE_dw * val;
                    }
                }
            }
        }
    }

    // --- 5. Cartesian Transform ---
    // Output Potential
    phi_atoms[idx] = phi_acc;

    // Transform Gradients (Lattice -> Cartesian)
    T grad_cart[3] = {(T)0.0};
    #pragma unroll
    for(int x=0; x<3; x++) {
        grad_cart[x] = grad_lat[0] * (n_star[0][x] * grid_K[0]) +
                       grad_lat[1] * (n_star[1][x] * grid_K[1]) +
                       grad_lat[2] * (n_star[2][x] * grid_K[2]);
    }

    if (RANK >= 1) {
        E_atoms[idx * 3 + 0] = grad_cart[0];
        E_atoms[idx * 3 + 1] = grad_cart[1];
        E_atoms[idx * 3 + 2] = grad_cart[2];
    }

    T grad_U_dip[3] = {(T)0.0};
    if (RANK >= 1) {
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_dip[x] = hess_p_lat[0] * (n_star[0][x] * grid_K[0]) +
                            hess_p_lat[1] * (n_star[1][x] * grid_K[1]) +
                            hess_p_lat[2] * (n_star[2][x] * grid_K[2]);
        }
    }

    T grad_U_quad[3] = {(T)0.0};
    if (RANK >= 2) {
        // Transform Force from Quadrupoles
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_quad[x] = grad_Q_lat[0] * (n_star[0][x] * grid_K[0]) +
                             grad_Q_lat[1] * (n_star[1][x] * grid_K[1]) +
                             grad_Q_lat[2] * (n_star[2][x] * grid_K[2]);
        }

        // Transform EFG
        T H_lat_mat[3][3];
        H_lat_mat[0][0]=hess_lat[0]; H_lat_mat[1][1]=hess_lat[1]; H_lat_mat[2][2]=hess_lat[2];
        H_lat_mat[0][1]=H_lat_mat[1][0]=hess_lat[3];
        H_lat_mat[0][2]=H_lat_mat[2][0]=hess_lat[4];
        H_lat_mat[1][2]=H_lat_mat[2][1]=hess_lat[5];

        T A_mat[3][3];
        for(int i=0; i<3; i++) for(int j=0; j<3; j++) A_mat[i][j] = grid_K[i] * n_star[i][j];

        T temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    temp[i][j] += A_mat[k][i] * H_lat_mat[k][j];

        T EFG[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    EFG[i][j] += temp[i][k] * A_mat[k][j];
        T factor = (T)-0.5;
        EG_atoms[idx*6 + 0] = EFG[0][0] * factor;
        EG_atoms[idx*6 + 1] = EFG[0][1] * factor;
        EG_atoms[idx*6 + 2] = EFG[0][2] * factor;
        EG_atoms[idx*6 + 3] = EFG[1][1] * factor;
        EG_atoms[idx*6 + 4] = EFG[1][2] * factor;
        EG_atoms[idx*6 + 5] = EFG[2][2] * factor;
    }

    force_atoms[idx*3 + 0] = (q_val * grad_cart[0] - grad_U_dip[0] + grad_U_quad[0]);
    force_atoms[idx*3 + 1] = (q_val * grad_cart[1] - grad_U_dip[1] + grad_U_quad[1]);
    force_atoms[idx*3 + 2] = (q_val * grad_cart[2] - grad_U_dip[2] + grad_U_quad[2]);
}

template <typename T>
__global__ void pme_convolution_fused_kernel(
    c10::complex<T>* __restrict__ grid_recip,
    const T* __restrict__ d_recip,
    int K1, int K2, int K3,
    T alpha,
    T V
) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    int K3_complex = K3 / 2 + 1;
    if (idx_x >= K1 || idx_y >= K2 || idx_z >= K3_complex) return;
    int flat_idx = idx_x * K2 * K3_complex + idx_y * K3_complex + idx_z;

    if (idx_x == 0 && idx_y == 0 && idx_z == 0) {
        grid_recip[flat_idx] = c10::complex<T>((T)0.0, (T)0.0); return;
    }
    constexpr T TWOPI = two_pi<T>();
    T mx = (idx_x <= K1/2) ? (T)idx_x : (T)(idx_x - K1);
    T my = (idx_y <= K2/2) ? (T)idx_y : (T)(idx_y - K2);
    T mz = (T)idx_z; 
    T kx = TWOPI * (mx * d_recip[0] + my * d_recip[3] + mz * d_recip[6]);
    T ky = TWOPI * (mx * d_recip[1] + my * d_recip[4] + mz * d_recip[7]);
    T kz = TWOPI * (mx * d_recip[2] + my * d_recip[5] + mz * d_recip[8]);
    T ksq = kx*kx + ky*ky + kz*kz;
    T C_k = ((T)2.0 * TWOPI / (V * ksq)) * exp(-ksq / ((T)4.0 * alpha * alpha));
    T theta_x = get_bspline_modulus_device<T>(idx_x, K1, 6);
    T theta_y = get_bspline_modulus_device<T>(idx_y, K2, 6);
    T theta_z = get_bspline_modulus_device<T>(idx_z, K3, 6);
    T theta = theta_x * theta_y * theta_z;
    T theta_sq = theta * theta;
    T scale_factor = ((T)1.0 / theta_sq);
    T factor = C_k * scale_factor;
    grid_recip[flat_idx] *= factor;
}

// ============================================================================
// 5. Host Pipeline
// ============================================================================
template <int RANK>
void compute_pme_cuda_pipeline(
    torch::Tensor coords, torch::Tensor Q, torch::Tensor recip_vecs,
    torch::Tensor phi_atoms,
    torch::Tensor E_atoms, torch::Tensor EG_atoms,
    torch::Tensor force_atoms,
    double alpha, double volume, int K1, int K2, int K3
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    double* d_coords = coords.data_ptr<double>();
    double* d_Q      = Q.data_ptr<double>();
    double* d_recip  = recip_vecs.data_ptr<double>();
    double* d_phi   = phi_atoms.data_ptr<double>();
    double* d_E     = E_atoms.data_ptr<double>();
    double* d_EG    = EG_atoms.data_ptr<double>();
    double* d_force = force_atoms.data_ptr<double>();

    int N_atoms = coords.size(0);
    int K3_complex = K3 / 2 + 1;

    // 1. Allocate 3D grid (contiguous) and spread charges
    auto grid_3d = torch::zeros({K1, K2, K3}, coords.options());
    double* d_grid_real = grid_3d.data_ptr<double>();
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = (N_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    spread_q_kernel<double, RANK><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_coords, d_Q, d_recip, d_grid_real, N_atoms, K1, K2, K3);

    // 2. Forward FFT (real to complex)
    auto grid_recip = torch::fft::rfftn(grid_3d).contiguous();
    c10::complex<double>* grid_recip_ptr = grid_recip.data_ptr<c10::complex<double>>();
    
    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid((K1+7)/8, (K2+7)/8, (K3_complex+7)/8);
    pme_convolution_fused_kernel<double><<<dimGrid, dimBlock, 0, stream>>>(grid_recip_ptr, d_recip, K1, K2, K3, alpha, volume);

    // 3. Inverse FFT (complex to real)
    auto grid_real = torch::fft::irfftn(grid_recip, {K1, K2, K3}, c10::nullopt, "forward");
    double* grid_real_ptr = grid_real.data_ptr<double>();

    // 4. Interpolate (AND FORCES)
    interpolate_kernel<double, RANK><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(grid_real_ptr, d_coords, d_recip, d_Q, d_phi, d_E, d_EG, d_force, alpha, N_atoms, K1, K2, K3);

}

// ============================================================================
// 6. Helper Function & Wrapper
// ============================================================================
at::Tensor assemble_pme_energy_only(
    const at::Tensor& q, const at::Tensor& p, const at::Tensor& t,
    const at::Tensor& phi, const at::Tensor& E, const at::Tensor& dE,
    int rank
) {
    at::Tensor energy_term = q.view({-1}) * phi; 
    
    if (rank >= 1) {
        energy_term -= (p * E).sum(1);
    }
    if (rank >= 2) {
        at::Tensor prod = t * dE;
        at::Tensor sum_diag = prod.select(1, 0) +
                              prod.select(1, 3) +
                              prod.select(1, 5);
        at::Tensor sum_off  = prod.select(1, 1) +
                              prod.select(1, 2) +
                              prod.select(1, 4);
        // Combine: Diagonals * 1.0 + Off-Diagonals * 2.0
        energy_term -= (sum_diag + 2.0 * sum_off);
    }
    
    return 0.5 * energy_term.sum();
}
struct PMELongRangeFunction : public torch::autograd::Function<PMELongRangeFunction> {

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor coords, at::Tensor box, at::Tensor q, at::Tensor p, at::Tensor t,
        at::Tensor K_t, at::Tensor rank_t, at::Tensor alpha_t
    ) {
        TORCH_CHECK(coords.scalar_type() == at::kDouble, "Coords must be double");
        TORCH_CHECK(q.scalar_type() == at::kDouble, "Charges must be double");

        double alpha = alpha_t.item<double>();
        int rank = rank_t.item<int64_t>();

        // --- 1. Setup Grid & Geometry ---
        int K1, K2, K3;
        if (K_t.numel() == 1) {
            K1 = K2 = K3 = K_t.item<int64_t>();
        } else {
            auto k_acc = K_t.accessor<int64_t, 1>();
            K1 = k_acc[0]; K2 = k_acc[1]; K3 = k_acc[2];
        }
        at::Tensor recip_vecs = torch::inverse(box).t().contiguous();
        double volume = torch::det(box).item<double>();
        int N = coords.size(0);

        // --- 2. Prepare Multipole Tensor ---
        at::Tensor q_view = q.view({-1, 1});
        at::Tensor Q_padded;
        if (rank == 0)      Q_padded = torch::cat({q_view, torch::zeros({N, 9}, coords.options())}, 1);
        else if (rank == 1) Q_padded = torch::cat({q_view, p, torch::zeros({N, 6}, coords.options())}, 1);
        else                Q_padded = torch::cat({q_view, p, t}, 1);

        at::Tensor Q_combined = Q_padded.contiguous();
        auto options = coords.options();

        // --- 3. Allocate Outputs ---
        at::Tensor phi = torch::zeros({N}, options);
        at::Tensor E   = torch::zeros({N, 3}, options);
        at::Tensor EG  = torch::zeros({N, 6}, options); // Flat (xx, xy, xz, yy, yz, zz)
        at::Tensor forces = torch::zeros({N, 3}, options);

        // --- 4. Run CUDA Pipeline (allocates its own grid internally) ---
        if (rank == 0) {
            compute_pme_cuda_pipeline<0>(coords, Q_combined, recip_vecs, phi, E, EG, forces, alpha, volume, K1, K2, K3);
        } else if (rank == 1) {
            compute_pme_cuda_pipeline<1>(coords, Q_combined, recip_vecs, phi, E, EG, forces, alpha, volume, K1, K2, K3);
        } else {
            compute_pme_cuda_pipeline<2>(coords, Q_combined, recip_vecs, phi, E, EG, forces, alpha, volume, K1, K2, K3);
        }

        // --- 5. Apply Self-Corrections ---
        constexpr double INV_ROOT_PI = inv_root_pi<double>();
        double alpha_over_root_pi = alpha * INV_ROOT_PI;
        double alpha2 = alpha * alpha;

        phi.sub_(q * (2.0 * alpha_over_root_pi));

        if (rank >= 1) {
            double factor_E = alpha_over_root_pi * (4.0 * alpha2 / 3.0);
            E.add_(p * factor_E); // self-field
        }

        if (rank >= 2) {
            double alpha4 = alpha2 * alpha2;
            double factor_EG = alpha_over_root_pi * (16.0 * alpha4 / 5.0) / 3.0;
            EG.add_(t * factor_EG); // self-field-gradient
        }

        // --- 6. Compute Total Energy ---
        at::Tensor energy = assemble_pme_energy_only(q, p, t, phi, E, EG, rank);

        // --- 7. Reshape EG to (N,3,3) for Backward Pass ---
        at::Tensor EG_reshaped = torch::zeros({N, 3, 3}, options);
        if (rank >= 2) {
            // EG is stored as: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
            at::Tensor xx = EG.select(1, 0);
            at::Tensor xy = EG.select(1, 1);
            at::Tensor xz = EG.select(1, 2);
            at::Tensor yy = EG.select(1, 3);
            at::Tensor yz = EG.select(1, 4);
            at::Tensor zz = EG.select(1, 5);

            // Row 0
            EG_reshaped.select(1, 0).select(1, 0).copy_(xx);
            EG_reshaped.select(1, 0).select(1, 1).copy_(xy);
            EG_reshaped.select(1, 0).select(1, 2).copy_(xz);
            // Row 1 (Symmetric)
            EG_reshaped.select(1, 1).select(1, 0).copy_(xy);
            EG_reshaped.select(1, 1).select(1, 1).copy_(yy);
            EG_reshaped.select(1, 1).select(1, 2).copy_(yz);
            // Row 2 (Symmetric)
            EG_reshaped.select(1, 2).select(1, 0).copy_(xz);
            EG_reshaped.select(1, 2).select(1, 1).copy_(yz);
            EG_reshaped.select(1, 2).select(1, 2).copy_(zz);
        }

        // --- 8. Save Variables for Backward ---
        // We save: Forces, Field(E), Field Gradient(EG_3x3), Rank, Alpha
        ctx->save_for_backward({forces, E, EG_reshaped, rank_t, alpha_t, p, t});

        return {phi, E, EG_reshaped, energy, forces};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
        // grad_outputs order: [0:phi, 1:E, 2:EG, 3:energy, 4:forces]

        // We care about derivatives w.r.t Energy (scalar) and potentially Field (polarization)
        const at::Tensor& g_energy = grad_outputs[3];
        const at::Tensor& g_field  = grad_outputs[1];

        // Retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        const at::Tensor& forces_internal = saved[0]; // (N,3) Calculated Forces (-dE/dx)
        const at::Tensor& field           = saved[1]; // (N,3) Electric Field
        const at::Tensor& field_grad      = saved[2]; // (N,3,3) Hessian
        const at::Tensor& rank_t          = saved[3];
        const at::Tensor& alpha_t         = saved[4];
        const at::Tensor& p               = saved[5];
        const at::Tensor& t               = saved[6];

        int64_t rank = rank_t.item<int64_t>();
        double alpha = alpha_t.item<double>();
        constexpr double INV_ROOT_PI = inv_root_pi<double>();
        double alpha_over_root_pi = alpha * INV_ROOT_PI;
        double alpha2 = alpha * alpha;

        // --- A. Compute d_coords (Forces on Atoms) ---
        at::Tensor dcoords = torch::zeros_like(forces_internal);

        // 1. Contribution from Energy Gradient
        if (g_energy.defined()) {
            auto scale = (g_energy.dim()==0) ? g_energy.view({1,1}) : g_energy;
            // PME returns force F = -dE/dx.
            // We want dL/dx = dL/dE * dE/dx = g_energy * (-F)
            dcoords.sub_(forces_internal * scale);
        }

        // --- B. Compute d_p (Torque on Dipoles) ---
        at::Tensor d_p;
        if (rank >= 1 && g_energy.defined()) {
            auto scale = (g_energy.dim()==0) ? g_energy : g_energy.view({1,1});
            // dE/dp = -Field
            // dL/dp = dL/dE * -E
            double factor_E = alpha_over_root_pi * (4.0 * alpha2 / 3.0);

            // dL/dp = g_energy * (-Field - 0.5 * factor_E * p)
            d_p = (-field - (0.5 * factor_E * p)) * scale;
        }

        // --- C. Compute d_t (Torque on Quadrupoles) ---
        at::Tensor d_t;
        if (rank >= 2 && g_energy.defined()) {
            auto scale = (g_energy.dim() == 0) ? g_energy : g_energy.view({1, 1});
            double alpha4 = alpha2 * alpha2;
            double factor_EG = alpha_over_root_pi * (16.0 * alpha4 / 5.0) / 3.0;
            // Extract field gradient components from the (N,3,3) tensor
            auto EG_xx = field_grad.select(1, 0).select(1, 0);
            auto EG_xy = field_grad.select(1, 0).select(1, 1);
            auto EG_xz = field_grad.select(1, 0).select(1, 2);
            auto EG_yy = field_grad.select(1, 1).select(1, 1);
            auto EG_yz = field_grad.select(1, 1).select(1, 2);
            auto EG_zz = field_grad.select(1, 2).select(1, 2);

            // Apply 2.0 to off-diagonals and 0.5 * factor to diagonals
            auto d_t_xx = (-EG_xx - 0.5 * factor_EG * t.select(1, 0)) * scale;
            auto d_t_xy = (-EG_xy * 2.0) * scale;
            auto d_t_xz = (-EG_xz * 2.0) * scale;
            auto d_t_yy = (-EG_yy - 0.5 * factor_EG * t.select(1, 3)) * scale;
            auto d_t_yz = (-EG_yz * 2.0) * scale;
            auto d_t_zz = (-EG_zz - 0.5 * factor_EG * t.select(1, 5)) * scale;

            d_t = torch::stack({d_t_xx, d_t_xy, d_t_xz, d_t_yy, d_t_yz, d_t_zz}, 1);
        }

        // --- Output List ---
        return {
            dcoords,        // coords
            at::Tensor(),   // box
            at::Tensor(),   // q (monopoles)
            d_p,            // p (Dipoles)
            d_t,            // t (Quadrupoles)
            at::Tensor(),   // K
            at::Tensor(),   // rank
            at::Tensor()    // alpha
        };
    }
};


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("pme_long_range",
        [](const at::Tensor& coords, const at::Tensor& box, const at::Tensor& q, const at::Tensor& p, const at::Tensor& t, int64_t K, int64_t rank, double alpha) {
            auto K_t     = at::scalar_tensor(K,     at::TensorOptions().dtype(at::kLong).device(coords.device()));
            auto rank_t  = at::scalar_tensor(rank,  at::TensorOptions().dtype(at::kLong).device(coords.device()));
            auto alpha_t = at::scalar_tensor(alpha, at::TensorOptions().dtype(at::kDouble).device(coords.device()));
            auto outs = PMELongRangeFunction::apply(coords, box, q, p, t, K_t, rank_t, alpha_t);
            return std::make_tuple(outs[0], outs[1], outs[2], outs[3], outs[4]);
        });
}
