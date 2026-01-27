#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <cufft.h>
#include <cmath>
#include <stdio.h> 
#include <ATen/cuda/CUDAContext.h>
#ifndef PI
#define PI 3.14159265358979323846
#endif

// ============================================================================
// 1. Device Helper: B-Spline Evaluation (Includes 3rd Derivative)
// ============================================================================
__device__ __forceinline__ void eval_b6_and_derivs(double u, double* val, double* d1, double* d2, double* d3) {
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u3 * u;
    double u5 = u4 * u;

    if (u < 1.0) {
        *val = u5 * (1.0/120.0);
        *d1  = u4 * (1.0/24.0);
        *d2  = u3 * (1.0/6.0);
        *d3  = u2 * 0.5;
    }
    else if (u < 2.0) {
        double u_1 = u - 1.0;
        double u_1_2 = u_1 * u_1;
        double u_1_3 = u_1_2 * u_1;
        double u_1_4 = u_1_3 * u_1;
        double u_1_5 = u_1_4 * u_1;

        *val = u5 * (1.0/120.0) - u_1_5 * (1.0/20.0);
        *d1  = u4 * (1.0/24.0)  - u_1_4 * 0.25;
        *d2  = u3 * (1.0/6.0)   - u_1_3;
        *d3  = u2 * 0.5         - 3.0 * u_1_2;
    }
    else if (u < 3.0) {
        // Range [2, 3)
        double u_1 = u - 1.0;
        double u_1_2 = u_1 * u_1;
        double u_1_4 = u_1_2 * u_1_2;
        double u_1_5 = u_1_4 * u_1;

        double u_2 = u - 2.0;
        double u_2_2 = u_2 * u_2;
        double u_2_4 = u_2_2 * u_2_2;
        double u_2_5 = u_2_4 * u_2;

        *val = u5 * (1.0/120.0) + u_2_5 * 0.125 - u_1_5 * (1.0/20.0);
        *d1  = u4 * (1.0/24.0)  + 0.625 * u_2_4 - u_1_4 * 0.25;
        *d2  = (5.0/3.0)*u3 - 12.0*u2 + 27.0*u - 19.0;
        *d3  = 5.0*u2 - 24.0*u + 27.0;
    }
    else if (u < 4.0) {
        // Range [3, 4)
        double u_1 = u - 1.0;
        double u_1_2 = u_1 * u_1; double u_1_4 = u_1_2 * u_1_2; double u_1_5 = u_1_4 * u_1;

        double u_2 = u - 2.0;
        double u_2_2 = u_2 * u_2; double u_2_4 = u_2_2 * u_2_2; double u_2_5 = u_2_4 * u_2;

        double u_3 = u - 3.0;
        double u_3_2 = u_3 * u_3; double u_3_4 = u_3_2 * u_3_2; double u_3_5 = u_3_4 * u_3;

        *val = u5*(1.0/120.0) - u_3_5*(1.0/6.0) + u_2_5*0.125 - u_1_5*(1.0/20.0);

        *d1 = (-5.0/12.0)*u4 + 6.0*u3 - 31.5*u2 + 71.0*u - 57.75;
        *d2 = (-5.0/3.0)*u3 + 18.0*u2 - 63.0*u + 71.0;
        *d3 = -5.0*u2 + 36.0*u - 63.0;
    }
    else if (u < 5.0) {
        // Range [4, 5)
        *val = u5*(1.0/24.0) - u4 + 9.5*u3 - 44.5*u2 + 102.25*u - 91.45;
        *d1  = (5.0/24.0)*u4 - 4.0*u3 + 28.5*u2 - 89.0*u + 102.25;
        *d2  = (5.0/6.0)*u3 - 12.0*u2 + 57.0*u - 89.0;
        *d3  = 2.5*u2 - 24.0*u + 57.0;
    }
    else if (u < 6.0) {
        // Range [5, 6) 
        *val = -u5*(1.0/120.0) + 0.25*u4 - 3.0*u3 + 18.0*u2 - 54.0*u + 64.8;
        *d1  = -u4*(1.0/24.0) + u3 - 9.0*u2 + 36.0*u - 54.0;
        *d2  = -u3*(1.0/6.0) + 3.0*u2 - 18.0*u + 36.0;
        *d3  = -0.5*u2 + 6.0*u - 18.0;
    }
    else {
        *val = 0.0; *d1 = 0.0; *d2 = 0.0; *d3 = 0.0;
    }
}

// ============================================================================
// 2. Spread Kernel
// ============================================================================
template <int RANK>
__global__ void spread_q_kernel(
    const double* __restrict__ coords,
    const double* __restrict__ Q,
    const double* __restrict__ recip_vecs,
    double* __restrict__ grid,
    int N_atoms,
    int K1, int K2, int K3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;
	
    double n_star[3][3];
    #pragma unroll
    for(int i=0; i<3; i++)
        #pragma unroll
        for(int j=0; j<3; j++)
            n_star[i][j] = recip_vecs[i*3 + j];

    double r[3];
    r[0] = coords[idx * 3 + 0]; r[1] = coords[idx * 3 + 1]; r[2] = coords[idx * 3 + 2];

    int   m_u0[3];
    double u_frac[3];
    double grid_K[3] = {(double)K1, (double)K2, (double)K3};

    #pragma unroll
    for(int i=0; i<3; i++) {
        double val = (n_star[i][0]*r[0] + n_star[i][1]*r[1] + n_star[i][2]*r[2]) * grid_K[i];
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ( (double)m_u0[i] - val) + 3.0;
    }

    double M[3][6], dM[3][6], d2M[3][6], d3M[3][6];
    //Calculate B-Spline and derivatives
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            double u_eval = u_frac[d] + (double)(k - 3);
            eval_b6_and_derivs(u_eval, &M[d][k], &dM[d][k], &d2M[d][k], &d3M[d][k]);
        }
    }
    //Initialize monopoles
    double q_val = Q[idx * 10 + 0];
    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        double Mz = M[2][iz];
        double dMz = (RANK >= 1) ? dM[2][iz] : 0.0;
        double d2Mz = (RANK >= 2) ? d2M[2][iz] : 0.0;

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            double My = M[1][iy];
            double dMy = (RANK >= 1) ? dM[1][iy] : 0.0;
            double d2My = (RANK >= 2) ? d2M[1][iy] : 0.0;

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                double Mx = M[0][ix];
                double dMx = (RANK >= 1) ? dM[0][ix] : 0.0;
                double d2Mx = (RANK >= 2) ? d2M[0][ix] : 0.0;

                double theta = Mx * My * Mz;
                double term = q_val * theta;

                if (RANK >= 1) {
		    //initialize dipoles
        	    double px = Q[idx * 10 + 1]; double py = Q[idx * 10 + 2]; double pz = Q[idx * 10 + 3];
		    //initialize b-spline derivative
                    double dt_du[3];
                    dt_du[0] = dMx * My * Mz;
                    dt_du[1] = Mx * dMy * Mz;
                    dt_du[2] = Mx * My * dMz;

                    double dt_dr[3] = {0,0,0};
                    #pragma unroll
                    for(int i=0; i<3; i++) {
                        dt_dr[0] += n_star[i][0] * grid_K[i] * dt_du[i];
                        dt_dr[1] += n_star[i][1] * grid_K[i] * dt_du[i];
                        dt_dr[2] += n_star[i][2] * grid_K[i] * dt_du[i];
                    }
                    term -= px * dt_dr[0] + py * dt_dr[1] + pz * dt_dr[2];

                    if (RANK >= 2) {
			 //initialize quadrupoles
			 double Qxx = Q[idx * 10 + 4];
			 double Qxy = Q[idx * 10 + 5];
			 double Qxz = Q[idx * 10 + 6];
			 double Qyy = Q[idx * 10 + 7];
			 double Qyz = Q[idx * 10 + 8];
			 double Qzz = Q[idx * 10 + 9];
			 //initialize b-spline double primes
                         double d2t_du2[3];
                         d2t_du2[0] = d2Mx * My * Mz;
                         d2t_du2[1] = Mx * d2My * Mz;
                         d2t_du2[2] = Mx * My * d2Mz;
                         double d2t_du_mix[3];
                         d2t_du_mix[0] = dMx * dMy * Mz;
                         d2t_du_mix[1] = dMx * My * dMz;
                         d2t_du_mix[2] = Mx * dMy * dMz;

                         double Hu_vals[3][3];
                         Hu_vals[0][0] = d2t_du2[0]; Hu_vals[1][1] = d2t_du2[1]; Hu_vals[2][2] = d2t_du2[2];
                         Hu_vals[0][1] = d2t_du_mix[0]; Hu_vals[1][0] = d2t_du_mix[0];
                         Hu_vals[0][2] = d2t_du_mix[1]; Hu_vals[2][0] = d2t_du_mix[1];
                         Hu_vals[1][2] = d2t_du_mix[2]; Hu_vals[2][1] = d2t_du_mix[2];

                         double H_r[3][3] = {0};
                         #pragma unroll
                         for(int i=0; i<3; i++) {
                             #pragma unroll
                             for(int j=0; j<3; j++) {
                                 double val = 0.0;
                                 #pragma unroll
                                 for(int m=0; m<3; m++) {
                                     #pragma unroll
                                     for(int n=0; n<3; n++) {
                                         val += n_star[m][i]*grid_K[m] * n_star[n][j]*grid_K[n] * Hu_vals[m][n];
                                     }
                                 }
                                 H_r[i][j] = val;
                             }
                         }


			 double interaction = Qxx * Hu_vals[0][0] + Qyy * Hu_vals[1][1] + Qzz * Hu_vals[2][2] + 2*(Qxy * Hu_vals[0][1] + Qxz * Hu_vals[0][2] + Qyz * Hu_vals[1][2]);
			 interaction *= 0.5;
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
template <int RANK>
__global__ void interpolate_kernel(
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    const double* __restrict__ recip_vecs,
    const double* __restrict__ Q,
    double* __restrict__ phi_atoms,
    double* __restrict__ E_atoms,
    double* __restrict__ EG_atoms,
    double* __restrict__ force_atoms,
    double alpha,
    int N_atoms,
    int K1, int K2, int K3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_atoms) return;

    // --- 1. Geometry Setup ---
    double n_star[3][3];
    #pragma unroll
    for(int i=0; i<3; i++)
        #pragma unroll
        for(int j=0; j<3; j++)
            n_star[i][j] = recip_vecs[i*3 + j];

    double r[3] = {coords[idx*3+0], coords[idx*3+1], coords[idx*3+2]};
    double grid_K[3] = {(double)K1, (double)K2, (double)K3};

    int m_u0[3];
    double u_frac[3];

    #pragma unroll
    for(int i=0; i<3; i++) {
        double val = (n_star[i][0]*r[0] + n_star[i][1]*r[1] + n_star[i][2]*r[2]) * grid_K[i];
        m_u0[i] = (int)ceil(val);
        u_frac[i] = ((double)m_u0[i] - val) + 3.0;
    }

    // --- 2. B-Spline Evaluation ---
    double M[3][6], dM[3][6], d2M[3][6], d3M[3][6];
    #pragma unroll
    for(int d=0; d<3; d++) {
        #pragma unroll
        for(int k=0; k<6; k++) {
            double u_eval = u_frac[d] + (double)(k - 3);
            eval_b6_and_derivs(u_eval, &M[d][k], &dM[d][k], &d2M[d][k], &d3M[d][k]);
        }
    }

    // --- 3. Multipole Setup ---
    double q_val = Q[idx * 10 + 0];

    // Dipoles
    double p_lat[3] = {0.0};
    if (RANK >= 1) {
        double px = Q[idx * 10 + 1];
        double py = Q[idx * 10 + 2];
        double pz = Q[idx * 10 + 3];
        #pragma unroll
        for(int i=0; i<3; i++)
            p_lat[i] = (px * n_star[i][0] + py * n_star[i][1] + pz * n_star[i][2]) * grid_K[i];
    }

    // Quadrupoles
    double Q_lat[6] = {0.0};
    if (RANK >= 2) {
	//Initialize quadrupoles
        double Qxx = Q[idx * 10 + 4];
        double Qxy = Q[idx * 10 + 5];
        double Qxz = Q[idx * 10 + 6];
        double Qyy = Q[idx * 10 + 7];
        double Qyz = Q[idx * 10 + 8];
        double Qzz = Q[idx * 10 + 9];

        double Q_cart[3][3];
        Q_cart[0][0]=Qxx; Q_cart[0][1]=Qxy; Q_cart[0][2]=Qxz;
        Q_cart[1][0]=Qxy; Q_cart[1][1]=Qyy; Q_cart[1][2]=Qyz;
        Q_cart[2][0]=Qxz; Q_cart[2][1]=Qyz; Q_cart[2][2]=Qzz;

        double A[3][3];
        #pragma unroll
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                A[i][j] = grid_K[i] * n_star[i][j];

        double Q_temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_temp[i][j] += Q_cart[i][k] * A[j][k];

        double Q_L[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    Q_L[i][j] += A[i][k] * Q_temp[k][j];

        Q_lat[0]=Q_L[0][0]; Q_lat[1]=Q_L[1][1]; Q_lat[2]=Q_L[2][2];
        Q_lat[3]=Q_L[0][1]; Q_lat[4]=Q_L[0][2]; Q_lat[5]=Q_L[1][2];
    }

    // --- 4. Grid Accumulation ---
    double phi_acc = 0.0;
    double grad_lat[3] = {0.0};
    double hess_p_lat[3] = {0.0};
    double grad_Q_lat[3] = {0.0};
    double hess_lat[6] = {0.0};

    #pragma unroll 6
    for (int iz = 0; iz < 6; iz++) {
        int gz = (m_u0[2] + (iz - 3) + 1000 * K3) % K3;
        double Mz = M[2][iz]; double dMz = dM[2][iz];
        double d2Mz = (RANK>=1)?d2M[2][iz]:0; double d3Mz = (RANK>=2)?d3M[2][iz]:0;

        #pragma unroll 6
        for (int iy = 0; iy < 6; iy++) {
            int gy = (m_u0[1] + (iy - 3) + 1000 * K2) % K2;
            double My = M[1][iy]; double dMy = dM[1][iy];
            double d2My = (RANK>=1)?d2M[1][iy]:0; double d3My = (RANK>=2)?d3M[1][iy]:0;

            #pragma unroll 6
            for (int ix = 0; ix < 6; ix++) {
                int gx = (m_u0[0] + (ix - 3) + 1000 * K1) % K1;
                double Mx = M[0][ix]; double dMx = dM[0][ix];
                double d2Mx = (RANK>=1)?d2M[0][ix]:0; double d3Mx = (RANK>=2)?d3M[0][ix]:0;

                double val = grid[gx * K2 * K3 + gy * K3 + gz];

                phi_acc      += (Mx * My * Mz) * val;
                grad_lat[0]  += (dMx * My * Mz) * val;
                grad_lat[1]  += (Mx * dMy * Mz) * val;
                grad_lat[2]  += (Mx * My * dMz) * val;

                if (RANK >= 1) {
                    double H_uu = d2Mx * My * Mz;
                    double H_vv = Mx * d2My * Mz;
                    double H_ww = Mx * My * d2Mz;
                    double H_uv = dMx * dMy * Mz;
                    double H_uw = dMx * My * dMz;
                    double H_vw = Mx * dMy * dMz;

                    hess_lat[0] += H_uu * val; hess_lat[1] += H_vv * val; hess_lat[2] += H_ww * val;
                    hess_lat[3] += H_uv * val; hess_lat[4] += H_uw * val; hess_lat[5] += H_vw * val;

                    hess_p_lat[0] += (p_lat[0]*H_uu + p_lat[1]*H_uv + p_lat[2]*H_uw) * val;
                    hess_p_lat[1] += (p_lat[0]*H_uv + p_lat[1]*H_vv + p_lat[2]*H_vw) * val;
                    hess_p_lat[2] += (p_lat[0]*H_uw + p_lat[1]*H_vw + p_lat[2]*H_ww) * val;

                    if (RANK >= 2) {
                        double T_uuu = d3Mx * My * Mz;
                        double T_vvv = Mx * d3My * Mz;
                        double T_www = Mx * My * d3Mz;
                        double T_uuv = d2Mx * dMy * Mz;
                        double T_uuw = d2Mx * My * dMz;
                        double T_uvv = dMx * d2My * Mz;
                        double T_vvw = Mx * d2My * dMz;
                        double T_uww = dMx * My * d2Mz;
                        double T_vww = Mx * dMy * d2Mz;
                        double T_uvw = dMx * dMy * dMz;

                        double dE_du = 0.5 * (Q_lat[0]*T_uuu + Q_lat[1]*T_uvv + Q_lat[2]*T_uww +
                                              2.0*(Q_lat[3]*T_uuv + Q_lat[4]*T_uuw + Q_lat[5]*T_uvw));
                        double dE_dv = 0.5 * (Q_lat[0]*T_uuv + Q_lat[1]*T_vvv + Q_lat[2]*T_vww +
                                              2.0*(Q_lat[3]*T_uvv + Q_lat[4]*T_uvw + Q_lat[5]*T_vvw));
                        double dE_dw = 0.5 * (Q_lat[0]*T_uuw + Q_lat[1]*T_vvw + Q_lat[2]*T_www +
                                              2.0*(Q_lat[3]*T_uvw + Q_lat[4]*T_uww + Q_lat[5]*T_vww));

			double factor = 2.0;
                        grad_Q_lat[0] += dE_du * val * factor;
                        grad_Q_lat[1] += dE_dv * val * factor;
                        grad_Q_lat[2] += dE_dw * val * factor;
                    }
                }
            }
        }
    }

    // --- 5.Cartesian Transform ---
    // Output Potential
    phi_atoms[idx] = phi_acc;

    // Transform Gradients (Lattice -> Cartesian)
    double grad_cart[3] = {0.0};
    #pragma unroll
    for(int x=0; x<3; x++) {
        grad_cart[x] = grad_lat[0] * (n_star[0][x] * grid_K[0]) +
                       grad_lat[1] * (n_star[1][x] * grid_K[1]) +
                       grad_lat[2] * (n_star[2][x] * grid_K[2]);
    }

    if (RANK >= 1) {
        E_atoms[idx * 3 + 0] = -grad_cart[0];
        E_atoms[idx * 3 + 1] = -grad_cart[1];
        E_atoms[idx * 3 + 2] = -grad_cart[2];
    }

    double grad_U_dip[3] = {0.0};
    if (RANK >= 1) {
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_dip[x] = hess_p_lat[0] * (n_star[0][x] * grid_K[0]) +
                            hess_p_lat[1] * (n_star[1][x] * grid_K[1]) +
                            hess_p_lat[2] * (n_star[2][x] * grid_K[2]);
        }
    }

    double grad_U_quad[3] = {0.0};
    if (RANK >= 2) {
        // Transform Force from Quadrupoles
        #pragma unroll
        for(int x=0; x<3; x++) {
            grad_U_quad[x] = grad_Q_lat[0] * (n_star[0][x] * grid_K[0]) +
                             grad_Q_lat[1] * (n_star[1][x] * grid_K[1]) +
                             grad_Q_lat[2] * (n_star[2][x] * grid_K[2]);
        }

        // Transform EFG
        double H_lat_mat[3][3];
        H_lat_mat[0][0]=hess_lat[0]; H_lat_mat[1][1]=hess_lat[1]; H_lat_mat[2][2]=hess_lat[2];
        H_lat_mat[0][1]=H_lat_mat[1][0]=hess_lat[3];
        H_lat_mat[0][2]=H_lat_mat[2][0]=hess_lat[4];
        H_lat_mat[1][2]=H_lat_mat[2][1]=hess_lat[5];

        double A_mat[3][3];
        for(int i=0; i<3; i++) for(int j=0; j<3; j++) A_mat[i][j] = grid_K[i] * n_star[i][j];

        double temp[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    temp[i][j] += A_mat[k][i] * H_lat_mat[k][j];

        double EFG[3][3] = {0};
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    EFG[i][j] += temp[i][k] * A_mat[k][j];

        EG_atoms[idx*6 + 0] = EFG[0][0];
        EG_atoms[idx*6 + 1] = EFG[0][1];
        EG_atoms[idx*6 + 2] = EFG[0][2];
        EG_atoms[idx*6 + 3] = EFG[1][1];
        EG_atoms[idx*6 + 4] = EFG[1][2];
        EG_atoms[idx*6 + 5] = EFG[2][2];
    }

    force_atoms[idx*3 + 0] = (q_val * grad_cart[0] + grad_U_dip[0] + grad_U_quad[0]);
    force_atoms[idx*3 + 1] = (q_val * grad_cart[1] + grad_U_dip[1] + grad_U_quad[1]);
    force_atoms[idx*3 + 2] = (q_val * grad_cart[2] + grad_U_dip[2] + grad_U_quad[2]);
}
// ============================================================================
// 4. Combined B-Spline Helper & Convolution
// ============================================================================
__device__ __forceinline__ double get_bspline_coeff_order6(int i) {
    switch(i) {
        case 1: return 1.0 / 120.0;
        case 2: return 26.0 / 120.0;
        case 3: return 66.0 / 120.0; 
        case 4: return 26.0 / 120.0;
        case 5: return 1.0 / 120.0;
        default: return 0.0; 
    }
}
__device__ __forceinline__ double get_bspline_modulus_device(int k, int K, int order) {
    double sum_val = 0.0;
    int half = order / 2; 
    for (int m = -half; m < half; m++) {
        double b_val = (order == 6) ? get_bspline_coeff_order6(m + half) : 0.0;
        double arg = (2.0 * PI * (double)m * (double)k) / (double)K;
        sum_val += b_val * cos(arg);
    }
    return sum_val;
}

__global__ void pme_convolution_fused_kernel(
    cufftDoubleComplex* __restrict__ d_grid,
    const double* __restrict__ d_recip,
    int K1, int K2, int K3,
    double alpha,
    double V
) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    int K3_complex = K3 / 2 + 1;
    if (idx_x >= K1 || idx_y >= K2 || idx_z >= K3_complex) return;
    int flat_idx = idx_x * K2 * K3_complex + idx_y * K3_complex + idx_z;

    if (idx_x == 0 && idx_y == 0 && idx_z == 0) {
        d_grid[flat_idx].x = 0.0; d_grid[flat_idx].y = 0.0; return;
    }
    double mx = (idx_x <= K1/2) ? (double)idx_x : (double)(idx_x - K1);
    double my = (idx_y <= K2/2) ? (double)idx_y : (double)(idx_y - K2);
    double mz = (double)idx_z; 
    double kx = 2.0 * PI * (mx * d_recip[0] + my * d_recip[3] + mz * d_recip[6]);
    double ky = 2.0 * PI * (mx * d_recip[1] + my * d_recip[4] + mz * d_recip[7]);
    double kz = 2.0 * PI * (mx * d_recip[2] + my * d_recip[5] + mz * d_recip[8]);
    double ksq = kx*kx + ky*ky + kz*kz;
    double C_k = (4.0 * PI / (V * ksq)) * exp(-ksq / (4.0 * alpha * alpha));
    double theta_x = get_bspline_modulus_device(idx_x, K1, 6);
    double theta_y = get_bspline_modulus_device(idx_y, K2, 6);
    double theta_z = get_bspline_modulus_device(idx_z, K3, 6);
    double theta = theta_x * theta_y * theta_z;
    double theta_sq = theta * theta;
    double scale_factor = (1.0 / theta_sq);
    double factor = C_k * scale_factor;
    d_grid[flat_idx].x *= factor;
    d_grid[flat_idx].y *= factor;
}

// ============================================================================
// 5. Host Pipeline
// ============================================================================
void compute_pme_cuda_pipeline(
    torch::Tensor coords, torch::Tensor Q, torch::Tensor recip_vecs,
    torch::Tensor grid_tensor, torch::Tensor phi_atoms,
    torch::Tensor E_atoms, torch::Tensor EG_atoms,
    torch::Tensor force_atoms,
    double alpha, double volume, int K1, int K2, int K3, int rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    double* d_coords = coords.data_ptr<double>();
    double* d_Q      = Q.data_ptr<double>();
    double* d_recip  = recip_vecs.data_ptr<double>();
    double* d_grid_real = grid_tensor.data_ptr<double>();
    double* d_phi = phi_atoms.data_ptr<double>();
    double* d_E   = E_atoms.data_ptr<double>();
    double* d_EG  = EG_atoms.data_ptr<double>();
    double* d_force = force_atoms.data_ptr<double>();

    int N_atoms = coords.size(0);
    int total_points = K1*K2*K3;
    int K3_complex = K3 / 2 + 1;

    auto grid_complex_tensor = torch::empty({K1 * K2 * K3_complex * 2},
        torch::TensorOptions().dtype(torch::kDouble).device(coords.device()));
    cufftDoubleComplex* d_grid_complex = (cufftDoubleComplex*)grid_complex_tensor.data_ptr<double>();

    cufftHandle plan_fwd, plan_bwd;
    cufftPlan3d(&plan_fwd, K1, K2, K3, CUFFT_D2Z);
    cufftPlan3d(&plan_bwd, K1, K2, K3, CUFFT_Z2D);
    cufftSetStream(plan_fwd, stream);
    cufftSetStream(plan_bwd, stream);

    // 1. Spread
    cudaMemsetAsync(d_grid_real, 0, sizeof(double)*total_points, stream);
    int threads = 256;
    int blocks = (N_atoms + threads - 1) / threads;
    
    if (rank == 0) {
        spread_q_kernel<0><<<blocks, threads, 0, stream>>>(d_coords, d_Q, d_recip, d_grid_real, N_atoms, K1, K2, K3);
    } else if (rank == 1) {
        spread_q_kernel<1><<<blocks, threads, 0, stream>>>(d_coords, d_Q, d_recip, d_grid_real, N_atoms, K1, K2, K3);
    } else {
        spread_q_kernel<2><<<blocks, threads, 0, stream>>>(d_coords, d_Q, d_recip, d_grid_real, N_atoms, K1, K2, K3);
    }

    // 2. FFTs
    cufftExecD2Z(plan_fwd, (cufftDoubleReal*)d_grid_real, d_grid_complex);
    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid((K1+7)/8, (K2+7)/8, (K3_complex+7)/8);
    pme_convolution_fused_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        d_grid_complex, d_recip, K1, K2, K3, alpha, volume);
    cufftExecZ2D(plan_bwd, d_grid_complex, (cufftDoubleReal*)d_grid_real);

    // 3. Interpolate (AND FORCES)
    if (rank == 0) {
        interpolate_kernel<0><<<blocks, threads, 0, stream>>>(d_grid_real, d_coords, d_recip, d_Q, d_phi, d_E, d_EG, d_force, alpha, N_atoms, K1, K2, K3);
    } else if (rank == 1) {
        interpolate_kernel<1><<<blocks, threads, 0, stream>>>(d_grid_real, d_coords, d_recip, d_Q, d_phi, d_E, d_EG, d_force, alpha, N_atoms, K1, K2, K3);
    } else {
        interpolate_kernel<2><<<blocks, threads, 0, stream>>>(d_grid_real, d_coords, d_recip, d_Q, d_phi, d_E, d_EG, d_force, alpha, N_atoms, K1, K2, K3);
    }

    cudaDeviceSynchronize();
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);
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

    // Prepare Q
    at::Tensor q_view = q.view({-1, 1});
    at::Tensor Q_padded;
    if (rank == 0)      Q_padded = torch::cat({q_view, torch::zeros({N, 9}, coords.options())}, 1);
    else if (rank == 1) Q_padded = torch::cat({q_view, p, torch::zeros({N, 6}, coords.options())}, 1);
    else                Q_padded = torch::cat({q_view, p, t}, 1);

    at::Tensor Q_combined = Q_padded.contiguous();
    auto options = coords.options();

    at::Tensor phi = torch::zeros({N}, options);
    at::Tensor E   = torch::zeros({N, 3}, options);
    at::Tensor EG  = torch::zeros({N, 6}, options);
    at::Tensor forces = torch::zeros({N, 3}, options); 
    at::Tensor grid_scratch = torch::zeros({K1, K2, K3}, options);

    // Call Pipeline with Forces
    compute_pme_cuda_pipeline(
        coords, Q_combined, recip_vecs, grid_scratch,
        phi, E, EG, forces, 
        alpha, volume, K1, K2, K3, rank
    );

    double pi = CUDART_PI;
    double alpha_over_root_pi = alpha / sqrt(pi);
    double alpha2 = alpha * alpha;

    // A. Potential Correction
    phi.sub_(q * (2.0 * alpha_over_root_pi));
    // --- Rank 1: Dipole Correction ---
    if (rank >= 1) {
        double factor_E = alpha_over_root_pi * (4.0 * alpha2 / 3.0);

        // 1. Apply correction to ALL atoms
        E.add_(p * factor_E);

        // 2. Print ONLY for Atom 0
        // We calculate the specific correction vector for the first atom: (p[0] * factor)
        // We use .cpu() to ensure we can print the tensor data from the Host.
        std::cout << "DEBUG Atom 0 [Dipole Correction]: \n"
                  << "   Factor: " << factor_E << "\n"
                  << "   Vector: " << (p[0] * factor_E).cpu() << std::endl;
    }

    // --- Rank 2: Quadrupole Correction ---
    if (rank >= 2) {
        double alpha4 = alpha2 * alpha2;
        double factor_EG = alpha_over_root_pi * (16.0 * alpha4 / 5.0) / 3.0;

        // 1. Apply correction to ALL atoms
        EG.add_(t * factor_EG);

        // 2. Print ONLY for Atom 0
        std::cout << "DEBUG Atom 0 [Quad Correction]: \n"
                  << "   Factor: " << factor_EG << "\n"
                  << "   Vector: " << (t[0] * factor_EG).cpu() << std::endl;
    }
    at::Tensor energy = assemble_pme_energy_only(q, p, t, phi, E, EG, rank);
    
    return {phi, E, EG, energy, forces};
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
    return {torch::zeros_like(grad_outputs[1]), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
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
