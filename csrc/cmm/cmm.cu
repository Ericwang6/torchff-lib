#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/vec3.cuh"
#include "common/pbc.cuh"
#include "common/switch.cuh"
#include "multipoles/multipoles.cuh"
#include "dispersion/tang_tonnies.cuh"
#include "damps.cuh"


template <typename scalar_t>
__global__ void cmm_nonbonded_non_elec_interaction_from_pairs_kernel(
    scalar_t* dist_vecs,
    int32_t* pairs,
    scalar_t* multipoles,
    scalar_t* q_pauli, scalar_t* Kdipo_pauli, scalar_t* Kquad_pauli, scalar_t* b_pauli_ij,
    scalar_t* q_xpol, scalar_t* Kdipo_xpol, scalar_t* Kquad_xpol, scalar_t* b_xpol_ij,
    scalar_t* q_ct_don, scalar_t* Kdipo_ct_don, scalar_t* Kquad_ct_don, 
    scalar_t* q_ct_acc, scalar_t* Kdipo_ct_acc, scalar_t* Kquad_ct_acc, scalar_t* b_ct_ij, scalar_t* eps_ct_ij,
    scalar_t* C6_disp_ij, scalar_t* b_disp_ij,
    scalar_t rcut_sr,
    scalar_t rcut_lr,
    scalar_t rcut_switch_buf,
    int32_t npairs,
    scalar_t* ene,
    scalar_t* dist_vecs_grad,
    scalar_t* multipoles_grad,
    scalar_t* q_pauli_grad,
    scalar_t* dq_a
)
{
    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < npairs; index += gridDim.x * blockDim.x) {
        int32_t i = pairs[index * 2];
        int32_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        scalar_t e = scalar_t(0.0);
        scalar_t mi[10]; 
        scalar_t mj[10]; 

        #pragma unroll
        for (int n = 0; n < 10; n++) {
            mi[n] = multipoles[i*10+n];
            mj[n] = multipoles[j*10+n];
        }

        scalar_t mi_grad[10] = {};
        scalar_t mj_grad[10] = {};
        scalar_t drx_grad = scalar_t(0.0);
        scalar_t dry_grad = scalar_t(0.0);
        scalar_t drz_grad = scalar_t(0.0);

        scalar_t mi_grad_tmp[10];
        scalar_t mj_grad_tmp[10];
        scalar_t drx_grad_tmp ;
        scalar_t dry_grad_tmp ;
        scalar_t drz_grad_tmp ;

        scalar_t damps[6];
        scalar_t kdi, kdj, kqi, kqj;

        scalar_t drx = dist_vecs[index*3];
        scalar_t dry = dist_vecs[index*3+1];
        scalar_t drz = dist_vecs[index*3+2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        scalar_t etmp = 0.0;

        constexpr scalar_t ZERO = scalar_t(0.0);
        constexpr scalar_t ONE = scalar_t(1.0);

        // short-range switch
        scalar_t s, sg;
        s = clamp_((dr - rcut_sr + rcut_switch_buf) / rcut_switch_buf, ZERO, ONE);
        sg = SWITCH_GRAD(s) / rcut_switch_buf / dr;
        s = SWITCH(s);
        // s = ONE; sg = ZERO;

        // Pauli
        kdi = Kdipo_pauli[i]; kdj = Kdipo_pauli[j];
        kqi = Kquad_pauli[i]; kqj = Kquad_pauli[j];
        two_center_damps(dr, b_pauli_ij[index], damps);
        pairwise_multipole_kernel_with_grad(
            q_pauli[i], mi[1]*kdi, mi[2]*kdi, mi[3]*kdi, mi[4]*kqi, mi[5]*kqi, mi[6]*kqi, mi[7]*kqi, mi[8]*kqi, mi[9]*kqi,
            q_pauli[j], mj[1]*kdj, mj[2]*kdj, mj[3]*kdj, mj[4]*kqj, mj[5]*kqj, mj[6]*kqj, mj[7]*kqj, mj[8]*kqj, mj[9]*kqj,
            drx, dry, drz,
            damps[0], damps[1], damps[2], damps[3], damps[4], damps[5],
            &etmp,
            mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
            mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
            &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp
        );
        e += etmp;
        atomicAdd(&q_pauli_grad[i], mi_grad_tmp[0]*s);
        atomicAdd(&q_pauli_grad[j], mj_grad_tmp[0]*s);

        #pragma unroll
        for (int n = 1; n < 4; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kdi;
            mj_grad[n] += mj_grad_tmp[n] * kdj;
        }
        #pragma unroll
        for (int n = 4; n < 10; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kqi;
            mj_grad[n] += mj_grad_tmp[n] * kqj;
        }
        drx_grad += drx_grad_tmp; dry_grad += dry_grad_tmp; drz_grad += drz_grad_tmp; 

        // XPOL
        kdi = Kdipo_xpol[i]; kdj = Kdipo_xpol[j];
        kqi = Kquad_xpol[i]; kqj = Kquad_xpol[j];
        two_center_damps(dr, b_xpol_ij[index], damps);
        pairwise_multipole_kernel_with_grad(
            q_xpol[i], mi[1]*kdi, mi[2]*kdi, mi[3]*kdi, mi[4]*kqi, mi[5]*kqi, mi[6]*kqi, mi[7]*kqi, mi[8]*kqi, mi[9]*kqi,
            q_xpol[j], mj[1]*kdj, mj[2]*kdj, mj[3]*kdj, mj[4]*kqj, mj[5]*kqj, mj[6]*kqj, mj[7]*kqj, mj[8]*kqj, mj[9]*kqj,
            drx, dry, drz,
            -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
            &etmp,
            mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
            mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
            &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp
        );
        e += etmp;

        #pragma unroll
        for (int n = 1; n < 4; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kdi;
            mj_grad[n] += mj_grad_tmp[n] * kdj;
        }
        #pragma unroll
        for (int n = 4; n < 10; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kqi;
            mj_grad[n] += mj_grad_tmp[n] * kqj;
        }
        drx_grad += drx_grad_tmp; dry_grad += dry_grad_tmp; drz_grad += drz_grad_tmp; 

        // CT
        kdi = Kdipo_ct_acc[i]; kdj = Kdipo_ct_don[j];
        kqi = Kquad_ct_acc[i]; kqj = Kquad_ct_don[j];
        two_center_damps(dr, b_ct_ij[index], damps);
        pairwise_multipole_kernel_with_grad(
            q_ct_acc[i], mi[1]*kdi, mi[2]*kdi, mi[3]*kdi, mi[4]*kqi, mi[5]*kqi, mi[6]*kqi, mi[7]*kqi, mi[8]*kqi, mi[9]*kqi,
            q_ct_don[j], mj[1]*kdj, mj[2]*kdj, mj[3]*kdj, mj[4]*kqj, mj[5]*kqj, mj[6]*kqj, mj[7]*kqj, mj[8]*kqj, mj[9]*kqj,
            drx, dry, drz,
            -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
            &etmp,
            mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
            mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
            &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp
        );

        #pragma unroll
        for (int n = 1; n < 4; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kdi;
            mj_grad[n] += mj_grad_tmp[n] * kdj;
        }
        #pragma unroll
        for (int n = 4; n < 10; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kqi;
            mj_grad[n] += mj_grad_tmp[n] * kqj;
        }
        e += etmp;
        drx_grad += drx_grad_tmp; dry_grad += dry_grad_tmp; drz_grad += drz_grad_tmp; 

        kdi = Kdipo_ct_don[i]; kdj = Kdipo_ct_acc[j];
        kqi = Kquad_ct_don[i]; kqj = Kquad_ct_acc[j];
        two_center_damps(dr, b_ct_ij[index], damps);
        pairwise_multipole_kernel_with_grad(
            q_ct_don[i], mi[1]*kdi, mi[2]*kdi, mi[3]*kdi, mi[4]*kqi, mi[5]*kqi, mi[6]*kqi, mi[7]*kqi, mi[8]*kqi, mi[9]*kqi,
            q_ct_acc[j], mj[1]*kdj, mj[2]*kdj, mj[3]*kdj, mj[4]*kqj, mj[5]*kqj, mj[6]*kqj, mj[7]*kqj, mj[8]*kqj, mj[9]*kqj,
            drx, dry, drz,
            -damps[0], -damps[1], -damps[2], -damps[3], -damps[4], -damps[5],
            &etmp,
            mi_grad_tmp, mi_grad_tmp+1, mi_grad_tmp+2, mi_grad_tmp+3, mi_grad_tmp+4, mi_grad_tmp+5, mi_grad_tmp+6, mi_grad_tmp+7, mi_grad_tmp+8, mi_grad_tmp+9,
            mj_grad_tmp, mj_grad_tmp+1, mj_grad_tmp+2, mj_grad_tmp+3, mj_grad_tmp+4, mj_grad_tmp+5, mj_grad_tmp+6, mj_grad_tmp+7, mj_grad_tmp+8, mj_grad_tmp+9,
            &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp
        );
        e += etmp;

        #pragma unroll
        for (int n = 1; n < 4; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kdi;
            mj_grad[n] += mj_grad_tmp[n] * kdj;
        }
        #pragma unroll
        for (int n = 4; n < 10; n++) {
            mi_grad[n] += mi_grad_tmp[n] * kqi;
            mj_grad[n] += mj_grad_tmp[n] * kqj;
        }
        drx_grad += drx_grad_tmp; dry_grad += dry_grad_tmp; drz_grad += drz_grad_tmp; 

        // switching functions
        drx_grad = drx_grad * s + e * sg * drx;
        dry_grad = dry_grad * s + e * sg * dry;
        drz_grad = drz_grad * s + e * sg * drz;
        e = e * s;

        #pragma unroll
        for (int n = 1; n < 10; n++) {
            atomicAdd(&multipoles_grad[i*10+n], mi_grad[n]*s);
            atomicAdd(&multipoles_grad[j*10+n], mj_grad[n]*s);
        }

        // transferred-charges
        scalar_t dqij = (q_ct_don[i] * q_ct_acc[j] - q_ct_don[j] * q_ct_acc[i]) / dr * (-damps[0]) * eps_ct_ij[index] * s;
        atomicAdd(&dq_a[j], dqij);
        atomicAdd(&dq_a[i], -dqij);
        
        // Dispersion
        scalar_t c6_disp = C6_disp_ij[index];
        scalar_t b_disp = b_disp_ij[index];
        tang_tonnies_6_dispersion(c6_disp, b_disp, dr, drx, dry, drz, &etmp, &drx_grad_tmp, &dry_grad_tmp, &drz_grad_tmp);

        s = clamp_((dr - rcut_lr + rcut_switch_buf) / rcut_switch_buf, ZERO, ONE);
        sg = SWITCH_GRAD(s) / rcut_switch_buf / dr;
        s = SWITCH(s);
        // s = ONE;
        // sg = ZERO;

        e += etmp * s;

        drx_grad += drx_grad_tmp * s + etmp * sg * drx;
        dry_grad += dry_grad_tmp * s + etmp * sg * dry;
        drz_grad += drz_grad_tmp * s + etmp * sg * drz;

        ene[index] = e;
        dist_vecs_grad[index*3] = drx_grad;
        dist_vecs_grad[index*3+1] = dry_grad;
        dist_vecs_grad[index*3+2] = drz_grad;
    }
}

template <typename scalar_t>
__global__ void cmm_charge_transfer_backward_kernel(
    scalar_t* dist_vecs,
    int32_t* pairs,
    scalar_t* q_ct_don, scalar_t* q_ct_acc, scalar_t* b_ct_ij, scalar_t* eps_ct_ij,
    scalar_t* dq_a_grad,
    scalar_t rcut_sr,
    scalar_t rcut_switch_buf,
    int32_t npairs,
    scalar_t* dist_vecs_grad
)
{
    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < npairs; index += gridDim.x * blockDim.x) {
        int32_t i = pairs[index * 2];
        int32_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
    
        // distance 
        scalar_t drx = dist_vecs[index*3];
        scalar_t dry = dist_vecs[index*3+1];
        scalar_t drz = dist_vecs[index*3+2];
        scalar_t dr = sqrt_(drx*drx+dry*dry+drz*drz);

        // short-range switch
        constexpr scalar_t ZERO = scalar_t(0.0);
        constexpr scalar_t ONE = scalar_t(1.0);

        scalar_t s, sg;
        s = clamp_((dr - rcut_sr + rcut_switch_buf) / rcut_switch_buf, ZERO, ONE);
        sg = SWITCH_GRAD(s) / rcut_switch_buf / dr;
        s = SWITCH(s);

        scalar_t u = b_ct_ij[index] * dr;
        scalar_t u2 = u * u;
        scalar_t expu = exp_(-u);

        constexpr scalar_t c1_2 = scalar_t(1.0/2.0);
        constexpr scalar_t c11_16 = scalar_t(11.0/16.0);
        constexpr scalar_t c3_16 = scalar_t(3.0/16.0);
        constexpr scalar_t c1_48 = scalar_t(1.0/48.0);
        constexpr scalar_t c7_48 = scalar_t(7.0/48.0);

        scalar_t d0 = -expu * (1 + u * c11_16 + u2 * c3_16 + u2*u * c1_48);
        scalar_t d1 = -expu * ( 1 + u + u2 * c1_2 + u2*u * c7_48 + u2*u2 * c1_48);

        scalar_t tmp = (q_ct_don[i] * q_ct_acc[j] - q_ct_don[j] * q_ct_acc[i]) * eps_ct_ij[index];
        scalar_t dqij = tmp / dr * d0;
        scalar_t dqij_drij = - tmp / (dr*dr*dr) * d1;
        scalar_t dqij_drij_x = (s * dqij_drij + dqij * sg) * drx;
        scalar_t dqij_drij_y = (s * dqij_drij + dqij * sg) * dry;
        scalar_t dqij_drij_z = (s * dqij_drij + dqij * sg) * drz;
        
        scalar_t dq_grad_ij = dq_a_grad[j] - dq_a_grad[i];

        dist_vecs_grad[index*3]   += dq_grad_ij * dqij_drij_x;
        dist_vecs_grad[index*3+1] += dq_grad_ij * dqij_drij_y;
        dist_vecs_grad[index*3+2] += dq_grad_ij * dqij_drij_z;
    }
}



class CMMNonElecNonbondedFromPairsFunctionCuda: public torch::autograd::Function<CMMNonElecNonbondedFromPairsFunctionCuda> {

public: 

static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& dist_vecs,
    at::Tensor& pairs,
    at::Tensor& multipoles,
    at::Tensor& q_pauli, at::Tensor& Kdipo_pauli, at::Tensor& Kquad_pauli, at::Tensor& b_pauli_ij,
    at::Tensor& q_xpol, at::Tensor& Kdipo_xpol, at::Tensor& Kquad_xpol, at::Tensor& b_xpol_ij,
    at::Tensor& q_ct_don, at::Tensor& Kdipo_ct_don, at::Tensor& Kquad_ct_don, 
    at::Tensor& q_ct_acc, at::Tensor& Kdipo_ct_acc, at::Tensor& Kquad_ct_acc, at::Tensor& b_ct_ij, at::Tensor& eps_ct_ij,
    at::Tensor& C6_disp_ij, at::Tensor& b_disp_ij,
    at::Scalar rcut_sr, at::Scalar rcut_lr, at::Scalar rcut_switch_buf
)
{
    int32_t npairs = pairs.size(0);
    at::Tensor ene = at::zeros({npairs}, dist_vecs.options());

    at::Tensor dist_vecs_grad = at::zeros_like(dist_vecs, dist_vecs.options());
    at::Tensor multipoles_grad = at::zeros_like(multipoles, multipoles.options());
    at::Tensor q_pauli_grad = at::zeros_like(q_pauli, q_pauli.options());
    at::Tensor dq_a = at::zeros_like(q_pauli, q_pauli.options());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = 128;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (npairs+block_dim-1)/block_dim);

    AT_DISPATCH_FLOATING_TYPES(dist_vecs.scalar_type(), "cmm_nonbonded_non_elec_interaction_from_pairs_kernel", ([&] {
        cmm_nonbonded_non_elec_interaction_from_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            dist_vecs.data_ptr<scalar_t>(),
            pairs.data_ptr<int32_t>(),
            multipoles.data_ptr<scalar_t>(),
            q_pauli.data_ptr<scalar_t>(), Kdipo_pauli.data_ptr<scalar_t>(), Kquad_pauli.data_ptr<scalar_t>(), b_pauli_ij.data_ptr<scalar_t>(),
            q_xpol.data_ptr<scalar_t>(), Kdipo_xpol.data_ptr<scalar_t>(), Kquad_xpol.data_ptr<scalar_t>(), b_xpol_ij.data_ptr<scalar_t>(),
            q_ct_don.data_ptr<scalar_t>(), Kdipo_ct_don.data_ptr<scalar_t>(), Kquad_ct_don.data_ptr<scalar_t>(), 
            q_ct_acc.data_ptr<scalar_t>(), Kdipo_ct_acc.data_ptr<scalar_t>(), Kquad_ct_acc.data_ptr<scalar_t>(), b_ct_ij.data_ptr<scalar_t>(),
            eps_ct_ij.data_ptr<scalar_t>(),
            C6_disp_ij.data_ptr<scalar_t>(), b_disp_ij.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rcut_sr.toDouble()),
            static_cast<scalar_t>(rcut_lr.toDouble()),
            static_cast<scalar_t>(rcut_switch_buf.toDouble()),
            npairs,
            ene.data_ptr<scalar_t>(),
            dist_vecs_grad.data_ptr<scalar_t>(),
            multipoles_grad.data_ptr<scalar_t>(),
            q_pauli_grad.data_ptr<scalar_t>(),
            dq_a.data_ptr<scalar_t>()
        );
    }));
    ctx->save_for_backward({
        dist_vecs_grad, multipoles_grad, q_pauli_grad, 
        dist_vecs, pairs, q_ct_don, q_ct_acc, b_ct_ij, eps_ct_ij
    });
    ctx->saved_data["rcut_sr"] = rcut_sr;
    ctx->saved_data["rcut_switch_buf"] = rcut_switch_buf;

    std::vector<at::Tensor> outs;
    outs.reserve(2);
    outs.push_back(at::sum(ene));
    outs.push_back(dq_a);

    return outs;
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();
    int32_t npairs = saved[4].size(0);
    at::Tensor dist_vecs_grad = saved[0] * grad_outputs[0];
    at::Tensor ignore;
    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = 128;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (npairs+block_dim-1)/block_dim);
    AT_DISPATCH_FLOATING_TYPES(dist_vecs_grad.scalar_type(), "cmm_charge_transfer_backward_kernel", ([&] {
        cmm_charge_transfer_backward_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            saved[3].data_ptr<scalar_t>(),
            saved[4].data_ptr<int32_t>(),
            saved[5].data_ptr<scalar_t>(), 
            saved[6].data_ptr<scalar_t>(), 
            saved[7].data_ptr<scalar_t>(), 
            saved[8].data_ptr<scalar_t>(),
            grad_outputs[1].contiguous().data_ptr<scalar_t>(), 
            static_cast<scalar_t>(ctx->saved_data["rcut_sr"].toDouble()),
            static_cast<scalar_t>(ctx->saved_data["rcut_switch_buf"].toDouble()),
            npairs,
            dist_vecs_grad.data_ptr<scalar_t>()
        );
    }));
    return {
        dist_vecs_grad, // dist_vecs grad
        ignore,
        saved[1] * grad_outputs[0], // multipoles grad
        saved[2] * grad_outputs[0], ignore, ignore, ignore,
        ignore, ignore, ignore, ignore,
        ignore, ignore, ignore,
        ignore, ignore, ignore, ignore, ignore,
        ignore, ignore,
        ignore, ignore, ignore
    };
}

};


std::tuple<at::Tensor, at::Tensor> cmm_non_elec_nonbonded_interaction_from_pairs_cuda(
    at::Tensor& dist_vecs,
    at::Tensor& pairs,
    at::Tensor& multipoles,
    at::Tensor& q_pauli, at::Tensor& Kdipo_pauli, at::Tensor& Kquad_pauli, at::Tensor& b_pauli_ij,
    at::Tensor& q_xpol, at::Tensor& Kdipo_xpol, at::Tensor& Kquad_xpol, at::Tensor& b_xpol_ij,
    at::Tensor& q_ct_don, at::Tensor& Kdipo_ct_don, at::Tensor& Kquad_ct_don, 
    at::Tensor& q_ct_acc, at::Tensor& Kdipo_ct_acc, at::Tensor& Kquad_ct_acc, at::Tensor& b_ct_ij, at::Tensor& eps_ct_ij,
    at::Tensor& C6_disp_ij, at::Tensor& b_disp_ij,
    at::Scalar rcut_sr, at::Scalar rcut_lr, at::Scalar rcut_switch_buf
){
    
    auto outs = CMMNonElecNonbondedFromPairsFunctionCuda::apply(
        dist_vecs,
        pairs,
        multipoles,
        q_pauli, Kdipo_pauli, Kquad_pauli, b_pauli_ij,
        q_xpol, Kdipo_xpol, Kquad_xpol, b_xpol_ij,
        q_ct_don, Kdipo_ct_don, Kquad_ct_don, 
        q_ct_acc, Kdipo_ct_acc, Kquad_ct_acc, b_ct_ij, eps_ct_ij,
        C6_disp_ij, b_disp_ij,
        rcut_sr, rcut_lr, rcut_switch_buf
    );
    return {outs[0], outs[1]};
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_non_elec_nonbonded_interaction_from_pairs", cmm_non_elec_nonbonded_interaction_from_pairs_cuda);
}