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
#include "damps.cuh"
#include "ewald/damps.cuh"



template <typename scalar_t>
__device__ __forceinline__ void pairwise_electric_data_multipole_kernel_rank_1(
    scalar_t c0_i,
    scalar_t dx_i, scalar_t dy_i, scalar_t dz_i,
    scalar_t c0_j,
    scalar_t dx_j, scalar_t dy_j, scalar_t dz_j,
    scalar_t drx, scalar_t dry, scalar_t drz,
    scalar_t damp1, scalar_t damp3, scalar_t damp5,
    scalar_t* c0_i_g,
    scalar_t* dx_i_g, scalar_t* dy_i_g, scalar_t* dz_i_g,
    scalar_t* c0_j_g,
    scalar_t* dx_j_g, scalar_t* dy_j_g, scalar_t* dz_j_g
) 
{
    // dr = rj - ri;
    scalar_t drinv = rsqrt_(drx*drx+dry*dry+drz*drz);
    scalar_t drinv2 = drinv * drinv;
    scalar_t drinv3 = drinv2 * drinv;
    scalar_t drinv5 = drinv3 * drinv2;

    drinv *= damp1;
    drinv3 *= damp3;
    drinv5 *= damp5;

    scalar_t tx = -drx * drinv3; 
    scalar_t ty = -dry * drinv3;
    scalar_t tz = -drz * drinv3;
    
    scalar_t txx = 3 * drx * drx * drinv5 - drinv3;
    scalar_t txy = 3 * drx * dry * drinv5;
    scalar_t txz = 3 * drx * drz * drinv5;
    scalar_t tyy = 3 * dry * dry * drinv5 - drinv3;
    scalar_t tyz = 3 * dry * drz * drinv5;
    scalar_t tzz = 3 * drz * drz * drinv5 - drinv3;     
    
    // charge gradient - electric potential
    *c0_i_g = drinv * c0_j + tx * dx_j + ty * dy_j + tz * dz_j;
    *c0_j_g = drinv * c0_i - tx * dx_i - ty * dy_i - tz * dz_i;
    
    // dipole gradient - electric field
    *dx_i_g = -c0_j * tx - txx * dx_j - txy * dy_j - txz * dz_j;
    *dy_i_g = -c0_j * ty - txy * dx_j - tyy * dy_j - tyz * dz_j;
    *dz_i_g = -c0_j * tz - txz * dx_j - tyz * dy_j - tzz * dz_j;

    *dx_j_g = c0_i * tx - txx * dx_i - txy * dy_i - txz * dz_i;
    *dy_j_g = c0_i * ty - txy * dx_i - tyy * dy_i - tyz * dz_i;
    *dz_j_g = c0_i * tz - txz * dx_i - tyz * dy_i - tzz * dz_i;
}


template <typename scalar_t>
__device__ __forceinline__ void pairwise_multipole_kernel_with_grad_rank_1(
    scalar_t c0_i,
    scalar_t dx_i, scalar_t dy_i, scalar_t dz_i,
    scalar_t c0_j,
    scalar_t dx_j, scalar_t dy_j, scalar_t dz_j,
    scalar_t drx, scalar_t dry, scalar_t drz,
    scalar_t damp1, scalar_t damp3, scalar_t damp5, scalar_t damp7,
    scalar_t* ene,
    scalar_t* drx_g, scalar_t* dry_g, scalar_t* drz_g
) 
{
    // dr = rj - ri;
    scalar_t drinv = rsqrt_(drx*drx+dry*dry+drz*drz);
    scalar_t drinv2 = drinv * drinv;
    scalar_t drinv3 = drinv2 * drinv;
    scalar_t drinv5 = drinv3 * drinv2;
    scalar_t drinv7 = drinv5 * drinv2;

    drinv *= damp1;
    drinv3 *= damp3;
    drinv5 *= damp5;
    drinv7 *= damp7;

    scalar_t tx = -drx * drinv3; 
    scalar_t ty = -dry * drinv3;
    scalar_t tz = -drz * drinv3;

    scalar_t x2 = drx * drx;
    scalar_t xy = drx * dry;
    scalar_t xz = drx * drz;
    scalar_t y2 = dry * dry;
    scalar_t yz = dry * drz;
    scalar_t z2 = drz * drz;
    scalar_t xyz = drx * dry * drz;
    
    scalar_t txx = 3 * x2 * drinv5 - drinv3;
    scalar_t txy = 3 * xy * drinv5;
    scalar_t txz = 3 * xz * drinv5;
    scalar_t tyy = 3 * y2 * drinv5 - drinv3;
    scalar_t tyz = 3 * yz * drinv5;
    scalar_t tzz = 3 * z2 * drinv5 - drinv3;     

    scalar_t txxx = -15 * x2 * drx * drinv7 + 9 * drx * drinv5;
    scalar_t txxy = -15 * x2 * dry * drinv7 + 3 * dry * drinv5;
    scalar_t txxz = -15 * x2 * drz * drinv7 + 3 * drz * drinv5;
    scalar_t tyyy = -15 * y2 * dry * drinv7 + 9 * dry * drinv5;
    scalar_t tyyx = -15 * y2 * drx * drinv7 + 3 * drx * drinv5;
    scalar_t tyyz = -15 * y2 * drz * drinv7 + 3 * drz * drinv5;
    scalar_t tzzz = -15 * z2 * drz * drinv7 + 9 * drz * drinv5;
    scalar_t tzzx = -15 * z2 * drx * drinv7 + 3 * drx * drinv5;
    scalar_t tzzy = -15 * z2 * dry * drinv7 + 3 * dry * drinv5;
    scalar_t txyz = -15 * xyz * drinv7;

    // interaction tensor graident
    scalar_t c0prod = c0_i * c0_j;

    scalar_t tx_g = c0_i * dx_j - c0_j * dx_i; 
    scalar_t ty_g = c0_i * dy_j - c0_j * dy_i;
    scalar_t tz_g = c0_i * dz_j - c0_j * dz_i;
    
    scalar_t txx_g = - dx_i * dx_j;
    scalar_t txy_g = - dx_i * dy_j - dx_j * dy_i;
    scalar_t txz_g = - dx_i * dz_j - dx_j * dz_i;
    scalar_t tyy_g = - dy_i * dy_j;
    scalar_t tyz_g = - dy_i * dz_j - dy_j * dz_i;
    scalar_t tzz_g = - dz_i * dz_j;     

    *ene = c0prod*drinv + tx * tx_g + ty * ty_g + tz * tz_g 
        + txx * txx_g + txy * txy_g + txz * txz_g + tyy * tyy_g + tyz * tyz_g + tzz * tzz_g;

    *drx_g = c0prod * tx + tx_g * txx + ty_g * txy + tz_g * txz
            + txxx * txx_g + txxy * txy_g + txxz * txz_g + tyyx * tyy_g + txyz * tyz_g + tzzx * tzz_g; 

    *dry_g = c0prod * ty + tx_g * txy + ty_g * tyy + tz_g * tyz
            + txxy * txx_g + tyyx * txy_g + txyz * txz_g + tyyy * tyy_g + tyyz * tyz_g + tzzy * tzz_g; 
    
    *drz_g = c0prod * tz + tx_g * txz + ty_g * tyz + tz_g * tzz
            + txxz * txx_g + txyz * txy_g + tzzx * txz_g + tyyz * tyy_g + tzzy * tyz_g + tzzz * tzz_g;
}


template <typename scalar_t>
__global__ void cmm_polarization_energy_from_induced_multipoles_kernel(
    scalar_t* dist_vecs,
    int32_t* pairs,
    scalar_t* dist_vecs_excl,
    int32_t* pairs_excl,
    scalar_t* charges,
    scalar_t* dipoles,
    scalar_t* b_elec_ij,
    int32_t npairs,
    int32_t npairs_excl,
    scalar_t ewald_alpha,
    scalar_t rcut_sr,
    scalar_t rcut_lr,
    scalar_t* ene,
    scalar_t* ene_excl,
    scalar_t* dist_vecs_grad,
    scalar_t* dist_vecs_excl_grad
)
{
    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < npairs; index += gridDim.x * blockDim.x) {
        int32_t i = pairs[index * 2];
        int32_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        scalar_t rij[3];
        rij[0] = dist_vecs[index*3];
        rij[1] = dist_vecs[index*3+1];
        rij[2] = dist_vecs[index*3+2];
        scalar_t dr = norm3d_(rij[0], rij[1], rij[2]);
        if ( dr >= rcut_lr ) { continue; }

        scalar_t c0_i = charges[i]; 
        scalar_t dx_i = dipoles[i*3]; scalar_t dy_i = dipoles[i*3+1]; scalar_t dz_i = dipoles[i*3+2];
        scalar_t c0_j = charges[j];
        scalar_t dx_j = dipoles[j*3]; scalar_t dy_j = dipoles[j*3+1]; scalar_t dz_j = dipoles[j*3+2];

        scalar_t damps[4];
        
        ewald_erfc_damps<scalar_t, 7>(dr, ewald_alpha, damps);
        if ( dr < rcut_sr ) {
            scalar_t tmp[4];
            polarization_damps<scalar_t, 7>(dr, b_elec_ij[index], tmp);
            damps[0] -= tmp[0]; damps[1] -= tmp[1]; damps[2] -= tmp[2]; damps[3] -= tmp[3];
        }
        
        scalar_t e;
        scalar_t drx_g, dry_g, drz_g;
        pairwise_multipole_kernel_with_grad_rank_1(
            c0_i, dx_i, dy_i, dz_i, c0_j, dx_j, dy_j, dz_j, rij[0], rij[1], rij[2],
            damps[0], damps[1], damps[2], damps[3], &e, &drx_g, &dry_g, &drz_g
        );
        ene[index] = e;

        atomicAdd(&dist_vecs_grad[index*3], drx_g);
        atomicAdd(&dist_vecs_grad[index*3+1], dry_g);
        atomicAdd(&dist_vecs_grad[index*3+2], drz_g);
    }

    for (int32_t index = start; index < npairs_excl; index += gridDim.x * blockDim.x) {
        int32_t i = pairs_excl[index * 2];
        int32_t j = pairs_excl[index * 2 + 1];

        scalar_t rij[3];
        rij[0] = dist_vecs_excl[index*3];
        rij[1] = dist_vecs_excl[index*3+1];
        rij[2] = dist_vecs_excl[index*3+2];
        scalar_t dr = norm3d_(rij[0], rij[1], rij[2]);

        scalar_t c0_i = charges[i];
        scalar_t c0_j = charges[j];

        scalar_t dx_i = dipoles[i*3]; scalar_t dy_i = dipoles[i*3+1]; scalar_t dz_i = dipoles[i*3+2];
        scalar_t dx_j = dipoles[j*3]; scalar_t dy_j = dipoles[j*3+1]; scalar_t dz_j = dipoles[j*3+2];

        scalar_t damps[4];
        ewald_erfc_damps<scalar_t, 7>(dr, ewald_alpha, damps);

        scalar_t e;
        scalar_t drx_g, dry_g, drz_g;
        pairwise_multipole_kernel_with_grad_rank_1(
            c0_i, dx_i, dy_i, dz_i, c0_j, dx_j, dy_j, dz_j, rij[0], rij[1], rij[2],
            damps[0]-scalar_t(1.0), damps[1]-scalar_t(1.0), damps[2]-scalar_t(1.0), damps[3]-scalar_t(1.0), 
            &e, &drx_g, &dry_g, &drz_g
        );
        ene_excl[index] = e;

        atomicAdd(&dist_vecs_excl_grad[index*3], drx_g);
        atomicAdd(&dist_vecs_excl_grad[index*3+1], dry_g);
        atomicAdd(&dist_vecs_excl_grad[index*3+2], drz_g);
    }
}


template <typename scalar_t>
__global__ void cmm_polarization_real_kernel(
    scalar_t* coords,
    scalar_t* g_box,
    scalar_t* g_box_inv,
    int32_t* pairs,
    scalar_t* b_elec_ij,
    int32_t* pairs_excl,
    scalar_t* charges,
    scalar_t* dipoles,
    scalar_t* epot,
    scalar_t* efield,
    int32_t npairs,
    int32_t npairs_excl,
    scalar_t ewald_alpha,
    scalar_t rcut_sr,
    scalar_t rcut_lr
)
{
    // Box
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }
    __syncthreads();
    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < npairs; index += gridDim.x * blockDim.x) {
        int32_t i = pairs[index * 2];
        int32_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        scalar_t rij[3];
        diff_vec3(coords+j*3, coords+i*3, rij);
        apply_pbc_triclinic(rij, box, box_inv, rij);
        scalar_t dr = norm3d_(rij[0], rij[1], rij[2]);
        if ( dr >= rcut_lr ) { continue; }

        scalar_t c0_i = charges[i]; 
        scalar_t dx_i = dipoles[i*3]; scalar_t dy_i = dipoles[i*3+1]; scalar_t dz_i = dipoles[i*3+2];
        scalar_t c0_j = charges[j];
        scalar_t dx_j = dipoles[j*3]; scalar_t dy_j = dipoles[j*3+1]; scalar_t dz_j = dipoles[j*3+2];

        scalar_t damps[3];
        
        ewald_erfc_damps<scalar_t, 5>(dr, ewald_alpha, damps);
        if ( dr < rcut_sr ) {
            scalar_t tmp[3];
            polarization_damps<scalar_t, 5>(dr, b_elec_ij[index], tmp);
            damps[0] -= tmp[0]; damps[1] -= tmp[1]; damps[2] -= tmp[2];
        }
        
        scalar_t edata_i[4]; scalar_t edata_j[4];
        pairwise_electric_data_multipole_kernel_rank_1(
            c0_i, dx_i, dy_i, dz_i, c0_j, dx_j, dy_j, dz_j, rij[0], rij[1], rij[2],
            damps[0], damps[1], damps[2],
            edata_i, edata_i+1, edata_i+2, edata_i+3,
            edata_j, edata_j+1, edata_j+2, edata_j+3
        );

        atomicAdd(&epot[i], edata_i[0]);
        atomicAdd(&epot[j], edata_j[0]);
        atomicAdd(&efield[i*3], edata_i[1]); atomicAdd(&efield[i*3+1], edata_i[2]); atomicAdd(&efield[i*3+2], edata_i[3]);
        atomicAdd(&efield[j*3], edata_j[1]); atomicAdd(&efield[j*3+1], edata_j[2]); atomicAdd(&efield[j*3+2], edata_j[3]);
    }

    for (int32_t index = start; index < npairs_excl; index += gridDim.x * blockDim.x) {
        int32_t i = pairs_excl[index * 2];
        int32_t j = pairs_excl[index * 2 + 1];

        scalar_t rij[3];
        diff_vec3(coords+j*3, coords+i*3, rij);
        apply_pbc_triclinic(rij, box, box_inv, rij);
        scalar_t dr = norm3d_(rij[0], rij[1], rij[2]);

        scalar_t c0_i = charges[i];
        scalar_t c0_j = charges[j];

        scalar_t dx_i = dipoles[i*3]; scalar_t dy_i = dipoles[i*3+1]; scalar_t dz_i = dipoles[i*3+2];
        scalar_t dx_j = dipoles[j*3]; scalar_t dy_j = dipoles[j*3+1]; scalar_t dz_j = dipoles[j*3+2];

        scalar_t epot_i = scalar_t(0.0); scalar_t epot_j = scalar_t(0.0);
        scalar_t efield_i[3] = {}; scalar_t efield_j[3] = {};

        scalar_t damps[3];
        ewald_erfc_damps<scalar_t, 5>(dr, ewald_alpha, damps);

        scalar_t edata_i[4]; scalar_t edata_j[4];
        pairwise_electric_data_multipole_kernel_rank_1(
            c0_i, dx_i, dy_i, dz_i, c0_j, dx_j, dy_j, dz_j, rij[0], rij[1], rij[2],
            damps[0]-scalar_t(1.0), damps[1]-scalar_t(1.0), damps[2]-scalar_t(1.0),
            edata_i, edata_i+1, edata_i+2, edata_i+3,
            edata_j, edata_j+1, edata_j+2, edata_j+3
        );

        atomicAdd(&epot[i], edata_i[0]);
        atomicAdd(&epot[j], edata_j[0]);
        atomicAdd(&efield[i*3], edata_i[1]); atomicAdd(&efield[i*3+1], edata_i[2]); atomicAdd(&efield[i*3+2], edata_i[3]);
        atomicAdd(&efield[j*3], edata_j[1]); atomicAdd(&efield[j*3+1], edata_j[2]); atomicAdd(&efield[j*3+2], edata_j[3]);
    }
}


class CMMPolarizationEnergyFromInducedMultipolesFromPairsFunctionCuda: public torch::autograd::Function<CMMPolarizationEnergyFromInducedMultipolesFromPairsFunctionCuda> {

public: 

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& dist_vecs, at::Tensor& pairs,
    at::Tensor& dist_vecs_excl, at::Tensor& pairs_excl,
    at::Tensor& induced_multipoles,
    at::Tensor& b_elec_ij,
    at::Scalar ewald_alpha,
    at::Scalar rcut_sr,
    at::Scalar rcut_lr,
    at::Scalar natoms
)
{
    int32_t npairs = pairs.size(0);
    int32_t npairs_excl = pairs_excl.size(0);
    at::Tensor ene = at::zeros({npairs}, dist_vecs.options());
    at::Tensor ene_excl = at::zeros({npairs_excl}, dist_vecs_excl.options());

    at::Tensor dist_vecs_grad = at::zeros_like(dist_vecs, dist_vecs.options());
    at::Tensor dist_vecs_excl_grad = at::zeros_like(dist_vecs_excl, dist_vecs_excl.options());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = 128;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (npairs+block_dim-1)/block_dim);

    AT_DISPATCH_FLOATING_TYPES(dist_vecs.scalar_type(), "cmm_polarization_energy_from_induced_multipoles_kernel", ([&] {
        scalar_t* charges_ptr = induced_multipoles.data_ptr<scalar_t>();
        scalar_t* dipoles_ptr = charges_ptr + natoms.toInt();
        cmm_polarization_energy_from_induced_multipoles_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            dist_vecs.data_ptr<scalar_t>(),
            pairs.data_ptr<int32_t>(),
            dist_vecs_excl.data_ptr<scalar_t>(),
            pairs_excl.data_ptr<int32_t>(),
            charges_ptr,
            dipoles_ptr,
            b_elec_ij.data_ptr<scalar_t>(),
            npairs, npairs_excl,
            static_cast<scalar_t>(ewald_alpha.toDouble()),
            static_cast<scalar_t>(rcut_sr.toDouble()),
            static_cast<scalar_t>(rcut_lr.toDouble()),
            ene.data_ptr<scalar_t>(),
            ene_excl.data_ptr<scalar_t>(),
            dist_vecs_grad.data_ptr<scalar_t>(),
            dist_vecs_excl_grad.data_ptr<scalar_t>()
        );
    }));
    
    ctx->save_for_backward({dist_vecs_grad, dist_vecs_excl_grad});

    return at::sum(ene)+at::sum(ene_excl);
}

static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<at::Tensor> grad_outputs
)
{
    auto saved = ctx->get_saved_variables();

    at::Tensor ignore;
    return {
        saved[0]*grad_outputs[0], ignore,
        saved[1]*grad_outputs[0], ignore,
        ignore, ignore, ignore, 
        ignore, ignore, ignore
    };
}

};


at::Tensor cmm_polarization_energy_from_induced_multipoles_cuda(
    at::Tensor& dist_vecs, at::Tensor& pairs,
    at::Tensor& dist_vecs_excl, at::Tensor& pairs_excl,
    at::Tensor& induced_multipoles,
    at::Tensor& b_elec_ij,
    at::Scalar ewald_alpha,
    at::Scalar rcut_sr,
    at::Scalar rcut_lr,
    at::Scalar natoms
)
{
    return CMMPolarizationEnergyFromInducedMultipolesFromPairsFunctionCuda::apply(
        dist_vecs, pairs, dist_vecs_excl, pairs_excl, induced_multipoles, b_elec_ij,
        ewald_alpha, rcut_sr, rcut_lr, natoms
    );
}


void compute_cmm_polarization_real_space_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& pairs,
    at::Tensor& pairs_excl,
    at::Tensor& b_elec_ij,
    at::Tensor& vec_in,
    at::Scalar ewald_alpha,
    at::Scalar rcut_sr,
    at::Scalar rcut_lr,
    at::Tensor& vec_out
)
{
    at::Tensor box_inv, ignore;
    std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t npairs = pairs.size(0);
    int32_t natoms = coords.size(0);
    int32_t block_dim = 256;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (npairs+block_dim-1)/block_dim);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_polarization_real_kernel", ([&] {
        scalar_t* charges_ptr = vec_in.data_ptr<scalar_t>();
        scalar_t* dipoles_ptr = charges_ptr + natoms;
        scalar_t* epot_ptr = vec_out.data_ptr<scalar_t>();
        scalar_t* efield_ptr = epot_ptr + natoms;
        cmm_polarization_real_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            pairs.data_ptr<int32_t>(),
            b_elec_ij.data_ptr<scalar_t>(),
            pairs_excl.data_ptr<int32_t>(),
            charges_ptr,
            dipoles_ptr,
            epot_ptr,
            efield_ptr,
            npairs, 
            static_cast<int32_t>(pairs_excl.size(0)),
            static_cast<scalar_t>(ewald_alpha.toDouble()),
            static_cast<scalar_t>(rcut_sr.toDouble()),
            static_cast<scalar_t>(rcut_lr.toDouble())
        );
    }));
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_cmm_polarization_real_space", compute_cmm_polarization_real_space_cuda);
    m.impl("cmm_polarization_energy_from_induced_multipoles", cmm_polarization_energy_from_induced_multipoles_cuda);
}