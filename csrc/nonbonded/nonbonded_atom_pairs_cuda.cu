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



template <typename scalar_t>
__global__ void nonbonded_atom_pairs_kernel(
    scalar_t* coords,  
    int32_t* pairs,
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t* sigma, 
    scalar_t* epsilon,
    scalar_t* charges,
    scalar_t* coul_constant_,
    scalar_t cutoff,
    bool do_shift,
    int32_t npairs,
    scalar_t* ene,
    scalar_t* coord_grad,
    scalar_t* sigma_grad,
    scalar_t* epsilon_grad,
    scalar_t* charges_grad,
    scalar_t sign
) {

    // Box
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
        box_inv[threadIdx.x] = g_box_inv[threadIdx.x];
    }
    __syncthreads();

    // Coulomb constant
    scalar_t coul_constant = coul_constant_[0];
    scalar_t cutoff2 = cutoff * cutoff;

    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < npairs; index += gridDim.x * blockDim.x) {
        int32_t i = pairs[index * 2];
        int32_t j = pairs[index * 2 + 1];
        if ( i < 0 || j < 0 ) {
            continue;
        }
        scalar_t crd_i[3];
        crd_i[0] = coords[i*3]; crd_i[1] = coords[i*3+1]; crd_i[2] = coords[i*3+2];
        scalar_t crd_j[3];
        crd_j[0] = coords[j*3]; crd_j[1] = coords[j*3+1]; crd_j[2] = coords[j*3+2];
        scalar_t rij[3];
        diff_vec3(crd_i, crd_j, rij);
        apply_pbc_triclinic(rij, box, box_inv, rij);
        scalar_t r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        if ( r2 > cutoff2 ) {
            continue;
        }
        scalar_t ci = charges[i];
        scalar_t cj = charges[j];
        scalar_t ei = epsilon[i];
        scalar_t ej = epsilon[j];
        scalar_t si = sigma[i];
        scalar_t sj = sigma[j];
        scalar_t rinv = rsqrt_(r2);
        scalar_t rinv2 = rinv * rinv;

        scalar_t ecoul = ci * cj * (rinv - do_shift / cutoff) * coul_constant;
        scalar_t eij = sqrt_(ei * ej);
        scalar_t sij = (si + sj) / 2;
        scalar_t sij_r6 = pow_(sij * rinv, scalar_t(6.0));
        scalar_t elj = 4 * eij * sij_r6 * (sij_r6 - 1);
        
        if ( coord_grad ) {
            scalar_t f = -sign * (ci * cj * rinv * rinv2 * coul_constant + 24 * eij * sij_r6 * rinv2 * (2 * sij_r6 - 1));
            scalar_t fx = f * rij[0];
            scalar_t fy = f * rij[1];
            scalar_t fz = f * rij[2];
            atomicAdd(&coord_grad[i*3],   fx);
            atomicAdd(&coord_grad[i*3+1], fy);
            atomicAdd(&coord_grad[i*3+2], fz);
            atomicAdd(&coord_grad[j*3],  -fx);
            atomicAdd(&coord_grad[j*3+1],-fy);
            atomicAdd(&coord_grad[j*3+2],-fz);
        }

        if ( charges_grad ) {
            scalar_t tmp = (rinv - do_shift / cutoff) * coul_constant;
            atomicAdd(&charges_grad[i], cj * tmp);
            atomicAdd(&charges_grad[j], ci * tmp);
        }
        if ( epsilon_grad ) {
            scalar_t tmp = 2 * sij_r6 * ( sij_r6 - 1 ) / eij;
            atomicAdd(&epsilon_grad[i], tmp * ej);
            atomicAdd(&epsilon_grad[j], tmp * ei);
        }
        if ( sigma_grad ) {
            scalar_t sg = 12 * eij / sij * sij_r6 * (2*sij_r6-1);
            atomicAdd(&sigma_grad[i], sg);
            atomicAdd(&sigma_grad[j], sg);
        }
        if ( ene ) {
            ene[index] = elj + ecoul;
        }
    }
}


class NonbondedFromAtomPairsFunctionCuda: public torch::autograd::Function<NonbondedFromAtomPairsFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& pairs,
        at::Tensor& box,
        at::Tensor& sigma,
        at::Tensor& epsilon,
        at::Tensor& charges,
        at::Tensor& coul_constant,
        at::Scalar cutoff,
        bool do_shift
    )
    {
        // at::linalg_inv does not support CUDA graph
        at::Tensor box_inv, ignore;
        std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);
        
        int32_t npairs = pairs.size(0);
        at::Tensor ene = at::zeros({npairs}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor sigma_grad = at::zeros_like(sigma, sigma.options());
        at::Tensor epsilon_grad = at::zeros_like(epsilon, epsilon.options());
        at::Tensor charges_grad = at::zeros_like(charges, charges.options());

        auto stream = at::cuda::getCurrentCUDAStream();
        int32_t block_dim = 512;
        int32_t grid_dim = (npairs + block_dim - 1) / block_dim;
        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_cuda", ([&] {
            nonbonded_atom_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                pairs.data_ptr<int32_t>(),
                box.data_ptr<scalar_t>(),
                box_inv.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                epsilon.data_ptr<scalar_t>(),
                charges.data_ptr<scalar_t>(),
                coul_constant.data_ptr<scalar_t>(),
                static_cast<scalar_t>(cutoff.toDouble()),
                do_shift,
                npairs,
                ene.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>(): nullptr,
                (sigma.requires_grad()) ? sigma_grad.data_ptr<scalar_t>(): nullptr,
                (epsilon.requires_grad()) ? epsilon_grad.data_ptr<scalar_t>() : nullptr,
                (charges.requires_grad()) ? charges_grad.data_ptr<scalar_t>(): nullptr,
                static_cast<scalar_t>(1.0)
            );
        }));
        ctx->save_for_backward({ene, coord_grad, sigma_grad, epsilon_grad, charges_grad, coul_constant});
        return at::sum(ene);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    )
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[1] * grad_outputs[0], // coords grad
            ignore,
            ignore,  // box grad (TODO: add this)
            saved[2] * grad_outputs[0], // sigma grad
            saved[3] * grad_outputs[0], // epsilon grad
            saved[4] * grad_outputs[0], // charges grad
            ignore, // coul constant grad
            ignore, ignore// do shift & cutoff
        };
    }

};


at::Tensor compute_nonbonded_energy_from_atom_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Tensor& charges,
    at::Tensor& coul_constant,
    at::Scalar cutoff,
    bool do_shift
)
{
    return NonbondedFromAtomPairsFunctionCuda::apply(
        coords, pairs, box,
        sigma, epsilon, charges,
        coul_constant,
        cutoff, do_shift
    );
}


void compute_nonbonded_forces_from_atom_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Tensor& charges,
    at::Tensor& coul_constant,
    at::Scalar cutoff,
    at::Tensor forces
)
{
    // at::linalg_inv does not support CUDA graph
    at::Tensor box_inv, ignore;
    std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t npairs = pairs.size(0);
    int32_t block_dim = 512;
    int32_t grid_dim = (npairs + block_dim - 1) / block_dim;
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_cuda", ([&] {
        nonbonded_atom_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int32_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            sigma.data_ptr<scalar_t>(),
            epsilon.data_ptr<scalar_t>(),
            charges.data_ptr<scalar_t>(),
            coul_constant.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff.toDouble()),
            true,
            npairs,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr,
            static_cast<scalar_t>(-1.0)
        );
    }));
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_nonbonded_energy_from_atom_pairs", compute_nonbonded_energy_from_atom_pairs_cuda);
    m.impl("compute_nonbonded_forces_from_atom_pairs", compute_nonbonded_forces_from_atom_pairs_cuda);
}