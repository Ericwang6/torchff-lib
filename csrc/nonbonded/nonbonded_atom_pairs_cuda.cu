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
#include "common/reduce.cuh"


template <typename scalar_t, int BLOCK_SIZE, bool DO_SHIFT, bool SIGN>
__global__ void nonbonded_atom_pairs_cuda_kernel(
    scalar_t* coords,
    int64_t* pairs,
    scalar_t* g_box,
    scalar_t* sigma, 
    scalar_t* epsilon,
    scalar_t* charges,
    scalar_t coul_constant,
    scalar_t cutoff,
    int64_t npairs,
    scalar_t* ene_out,
    scalar_t* coord_grad,
    scalar_t* sigma_grad,
    scalar_t* epsilon_grad,
    scalar_t* charges_grad
) {

    // Box
    __shared__ scalar_t box[9];
    __shared__ scalar_t box_inv[9];
    if (threadIdx.x < 9) {
        box[threadIdx.x] = g_box[threadIdx.x];
    }
    __syncthreads();

    // Invert box inside the kernel using first thread
    if (threadIdx.x == 0) {
        invert_box_3x3(box, box_inv);
    }
    __syncthreads();

    scalar_t cutoff2 = cutoff * cutoff;

    scalar_t ene = static_cast<scalar_t>(0.0);

    for (int64_t index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
         index < npairs;
         index += BLOCK_SIZE * gridDim.x) {
        int64_t i = pairs[index * 2];
        int64_t j = pairs[index * 2 + 1];
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

        scalar_t ecoul;
        if constexpr (DO_SHIFT) {
            ecoul = ci * cj * (rinv - static_cast<scalar_t>(1.0) / cutoff) * coul_constant;
        } else {
            ecoul = ci * cj * rinv * coul_constant;
        }
        scalar_t eij = sqrt_(ei * ej);
        scalar_t sij = (si + sj) / 2;
        scalar_t sij_r6 = pow_(sij * rinv, scalar_t(6.0));
        scalar_t elj = 4 * eij * sij_r6 * (sij_r6 - 1);
        scalar_t ene_pair = elj + ecoul;
        ene += ene_pair;

        if ( coord_grad ) {
            scalar_t base_f = ci * cj * rinv * rinv2 * coul_constant
                              + 24 * eij * sij_r6 * rinv2 * (2 * sij_r6 - 1);
            scalar_t f;
            if constexpr (SIGN) {
                // energy-mode: f = -dE/dr
                f = -base_f;
            } else {
                // force-mode: f = +dE/dr
                f = base_f;
            }
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
            scalar_t tmp;
            if constexpr (DO_SHIFT) {
                tmp = (rinv - static_cast<scalar_t>(1.0) / cutoff) * coul_constant;
            } else {
                tmp = rinv * coul_constant;
            }
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
    }

    if (ene_out) {
        block_reduce_sum<scalar_t, BLOCK_SIZE>(ene, ene_out);
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
        at::Scalar coul_constant,
        at::Scalar cutoff,
        bool do_shift
    )
    {
        int64_t npairs = pairs.size(0);

        at::Tensor ene = at::zeros({1}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor sigma_grad = at::zeros_like(sigma, sigma.options());
        at::Tensor epsilon_grad = at::zeros_like(epsilon, epsilon.options());
        at::Tensor charges_grad = at::zeros_like(charges, charges.options());

        auto props = at::cuda::getCurrentDeviceProperties();
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int BLOCK_SIZE = 256;
        int grid_dim = std::min(
            static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
            props->maxBlocksPerMultiProcessor * props->multiProcessorCount
        );

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_cuda", ([&] {
            const scalar_t cutoff_val = static_cast<scalar_t>(cutoff.toDouble());
            const scalar_t coul_constant_val = static_cast<scalar_t>(coul_constant.toDouble());

            if (do_shift) {
                nonbonded_atom_pairs_cuda_kernel<scalar_t, BLOCK_SIZE, true, true><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    sigma.data_ptr<scalar_t>(),
                    epsilon.data_ptr<scalar_t>(),
                    charges.data_ptr<scalar_t>(),
                    coul_constant_val,
                    cutoff_val,
                    npairs,
                    ene.data_ptr<scalar_t>(),
                    coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                    sigma.requires_grad() ? sigma_grad.data_ptr<scalar_t>() : nullptr,
                    epsilon.requires_grad() ? epsilon_grad.data_ptr<scalar_t>() : nullptr,
                    charges.requires_grad() ? charges_grad.data_ptr<scalar_t>() : nullptr
                );
            } else {
                nonbonded_atom_pairs_cuda_kernel<scalar_t, BLOCK_SIZE, false, true><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
                    coords.data_ptr<scalar_t>(),
                    pairs.data_ptr<int64_t>(),
                    box.data_ptr<scalar_t>(),
                    sigma.data_ptr<scalar_t>(),
                    epsilon.data_ptr<scalar_t>(),
                    charges.data_ptr<scalar_t>(),
                    coul_constant_val,
                    cutoff_val,
                    npairs,
                    ene.data_ptr<scalar_t>(),
                    coords.requires_grad() ? coord_grad.data_ptr<scalar_t>() : nullptr,
                    sigma.requires_grad() ? sigma_grad.data_ptr<scalar_t>() : nullptr,
                    epsilon.requires_grad() ? epsilon_grad.data_ptr<scalar_t>() : nullptr,
                    charges.requires_grad() ? charges_grad.data_ptr<scalar_t>() : nullptr
                );
            }
        }));

        ctx->save_for_backward({coord_grad, sigma_grad, epsilon_grad, charges_grad});
        return ene;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_outputs
    )
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor ignore;
        return {
            saved[0] * grad_outputs[0], // coords grad
            ignore,
            ignore,  // box grad (TODO: add this)
            saved[1] * grad_outputs[0], // sigma grad
            saved[2] * grad_outputs[0], // epsilon grad
            saved[3] * grad_outputs[0], // charges grad
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
    at::Scalar coul_constant,
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
    at::Scalar coul_constant,
    at::Scalar cutoff,
    at::Tensor forces
)
{
    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t npairs = pairs.size(0);
    constexpr int BLOCK_SIZE = 256;
    int grid_dim = std::min(
        static_cast<int>((npairs + BLOCK_SIZE - 1) / BLOCK_SIZE),
        props->maxBlocksPerMultiProcessor * props->multiProcessorCount
    );

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_cuda", ([&] {
        const scalar_t cutoff_val = static_cast<scalar_t>(cutoff.toDouble());
        const scalar_t coul_constant_val = static_cast<scalar_t>(coul_constant.toDouble());

        nonbonded_atom_pairs_cuda_kernel<scalar_t, BLOCK_SIZE, true, false><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int64_t>(),
            box.data_ptr<scalar_t>(),
            sigma.data_ptr<scalar_t>(),
            epsilon.data_ptr<scalar_t>(),
            charges.data_ptr<scalar_t>(),
            coul_constant_val,
            cutoff_val,
            npairs,
            nullptr,
            forces.data_ptr<scalar_t>(),
            nullptr,
            nullptr,
            nullptr
        );
    }));
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_nonbonded_energy_from_atom_pairs", compute_nonbonded_energy_from_atom_pairs_cuda);
    m.impl("compute_nonbonded_forces_from_atom_pairs", compute_nonbonded_forces_from_atom_pairs_cuda);
}