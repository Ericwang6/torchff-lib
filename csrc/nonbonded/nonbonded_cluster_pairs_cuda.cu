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
__global__ void nonbonded_cluster_pairs_kernel(
    scalar_t* coords,  
    scalar_t* g_box,
    scalar_t* g_box_inv,
    scalar_t* sigma, 
    scalar_t* epsilon,
    scalar_t* charges,
    scalar_t* coul_constant_,
    scalar_t cutoff,
    bool do_shift,
    int32_t* sorted_atom_indices,
    int32_t* cluster_exclusions,
    int32_t num_cluster_exclusions,
    uint32_t* bitmask_exclusions,
    int32_t* interacting_clusters,
    int32_t* interacting_atoms,
    int32_t num_interacting_clusters,
    int32_t natoms,
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

    int32_t totalWarps = blockDim.x * gridDim.x / warpSize;
    int32_t warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t idxInWarp = threadIdx.x & (warpSize - 1);

    scalar_t energy = 0.0;

    // First loop: clusters with exclusions, each warp process a cluster pair
    for (int32_t pos = warpIdx; pos < num_cluster_exclusions; pos += totalWarps) {
        int32_t x = cluster_exclusions[pos];
        int32_t y = cluster_exclusions[pos+num_cluster_exclusions];
        if ( x < 0 || y < 0 || x > y ) {
            continue;
        }
        int32_t index_i = x * warpSize + idxInWarp;
        int32_t index_j = y * warpSize + idxInWarp;
        int32_t i = -1;
        int32_t j = -1;

        // Coords
        scalar_t crd_i[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j[3] = {0.0, 0.0, 0.0};

        // Parameters
        scalar_t si = 0.0, ei = 0.0, ci = 0.0;
        scalar_t sj = 0.0, ej = 0.0, cj = 0.0;

        // Coordinate gradient
        scalar_t fi[3] = {0.0, 0.0, 0.0};
        scalar_t fj[3] = {0.0, 0.0, 0.0};

        // Parameters gradient
        scalar_t sgi = 0.0, egi = 0.0, cgi = 0.0;
        scalar_t sgj = 0.0, egj = 0.0, cgj = 0.0;

        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            crd_i[0] = coords[i*3];
            crd_i[1] = coords[i*3+1];
            crd_i[2] = coords[i*3+2];
            si = sigma[i];
            ei = epsilon[i];
            ci = charges[i];
        }
        if ( index_j < natoms ) {
            j = sorted_atom_indices[index_j];
            crd_j[0] = coords[j*3];
            crd_j[1] = coords[j*3+1];
            crd_j[2] = coords[j*3+2];
            sj = sigma[j];
            ej = epsilon[j];
            cj = charges[j];
        }

        // Prepare exclusions. "Rotate" the exlcusion where the lowest bit is i-i pairs
        uint32_t excl = bitmask_exclusions[pos * warpSize + idxInWarp];
        excl = (excl >> idxInWarp) | (excl << (warpSize-idxInWarp));

        for (int32_t srcLane = 0; srcLane < warpSize; ++srcLane) {
            if ( i >= 0 && j >= 0 && !(excl & 0x1) ) {
                scalar_t rij[3];
                diff_vec3(crd_i, crd_j, rij);
                apply_pbc_triclinic(rij, box, box_inv, rij);
                scalar_t r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if ( r2 < cutoff2 ) {
                    scalar_t rinv = rsqrt_(r2);
                    scalar_t rinv2 = rinv * rinv;

                    // Coulomb
                    scalar_t ecoul = ci * cj * (rinv - do_shift / cutoff) * coul_constant;
                    scalar_t eij = sqrt_(ei * ej);
                    scalar_t sij = (si + sj) / 2;
                    scalar_t sij_r6 = pow_(sij * rinv, scalar_t(6.0));
                    // LJ
                    scalar_t elj = 4 * eij * sij_r6 * (sij_r6 - 1);
                    
                    energy += ecoul + elj;

                    if ( coord_grad ) {
                        scalar_t f = -sign * (ci * cj * rinv * rinv2 * coul_constant + 24 * eij * sij_r6 * rinv2 * (2 * sij_r6 - 1));
                        scalar_t fx = f * rij[0];
                        scalar_t fy = f * rij[1];
                        scalar_t fz = f * rij[2];
                        fi[0] += fx; fi[1] += fy; fi[2] += fz;
                        fj[0] -= fx; fj[1] -= fy; fj[2] -= fz;
                    }
                    if ( charges_grad ) {
                        scalar_t tmp = (rinv - do_shift / cutoff) * coul_constant;
                        cgi += cj * tmp; cgj += ci * tmp;
                    }
                    if ( epsilon_grad ) {
                        scalar_t tmp = 2 * sij_r6 * ( sij_r6 - 1 ) / eij;
                        egi += tmp * ej; egj += tmp * ei;
                    }
                    if ( sigma_grad ) {
                        scalar_t sg = 12 * eij / sij * sij_r6 * (2*sij_r6-1);
                        sgi += sg; sgj += sg;
                    }
                }
            }

            j = __shfl_sync(0xFFFFFFFFu, j, idxInWarp+1);
            crd_j[0] = __shfl_sync(0xFFFFFFFFu, crd_j[0], idxInWarp+1);
            crd_j[1] = __shfl_sync(0xFFFFFFFFu, crd_j[1], idxInWarp+1);
            crd_j[2] = __shfl_sync(0xFFFFFFFFu, crd_j[2], idxInWarp+1);
            cj = __shfl_sync(0xFFFFFFFFu, cj, idxInWarp+1);
            sj = __shfl_sync(0xFFFFFFFFu, sj, idxInWarp+1);
            ej = __shfl_sync(0xFFFFFFFFu, ej, idxInWarp+1);

            if ( coord_grad ) {
                fj[0] = __shfl_sync(0xFFFFFFFFu, fj[0], idxInWarp+1);
                fj[1] = __shfl_sync(0xFFFFFFFFu, fj[1], idxInWarp+1);
                fj[2] = __shfl_sync(0xFFFFFFFFu, fj[2], idxInWarp+1);
            }
            if ( charges_grad ) {
                cgj = __shfl_sync(0xFFFFFFFFu, cgj, idxInWarp+1);
            }
            if ( epsilon_grad ) {
                egj = __shfl_sync(0xFFFFFFFFu, egj, idxInWarp+1);
            }
            if ( sigma_grad ) {
                sgj = __shfl_sync(0xFFFFFFFFu, sgj, idxInWarp+1);
            }
            excl >>= 1;
        }

        if ( coord_grad ) {
            if ( i >= 0 ) {
                atomicAdd(&coord_grad[i*3],   fi[0]); 
                atomicAdd(&coord_grad[i*3+1], fi[1]);
                atomicAdd(&coord_grad[i*3+2], fi[2]);
            }
            if ( j >= 0 ) {
                atomicAdd(&coord_grad[j*3],   fj[0]); 
                atomicAdd(&coord_grad[j*3+1], fj[1]);
                atomicAdd(&coord_grad[j*3+2], fj[2]);
            }
        }
        if ( charges_grad ) {
            if (i >= 0) { atomicAdd(&charges_grad[i], cgi); }
            if (j >= 0) { atomicAdd(&charges_grad[j], cgj); }
        }
        if ( epsilon_grad ) {
            if (i >= 0) { atomicAdd(&epsilon_grad[i], egi); }
            if (j >= 0) { atomicAdd(&epsilon_grad[j], egj); }
        }
        if ( sigma_grad ) {
            if (i >= 0) { atomicAdd(&sigma_grad[i], sgi); }
            if (j >= 0) { atomicAdd(&sigma_grad[j], sgj); }
        }
    }

    // Second loop: atoms without exclusions, read through a neighbor list
    for (int32_t pos = warpIdx; pos < num_interacting_clusters; pos += totalWarps) {
        int32_t x = interacting_clusters[pos];
        if ( x == -1 ) {
            continue;
        }
        int32_t index_i = x * warpSize + idxInWarp;

        int32_t i = -1;
        int32_t j = -1;

        // Coords
        scalar_t crd_i[3] = {0.0, 0.0, 0.0};
        scalar_t crd_j[3] = {0.0, 0.0, 0.0};

        // Parameters
        scalar_t si = 0.0, ei = 0.0, ci = 0.0;
        scalar_t sj = 0.0, ej = 0.0, cj = 0.0;

        // Coordinate gradient
        scalar_t fi[3] = {0.0, 0.0, 0.0};
        scalar_t fj[3] = {0.0, 0.0, 0.0};

        // Parameters gradient
        scalar_t sgi = 0.0, egi = 0.0, cgi = 0.0;
        scalar_t sgj = 0.0, egj = 0.0, cgj = 0.0;

        if ( index_i < natoms ) {
            i = sorted_atom_indices[index_i];
            crd_i[0] = coords[i*3];
            crd_i[1] = coords[i*3+1];
            crd_i[2] = coords[i*3+2];
            si = sigma[i];
            ei = epsilon[i];
            ci = charges[i];
        }

        j = interacting_atoms[pos*warpSize+idxInWarp];
        if ( j >= 0 ) {
            crd_j[0] = coords[j*3];
            crd_j[1] = coords[j*3+1];
            crd_j[2] = coords[j*3+2];
            sj = sigma[j];
            ej = epsilon[j];
            cj = charges[j];
        }

        for (int32_t srcLane = 0; srcLane < warpSize; ++srcLane) {

            if ( i >= 0 && j >= 0 ) {
                scalar_t rij[3];
                diff_vec3(crd_i, crd_j, rij);
                apply_pbc_triclinic(rij, box, box_inv, rij);
                scalar_t r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                if ( r2 < cutoff2 ) {
                    scalar_t rinv = rsqrt_(r2);
                    scalar_t rinv2 = rinv * rinv;

                    // Coulomb
                    scalar_t ecoul = ci * cj * (rinv - do_shift / cutoff) * coul_constant;
                    scalar_t eij = sqrt_(ei * ej);
                    scalar_t sij = (si + sj) / 2;
                    scalar_t sij_r6 = pow_(sij * rinv, scalar_t(6.0));
                    // LJ
                    scalar_t elj = 4 * eij * sij_r6 * (sij_r6 - 1);
                    
                    energy += ecoul + elj;

                    if ( coord_grad ) {
                        scalar_t f = -sign * (ci * cj * rinv * rinv2 * coul_constant + 24 * eij * sij_r6 * rinv2 * (2 * sij_r6 - 1));
                        scalar_t fx = f * rij[0];
                        scalar_t fy = f * rij[1];
                        scalar_t fz = f * rij[2];
                        fi[0] += fx; fi[1] += fy; fi[2] += fz;
                        fj[0] -= fx; fj[1] -= fy; fj[2] -= fz;
                    }
                    if ( charges_grad ) {
                        scalar_t tmp = (rinv - do_shift / cutoff) * coul_constant;
                        cgi += cj * tmp; cgj += ci * tmp;
                    }
                    if ( epsilon_grad ) {
                        scalar_t tmp = 2 * sij_r6 * ( sij_r6 - 1 ) / eij;
                        egi += tmp * ej; egj += tmp * ei;
                    }
                    if ( sigma_grad ) {
                        scalar_t sg = 12 * eij / sij * sij_r6 * (2*sij_r6-1);
                        sgi += sg; sgj += sg;
                    }
                }
            }

            j = __shfl_sync(0xFFFFFFFFu, j, idxInWarp+1);
            crd_j[0] = __shfl_sync(0xFFFFFFFFu, crd_j[0], idxInWarp+1);
            crd_j[1] = __shfl_sync(0xFFFFFFFFu, crd_j[1], idxInWarp+1);
            crd_j[2] = __shfl_sync(0xFFFFFFFFu, crd_j[2], idxInWarp+1);
            cj = __shfl_sync(0xFFFFFFFFu, cj, idxInWarp+1);
            sj = __shfl_sync(0xFFFFFFFFu, sj, idxInWarp+1);
            ej = __shfl_sync(0xFFFFFFFFu, ej, idxInWarp+1);

            if ( coord_grad ) {
                fj[0] = __shfl_sync(0xFFFFFFFFu, fj[0], idxInWarp+1);
                fj[1] = __shfl_sync(0xFFFFFFFFu, fj[1], idxInWarp+1);
                fj[2] = __shfl_sync(0xFFFFFFFFu, fj[2], idxInWarp+1);
            }
            if ( charges_grad ) {
                cgj = __shfl_sync(0xFFFFFFFFu, cgj, idxInWarp+1);
            }
            if ( epsilon_grad ) {
                egj = __shfl_sync(0xFFFFFFFFu, egj, idxInWarp+1);
            }
            if ( sigma_grad ) {
                sgj = __shfl_sync(0xFFFFFFFFu, sgj, idxInWarp+1);
            }
        }
        if ( coord_grad ) {
            if ( i >= 0 ) {
                atomicAdd(&coord_grad[i*3],   fi[0]); 
                atomicAdd(&coord_grad[i*3+1], fi[1]);
                atomicAdd(&coord_grad[i*3+2], fi[2]);
            }
            if ( j >= 0 ) {
                atomicAdd(&coord_grad[j*3],   fj[0]); 
                atomicAdd(&coord_grad[j*3+1], fj[1]);
                atomicAdd(&coord_grad[j*3+2], fj[2]);
            }
        }
        if ( charges_grad ) {
            if (i >= 0) { atomicAdd(&charges_grad[i], cgi); }
            if (j >= 0) { atomicAdd(&charges_grad[j], cgj); }
        }
        if ( epsilon_grad ) {
            if (i >= 0) { atomicAdd(&epsilon_grad[i], egi); }
            if (j >= 0) { atomicAdd(&epsilon_grad[j], egj); }
        }
        if ( sigma_grad ) {
            if (i >= 0) { atomicAdd(&sigma_grad[i], sgi); }
            if (j >= 0) { atomicAdd(&sigma_grad[j], sgj); }
        }
    }

    if ( ene ) {
        ene[blockIdx.x*blockDim.x+threadIdx.x] = energy;
    }

}


class NonbondedFromClusterPairsFunctionCuda: public torch::autograd::Function<NonbondedFromClusterPairsFunctionCuda> {

public: 
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor& coords,
        at::Tensor& box,
        at::Tensor& sigma,
        at::Tensor& epsilon,
        at::Tensor& charges,
        at::Tensor& coul_constant,
        at::Scalar cutoff,
        at::Tensor& sorted_atom_indices,
        at::Tensor& cluster_exclusions,
        at::Tensor& bitmask_exclusions,
        at::Tensor& interacting_clusters,
        at::Tensor& interacting_atoms,
        bool do_shift
    )
    {
        // at::linalg_inv does not support CUDA graph
        at::Tensor box_inv, ignore;
        std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);

        auto stream = at::cuda::getCurrentCUDAStream();
        auto props = at::cuda::getCurrentDeviceProperties();
        int32_t block_dim = 256;
        int32_t warpsPerBlock = block_dim / props->warpSize;
        int32_t grid_dim = std::min(
            props->multiProcessorCount*props->maxBlocksPerMultiProcessor,
            (int32_t)(std::max(cluster_exclusions.size(1), interacting_clusters.size(0)) + warpsPerBlock - 1) / warpsPerBlock
        );

        at::Tensor ene = at::zeros({block_dim*grid_dim}, coords.options());
        at::Tensor coord_grad = at::zeros_like(coords, coords.options());
        at::Tensor sigma_grad = at::zeros_like(sigma, sigma.options());
        at::Tensor epsilon_grad = at::zeros_like(epsilon, epsilon.options());
        at::Tensor charges_grad = at::zeros_like(charges, charges.options());

        AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_from_cluster_pairs", ([&] {
            nonbonded_cluster_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
                coords.data_ptr<scalar_t>(),
                box.data_ptr<scalar_t>(),
                box_inv.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                epsilon.data_ptr<scalar_t>(),
                charges.data_ptr<scalar_t>(),
                coul_constant.data_ptr<scalar_t>(),
                static_cast<scalar_t>(cutoff.toDouble()),
                do_shift,
                sorted_atom_indices.data_ptr<int32_t>(),
                cluster_exclusions.data_ptr<int32_t>(),
                (int32_t)cluster_exclusions.size(1),
                bitmask_exclusions.data_ptr<uint32_t>(),
                interacting_clusters.data_ptr<int32_t>(),
                interacting_atoms.data_ptr<int32_t>(),
                (int32_t)interacting_clusters.size(0),
                (int32_t)coords.size(0),
                ene.data_ptr<scalar_t>(),
                (coords.requires_grad()) ? coord_grad.data_ptr<scalar_t>(): nullptr,
                (sigma.requires_grad()) ? sigma_grad.data_ptr<scalar_t>(): nullptr,
                (epsilon.requires_grad()) ? epsilon_grad.data_ptr<scalar_t>() : nullptr,
                (charges.requires_grad()) ? charges_grad.data_ptr<scalar_t>(): nullptr,
                static_cast<scalar_t>(1.0)
            );
        }));
        at::Tensor e = at::sum(ene);
        ctx->save_for_backward({e, coord_grad, sigma_grad, epsilon_grad, charges_grad, coul_constant});
        return e;
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
            ignore,  // box grad (TODO: add this)
            saved[2] * grad_outputs[0], // sigma grad
            saved[3] * grad_outputs[0], // epsilon grad
            saved[4] * grad_outputs[0], // charges grad
            ignore, // coul constant grad
            ignore, // cutoff
            ignore, ignore, ignore, ignore, ignore, // cluster-pair based nblist data structures
            ignore // do-shift
        };
    }

};


at::Tensor compute_nonbonded_energy_from_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Tensor& charges,
    at::Tensor& coul_constant,
    at::Scalar cutoff,
    at::Tensor& sorted_atom_indices,
    at::Tensor& cluster_exclusions,
    at::Tensor& bitmask_exclusions,
    at::Tensor& interacting_clusters,
    at::Tensor& interacting_atoms,
    bool do_shift
)
{
    return NonbondedFromClusterPairsFunctionCuda::apply(
        coords, box,
        sigma, epsilon, charges,
        coul_constant,
        cutoff,
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms,
        do_shift
    );
}


void compute_nonbonded_forces_from_cluster_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& box,
    at::Tensor& sigma,
    at::Tensor& epsilon,
    at::Tensor& charges,
    at::Tensor& coul_constant,
    at::Scalar cutoff,
    at::Tensor& sorted_atom_indices,
    at::Tensor& cluster_exclusions,
    at::Tensor& bitmask_exclusions,
    at::Tensor& interacting_clusters,
    at::Tensor& interacting_atoms,
    at::Tensor& forces
)
{
    // at::linalg_inv does not support CUDA graph
    at::Tensor box_inv, ignore;
    std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto props = at::cuda::getCurrentDeviceProperties();
    int32_t block_dim = 256;
    int32_t warpsPerBlock = block_dim / props->warpSize;
    int32_t grid_dim = std::min(
        props->multiProcessorCount*props->maxBlocksPerMultiProcessor,
        (int32_t)(std::max(cluster_exclusions.size(1), interacting_clusters.size(0)) + warpsPerBlock - 1) / warpsPerBlock
    );
    
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "compute_nonbonded_from_cluster_pairs", ([&] {
        nonbonded_cluster_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            box.data_ptr<scalar_t>(),
            box_inv.data_ptr<scalar_t>(),
            sigma.data_ptr<scalar_t>(),
            epsilon.data_ptr<scalar_t>(),
            charges.data_ptr<scalar_t>(),
            coul_constant.data_ptr<scalar_t>(),
            static_cast<scalar_t>(cutoff.toDouble()),
            true,
            sorted_atom_indices.data_ptr<int32_t>(),
            cluster_exclusions.data_ptr<int32_t>(),
            (int32_t)cluster_exclusions.size(1),
            bitmask_exclusions.data_ptr<uint32_t>(),
            interacting_clusters.data_ptr<int32_t>(),
            interacting_atoms.data_ptr<int32_t>(),
            (int32_t)interacting_clusters.size(0),
            (int32_t)coords.size(0),
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
    m.impl("compute_nonbonded_energy_from_cluster_pairs", compute_nonbonded_energy_from_cluster_pairs_cuda);
    m.impl("compute_nonbonded_forces_from_cluster_pairs", compute_nonbonded_forces_from_cluster_pairs_cuda);
}