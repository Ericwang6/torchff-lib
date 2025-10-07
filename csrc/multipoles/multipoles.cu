#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "multipoles.cuh"

template <typename scalar_t>
__global__ void multipolar_interaction_atom_pairs_kernel(
    scalar_t* coords,
    int32_t* pairs,
    scalar_t* multipoles,
    int32_t npairs,
    scalar_t* ene,
    scalar_t* coord_grad,
    scalar_t* multipoles_grad
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
        scalar_t* mi = multipoles + i * 10;
        scalar_t* mj = multipoles + j * 10;
        scalar_t mi_grad[10] = {};
        scalar_t mj_grad[10] = {};
        scalar_t drx_grad = scalar_t(0.0);
        scalar_t dry_grad = scalar_t(0.0);
        scalar_t drz_grad = scalar_t(0.0);

        constexpr scalar_t ONE = scalar_t(1.0);
        pairwise_multipole_kernel_with_grad(
            mi[0], mi[1], mi[2], mi[3], mi[4], mi[5], mi[6], mi[7], mi[8], mi[9],
            mj[0], mj[1], mj[2], mj[3], mj[4], mj[5], mj[6], mj[7], mj[8], mj[9],
            coords[j*3] - coords[i*3], 
            coords[j*3+1] - coords[i*3+1],
            coords[j*3+2] - coords[i*3+2],
            ONE, ONE, ONE, ONE, ONE, ONE,
            &e,
            mi_grad, mi_grad+1, mi_grad+2, mi_grad+3, mi_grad+4, mi_grad+5, mi_grad+6, mi_grad+7, mi_grad+8, mi_grad+9,
            mj_grad, mj_grad+1, mj_grad+2, mj_grad+3, mj_grad+4, mj_grad+5, mj_grad+6, mj_grad+7, mj_grad+8, mj_grad+9,
            &drx_grad, &dry_grad, &drz_grad
        );
        ene[index] = e;
        atomicAdd(&coord_grad[i*3], -drx_grad); atomicAdd(&coord_grad[i*3+1], -dry_grad); atomicAdd(&coord_grad[i*3+2], -drz_grad);
        atomicAdd(&coord_grad[j*3], drx_grad); atomicAdd(&coord_grad[j*3+1], dry_grad); atomicAdd(&coord_grad[j*3+2], drz_grad);
        #pragma unroll
        for (int n = 0; n < 10; n++) {
            atomicAdd(&multipoles_grad[i*10+n], mi_grad[n]);
            atomicAdd(&multipoles_grad[j*10+n], mj_grad[n]);
        }
    }
}

class MultipolarInteractionAtomPairsFunctionCuda: public torch::autograd::Function<MultipolarInteractionAtomPairsFunctionCuda> {

public: 

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& multipoles
)
{
    int32_t npairs = pairs.size(0);
    at::Tensor ene = at::zeros({npairs}, coords.options());
    at::Tensor coords_grad = at::zeros_like(coords, coords.options());
    at::Tensor multipoles_grad = at::zeros_like(multipoles, multipoles.options());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = 128;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (npairs+block_dim-1)/block_dim);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "multipolar_interaction_atom_pairs_cuda", ([&] {
        multipolar_interaction_atom_pairs_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            pairs.data_ptr<int32_t>(),
            multipoles.data_ptr<scalar_t>(),
            npairs,
            ene.data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>(),
            multipoles_grad.data_ptr<scalar_t>()
        );
    }));
    ctx->save_for_backward({ene, coords_grad, multipoles_grad});
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
        saved[2] * grad_outputs[0], // multipoles grad
    };
}

};

at::Tensor compute_multipolar_energy_from_atom_pairs_cuda(
    at::Tensor& coords,
    at::Tensor& pairs,
    at::Tensor& multipoles
)
{
    return MultipolarInteractionAtomPairsFunctionCuda::apply(
        coords, pairs, multipoles
    );
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_multipolar_energy_from_atom_pairs", compute_multipolar_energy_from_atom_pairs_cuda);
}