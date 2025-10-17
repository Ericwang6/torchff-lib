#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "bond/morse_bond.cuh"


template <typename scalar_t>
__global__ void cmm_field_dependent_morse_bond_kernel(
    scalar_t* coords,
    int32_t* bonds,
    scalar_t* req_0,
    scalar_t* kb_0,
    scalar_t* d,
    scalar_t* dipole_deriv_1,
    scalar_t* dipole_deriv_2,
    scalar_t* efield,
    int32_t nbonds,
    scalar_t* ene,
    scalar_t* coords_grad,
    scalar_t* efield_grad
)
{

    int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t index = start; index < nbonds; index += gridDim.x * blockDim.x) {
        int32_t i = bonds[index * 2];
        int32_t j = bonds[index * 2 + 1];
        scalar_t rx = coords[j*3] - coords[i*3];
        scalar_t ry = coords[j*3+1] - coords[i*3+1];
        scalar_t rz = coords[j*3+2] - coords[i*3+2];
        scalar_t r = sqrt_(rx*rx+ry*ry+rz*rz);
        scalar_t e, drx, dry, drz, defx, defy, defz;
        fd_morse_bond_from_kb_d(
            r, rx, ry, rz, req_0[index], kb_0[index], d[index],
            efield[j*3], efield[j*3+1], efield[j*3+2], dipole_deriv_1[index], dipole_deriv_2[index],
            &e, &drx, &dry, &drz, &defx, &defy, &defz
        );
        ene[index] = e;
        atomicAdd(&coords_grad[i*3], -drx);
        atomicAdd(&coords_grad[i*3+1], -dry);
        atomicAdd(&coords_grad[i*3+2], -drz);

        atomicAdd(&coords_grad[j*3], drx);
        atomicAdd(&coords_grad[j*3+1], dry);
        atomicAdd(&coords_grad[j*3+2], drz);

        atomicAdd(&efield_grad[j*3], defx);
        atomicAdd(&efield_grad[j*3+1], defy);
        atomicAdd(&efield_grad[j*3+2], defz);

    }
}


class CMMFieldDependentMorseBondCuda: public torch::autograd::Function<CMMFieldDependentMorseBondCuda> {

public: 

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, at::Tensor& bonds,
    at::Tensor& req_0, at::Tensor& kb_0, at::Tensor& D,
    at::Tensor& dipole_deriv_1, at::Tensor& dipole_deriv_2, 
    at::Tensor& efield
)
{
    int32_t nbonds = bonds.size(0);
    at::Tensor coords_grad = at::zeros_like(coords, coords.options());
    at::Tensor efield_grad = at::zeros_like(efield, efield.options());
    at::Tensor ene = at::zeros({nbonds}, coords.options());

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream();
    int32_t block_dim = 256;
    int32_t grid_dim = std::min(props->maxBlocksPerMultiProcessor*props->multiProcessorCount, (nbonds+block_dim-1)/block_dim);

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cmm_field_dependent_morse_bond_kernel", ([&] {
        cmm_field_dependent_morse_bond_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
            coords.data_ptr<scalar_t>(),
            bonds.data_ptr<int32_t>(),
            req_0.data_ptr<scalar_t>(),
            kb_0.data_ptr<scalar_t>(),
            D.data_ptr<scalar_t>(),
            dipole_deriv_1.data_ptr<scalar_t>(),
            dipole_deriv_2.data_ptr<scalar_t>(),
            efield.data_ptr<scalar_t>(),
            nbonds,
            ene.data_ptr<scalar_t>(),
            coords_grad.data_ptr<scalar_t>(),
            efield_grad.data_ptr<scalar_t>()
        );
    }));
    
    ctx->save_for_backward({coords_grad, efield_grad});
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
        saved[0] * grad_outputs[0], 
        ignore, ignore, ignore, ignore, ignore, ignore,
        saved[1] * grad_outputs[0]
    };
}

};

at::Tensor cmm_field_dependent_morse_bond_cuda(
    at::Tensor& coords, at::Tensor& bonds,
    at::Tensor& req_0, at::Tensor& kb_0, at::Tensor& D,
    at::Tensor& dipole_deriv_1, at::Tensor& dipole_deriv_2, 
    at::Tensor& efield
) {
    return CMMFieldDependentMorseBondCuda::apply(
        coords, bonds, req_0, kb_0, D, dipole_deriv_1, dipole_deriv_2,
        efield
    );
}

TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("cmm_field_dependent_morse_bond", cmm_field_dependent_morse_bond_cuda);
}