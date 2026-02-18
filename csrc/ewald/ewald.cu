#include <torch/extension.h>
#include <torch/autograd.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "kernels.cuh"


class EwaldEnergyFunctionCuda : public torch::autograd::Function<EwaldEnergyFunctionCuda> {

public:

static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, 
    at::Tensor& box, 
    at::Tensor& q, 
    c10::optional<at::Tensor>& p, 
    c10::optional<at::Tensor>& t,
    at::Scalar k_max,  
    at::Scalar alpha
) {

    at::cuda::CUDAGuard guard(coords.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = coords.options();

    const int64_t K = k_max.toLong();
    const int64_t N = coords.size(0);

    // Determine rank from optional tensors
    int64_t rank = 0;
    if (t.has_value()) {
        rank = 2;
    } else if (p.has_value()) {
        rank = 1;
    }

    // Prepare output tensors
    const int64_t M = K * ( K * (4*K+6) + 3 );
    auto kvec = at::empty({4, M}, opts);
    auto Sreal = at::zeros({M}, opts);
    auto Simag = at::zeros({M}, opts);
    auto pot = at::zeros({N}, opts);
    auto field = at::zeros({N,3}, opts);
    auto field_grad = at::zeros({N,3,3}, opts);
    auto forces = at::zeros({N,3}, opts);
    auto energy = at::zeros({}, opts);

    ctx->saved_data["rank"] = rank;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["K"] = K;
    ctx->saved_data["N"] = N;
    ctx->saved_data["M"] = M;

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "ewald_forward", ([&] {
        const scalar_t alpha_val = static_cast<scalar_t>(alpha.toDouble());

        // 1. Prepare k-vectors
        constexpr int BLOCK_SIZE_KVECS = 256;
        const int GRID_SIZE_KVECS = (M + BLOCK_SIZE_KVECS - 1) / BLOCK_SIZE_KVECS;
        prepare_k_constants_kernel<scalar_t><<<GRID_SIZE_KVECS, BLOCK_SIZE_KVECS, 0, stream>>>(
            K, alpha_val,
            box.data_ptr<scalar_t>(),
            kvec.data_ptr<scalar_t>()
        );

        // 2. Forward 
        const int GRID_SIZE_FWD = M;
        constexpr int BLOCK_SIZE = 256;
        if (rank == 0) {
            compute_self_contribution_kernel_rank_0<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 0><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                nullptr,
                nullptr,
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
        } else if (rank == 1) {
            compute_self_contribution_kernel_rank_1<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), p->data_ptr<scalar_t>(),
                N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 1><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                nullptr,
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
        } else {
            compute_self_contribution_kernel_rank_2<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), p->data_ptr<scalar_t>(), t->data_ptr<scalar_t>(),
                N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 2><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                t->data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
        }
    }));
        
    // save for backward pass
    at::Tensor p_saved = p.has_value() ? p.value() : at::Tensor();
    at::Tensor t_saved = t.has_value() ? t.value() : at::Tensor();
    ctx->save_for_backward({
        coords, box, q, p_saved, t_saved, 
        kvec, Sreal, Simag, forces, pot, field, field_grad});

    return energy;
}

static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    // Retrieve saved tensors: {coords, box, q, p_saved, t_saved, kvec, Sreal, Simag, forces, pot, field, field_grad}
    auto saved = ctx->get_saved_variables();
    at::Tensor coords = saved[0];
    at::Tensor box = saved[1];
    at::Tensor q = saved[2];
    at::Tensor p_saved = saved[3];
    at::Tensor t_saved = saved[4];
    at::Tensor kvec = saved[5];
    at::Tensor Sreal = saved[6];
    at::Tensor Simag = saved[7];
    at::Tensor forces = saved[8];
    at::Tensor pot = saved[9];
    at::Tensor field = saved[10];
    at::Tensor field_grad = saved[11];

    // Get gradient w.r.t. energy (scalar)
    at::Tensor grad_energy = grad_outputs[0];

    int64_t rank = ctx->saved_data["rank"].toInt();
    int64_t N = ctx->saved_data["N"].toInt();
    int64_t M = ctx->saved_data["M"].toInt();

    at::cuda::CUDAGuard guard(coords.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = coords.options();

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "ewald_backward", ([&] {
        scalar_t alpha = static_cast<scalar_t>(ctx->saved_data["alpha"].toDouble());
        constexpr int BLOCK_SIZE = 256;
        const int GRID_SIZE_BWD = N;

        // Get pointers for optional tensors (can be null for rank < 1 or rank < 2)
        const scalar_t* p_ptr = (rank >= 1 && p_saved.defined()) ? p_saved.data_ptr<scalar_t>() : nullptr;
        const scalar_t* t_ptr = (rank >= 2 && t_saved.defined()) ? t_saved.data_ptr<scalar_t>() : nullptr;

        if (rank == 0) {
            ewald_backward_kernel<scalar_t, 0, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                nullptr,
                nullptr,
                M, N, box.data_ptr<scalar_t>(), alpha,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        } else if (rank == 1) {
            ewald_backward_kernel<scalar_t, 1, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p_ptr,
                nullptr,
                M, N, box.data_ptr<scalar_t>(), alpha,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        } else if (rank == 2) {
            ewald_backward_kernel<scalar_t, 2, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p_ptr,
                t_ptr,
                M, N, box.data_ptr<scalar_t>(), alpha,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        }
    }));

    // Compute gradients:
    // - dE/dcoords = -forces (forces are -dE/dcoords)
    // - dE/dq = pot (potential at each atom)
    // - dE/dp = -field (negative electric field)
    // - dE/dt = field_grad (field gradient, reshaped appropriately)
    at::Tensor coords_grad = -grad_energy * forces;
    at::Tensor q_grad = grad_energy * pot;
    
    at::Tensor p_grad, t_grad;
    if (rank >= 1) {
        p_grad = -grad_energy * field;
    }
    if (rank >= 2) {
        // field_grad_grad is (N, 9), need to reshape to (N, 3, 3) if needed, but return as (N, 9)
        t_grad = -grad_energy * field_grad * (1.0/3.0);
    }

    at::Tensor ignore;
    torch::autograd::variable_list grads(7);
    grads[0] = coords_grad;                    // d/d coords
    grads[1] = ignore;                        // d/d box
    grads[2] = q_grad;                        // d/d q
    grads[3] = (rank >= 1) ? p_grad : ignore; // d/d p (Dipoles)
    grads[4] = (rank >= 2) ? t_grad : ignore; // d/d t (Quadrupoles)
    grads[5] = ignore;                        // d/d k_max
    grads[6] = ignore;                        // d/d alpha
    return grads;
}

};


at::Tensor ewald_long_range_cuda(
    at::Tensor& coords, 
    at::Tensor& box, 
    at::Tensor& q, 
    c10::optional<at::Tensor> p, 
    c10::optional<at::Tensor> t,
    at::Scalar k_max,  
    at::Scalar alpha
) {
    return EwaldEnergyFunctionCuda::apply(coords, box, q, p, t, k_max, alpha);
}



class EwaldAllFunctionCuda : public torch::autograd::Function<EwaldAllFunctionCuda> {

public:

static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor& coords, 
    at::Tensor& box, 
    at::Tensor& q, 
    c10::optional<at::Tensor>& p, 
    c10::optional<at::Tensor>& t,
    at::Scalar k_max,  
    at::Scalar alpha
) {

    at::cuda::CUDAGuard guard(coords.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = coords.options();

    const int64_t K = k_max.toLong();
    const int64_t N = coords.size(0);

    // Determine rank from optional tensors
    int64_t rank = 0;
    if (t.has_value()) {
        rank = 2;
    } else if (p.has_value()) {
        rank = 1;
    }

    // Prepare output tensors
    const int64_t M = K * ( K * (4*K+6) + 3 );
    auto kvec = at::empty({4, M}, opts);
    auto Sreal = at::zeros({M}, opts);
    auto Simag = at::zeros({M}, opts);
    auto pot = at::zeros({N}, opts);
    auto field = at::zeros({N,3}, opts);
    auto field_grad = at::zeros({N,3,3}, opts);
    auto forces = at::zeros({N,3}, opts);
    auto energy = at::zeros({}, opts);

    ctx->saved_data["rank"] = rank;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["K"] = K;
    ctx->saved_data["N"] = N;
    ctx->saved_data["M"] = M;

    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "ewald_forward", ([&] {
        const scalar_t alpha_val = static_cast<scalar_t>(alpha.toDouble());

        // 1. Prepare k-vectors
        constexpr int BLOCK_SIZE_KVECS = 256;
        const int GRID_SIZE_KVECS = (M + BLOCK_SIZE_KVECS - 1) / BLOCK_SIZE_KVECS;
        prepare_k_constants_kernel<scalar_t><<<GRID_SIZE_KVECS, BLOCK_SIZE_KVECS, 0, stream>>>(
            K, alpha_val,
            box.data_ptr<scalar_t>(),
            kvec.data_ptr<scalar_t>()
        );

        // 2. Forward 
        const int GRID_SIZE_FWD = M;
        const int GRID_SIZE_BWD = N;
        constexpr int BLOCK_SIZE = 256;
        if (rank == 0) {
            compute_self_contribution_kernel_rank_0<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 0><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                nullptr,
                nullptr,
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
            ewald_backward_kernel<scalar_t, 0, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                nullptr,
                nullptr,
                M, N, box.data_ptr<scalar_t>(), alpha_val,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        } else if (rank == 1) {
            compute_self_contribution_kernel_rank_1<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), p->data_ptr<scalar_t>(),
                N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 1><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                nullptr,
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
            ewald_backward_kernel<scalar_t, 1, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                nullptr,
                M, N, box.data_ptr<scalar_t>(), alpha_val,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        } else {
            compute_self_contribution_kernel_rank_2<scalar_t><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                q.data_ptr<scalar_t>(), p->data_ptr<scalar_t>(), t->data_ptr<scalar_t>(),
                N, alpha_val,
                energy.data_ptr<scalar_t>(),
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>()
            );
            ewald_forward_kernel<scalar_t, 2><<<GRID_SIZE_FWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                t->data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>()
            );
            ewald_backward_kernel<scalar_t, 2, BLOCK_SIZE><<<GRID_SIZE_BWD, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                p->data_ptr<scalar_t>(),
                t->data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(), alpha_val,
                pot.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                field_grad.data_ptr<scalar_t>(),
                forces.data_ptr<scalar_t>()
            );
        }
    }));
        
    // save for backward pass
    at::Tensor p_saved = p.has_value() ? p.value() : at::Tensor();
    at::Tensor t_saved = t.has_value() ? t.value() : at::Tensor();
    ctx->save_for_backward({
        coords, box, q, p_saved, t_saved, 
        kvec, Sreal, Simag, forces, pot, field, field_grad});

    // Prepare output list
    torch::autograd::variable_list out(3);
    out[0] = energy;
    out[1] = pot;
    out[2] = field;
    return out;
}

static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    
    int64_t rank = ctx->saved_data["rank"].toInt();
    int64_t K = ctx->saved_data["K"].toInt();
    int64_t N = ctx->saved_data["N"].toInt();
    int64_t M = ctx->saved_data["M"].toInt();

    // Retrieve saved tensors: {coords, box, q, p_saved, t_saved, kvec, Sreal, Simag, forces, pot, field, field_grad}
    auto saved = ctx->get_saved_variables();
    at::Tensor coords = saved[0];
    at::Tensor box = saved[1];
    at::Tensor q = saved[2];
    at::Tensor p = saved[3];
    at::Tensor t = saved[4];
    at::Tensor kvec = saved[5];
    at::Tensor Sreal = saved[6];
    at::Tensor Simag = saved[7];
    at::Tensor forces = saved[8];
    at::Tensor pot = saved[9];
    at::Tensor field = saved[10];
    at::Tensor field_grad = saved[11];

    at::cuda::CUDAGuard guard(coords.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts = coords.options();

    // Get gradient w.r.t. energy (scalar)
    at::Tensor grad_energy = grad_outputs[0];
    at::Tensor grad_pot = grad_outputs[1];
    at::Tensor grad_field = grad_outputs[2];

    // Create zero tensors for grad_pot and grad_field if they're not defined
    // This ensures we can safely call .data_ptr() on them in the kernel
    if (!grad_pot.defined()) {
        grad_pot = at::zeros({N}, opts);
    }
    if (!grad_field.defined()) {
        grad_field = at::zeros({N, 3}, opts);
    }   

    // Compute gradients:
    at::Tensor coords_grad = grad_energy.defined() ? -grad_energy * forces : at::zeros({N,3}, opts);    
    at::Tensor q_grad = grad_energy.defined() ? grad_energy * pot : at::zeros({N}, opts);
    at::Tensor p_grad, t_grad;
    if (rank >= 1) {
        p_grad = grad_energy.defined() ? -grad_energy * field : at::zeros({N, 3}, opts);
    }
    if (rank >= 2) {
        t_grad = grad_energy.defined() ? -grad_energy * field_grad * (1.0/3.0) : at::zeros({N, 3, 3}, opts);
    }

    // Aux tensors for the Fourier gradient kernel
    at::Tensor Fdr = at::zeros({M}, opts);
    at::Tensor Fdi = at::zeros({M}, opts);
    
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "ewald_backward_2", ([&] {
        scalar_t alpha = static_cast<scalar_t>(ctx->saved_data["alpha"].toDouble());
        constexpr int BLOCK_SIZE = 256;

        if ( rank == 2 ) {
            ewald_fourier_gradient_kernel<scalar_t, 2, BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(), Fdi.data_ptr<scalar_t>()
            );
            ewald_backward_with_fields_kernel<scalar_t, 2, BLOCK_SIZE><<<N, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(), Fdi.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), t.data_ptr<scalar_t>(),
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(), Simag.data_ptr<scalar_t>(),
                M, alpha, box.data_ptr<scalar_t>(),
                coords_grad.data_ptr<scalar_t>(),
                q_grad.data_ptr<scalar_t>(), p_grad.data_ptr<scalar_t>(), t_grad.data_ptr<scalar_t>()
            );
        }
        else if ( rank == 1 ) {
            q_grad = grad_energy.defined() ? grad_energy * pot : at::zeros({N}, opts);
            p_grad = grad_energy.defined() ? -grad_energy * field : at::zeros({N, 3}, opts);
            ewald_fourier_gradient_kernel<scalar_t, 1, BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(), Fdi.data_ptr<scalar_t>()
            );
            ewald_backward_with_fields_kernel<scalar_t, 1, BLOCK_SIZE><<<N, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(),
                Fdi.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), nullptr,
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                M, alpha, box.data_ptr<scalar_t>(),
                coords_grad.data_ptr<scalar_t>(),
                q_grad.data_ptr<scalar_t>(), p_grad.data_ptr<scalar_t>(), nullptr
            );
        }
        else {
            ewald_fourier_gradient_kernel<scalar_t, 0, BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                M, N, box.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(), Fdi.data_ptr<scalar_t>()
            );
            ewald_backward_with_fields_kernel<scalar_t, 0, BLOCK_SIZE><<<N, BLOCK_SIZE, 0, stream>>>(
                kvec.data_ptr<scalar_t>(),
                Fdr.data_ptr<scalar_t>(), Fdi.data_ptr<scalar_t>(),
                coords.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(), nullptr, nullptr,
                grad_pot.data_ptr<scalar_t>(),
                grad_field.data_ptr<scalar_t>(),
                Sreal.data_ptr<scalar_t>(),
                Simag.data_ptr<scalar_t>(),
                M, alpha, box.data_ptr<scalar_t>(),
                coords_grad.data_ptr<scalar_t>(),
                q_grad.data_ptr<scalar_t>(), nullptr, nullptr
            );
        }
    }));

    at::Tensor ignore;
    torch::autograd::variable_list grads(7);
    grads[0] = coords_grad;                   // d/d coords
    grads[1] = ignore;                        // d/d box
    grads[2] = q_grad;                        // d/d q
    grads[3] = p_grad;                        // d/d p (Dipoles)
    grads[4] = t_grad;                        // d/d t (Quadrupoles)
    grads[5] = ignore;                        // d/d k_max
    grads[6] = ignore;                        // d/d alpha
    return grads;
}

};


std::tuple<at::Tensor, at::Tensor, at::Tensor> ewald_long_range_all_cuda(
    at::Tensor& coords, 
    at::Tensor& box, 
    at::Tensor& q, 
    c10::optional<at::Tensor> p, 
    c10::optional<at::Tensor> t,
    at::Scalar k_max,  
    at::Scalar alpha
) {
    auto outs = EwaldAllFunctionCuda::apply(coords, box, q, p, t, k_max, alpha);
    return {outs[0], outs[1], outs[2]};
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("ewald_long_range", ewald_long_range_cuda);
    m.impl("ewald_long_range_all", ewald_long_range_all_cuda);
}