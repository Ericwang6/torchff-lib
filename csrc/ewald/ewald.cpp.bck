#include <torch/autograd.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using at::Tensor;
using at::Scalar;
#define INV_ROOT_PI 0.5641895835477563;


std::tuple<Tensor, Tensor, Tensor> ewald_long_range_potential(
    Tensor& coords,
    Tensor& box,
    Tensor& q,
    Tensor& p,
    Tensor& t,
    Tensor& all_hkl,
    Tensor& sym_factors,
    Scalar alpha,
    Scalar rank
) {
    
    // Volume & Reciprocal
    Tensor V_t = at::det(box); 
    Tensor box_inv, ignore;
    std::tie(box_inv, ignore) = at::linalg_inv_ex(box, false);

    // k^2 and symmetry factors
    Tensor kvectors = at::matmul(all_hkl, box_inv); // (K',3)
    Tensor k_squared = at::einsum("ij,ij->i", {kvectors, kvectors}); // (K',)

    // Gaussian factor: exp(-pi^2 k^2 / alpha^2) / k^2
    double alpha2 = alpha.toDouble() * alpha.toDouble();
    Tensor gaussian_factors = at::exp(- M_PI * M_PI * k_squared / alpha2) / k_squared;        // (K',)

    // k.r
    Tensor k_dot_r = at::matmul(kvectors, coords.t());                // (K',N)
    Tensor cos_k_dot_r = at::cos(2.0 * M_PI * k_dot_r);               // (K',N)
    Tensor sin_k_dot_r = at::sin(2.0 * M_PI * k_dot_r);               // (K',N)
    Tensor exp_k_dot_r      = at::complex(cos_k_dot_r,  sin_k_dot_r); // (K',N), complex
    Tensor exp_minus_k_dot_r= at::complex(cos_k_dot_r, -sin_k_dot_r); // (K',N), complex

    Tensor F_l_real, F_l_imag;
    if ( rank.toInt() == 2 ) {
        F_l_real = q.expand({kvectors.size(0), q.size(0)}) - at::einsum("kj,nij,ki->kn", {kvectors, t, kvectors}) * (2.0 * M_PI) * (2.0 * M_PI) / 3.0; // (K',N)
        F_l_imag = at::matmul(kvectors, p.t()) * (2.0 * M_PI);          // (K',N)
    }
    else if ( rank.toInt() == 1 ) {
        F_l_real = q.expand({kvectors.size(0), q.size(0)});
        F_l_imag = at::matmul(kvectors, p.t()) * (2.0 * M_PI);
    }
    else {
        F_l_real = q.expand({kvectors.size(0), q.size(0)});
        F_l_imag = at::zeros({kvectors.size(0), q.size(0)}, q.options());
    }

    Tensor F_2 = at::complex(F_l_real, F_l_imag);                           // (K',N) complex

    // structure_factors: sum over atoms n of F_2 * e^{i k·r}
    Tensor structure_factors = (F_2 * exp_k_dot_r).sum(1);          // (K') complex

    // phi_expanded per-atom:
    Tensor phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * sym_factors) * exp_minus_k_dot_r;      // (K',N) complex

    double alpha_over_root_pi = alpha.toDouble() * INV_ROOT_PI;
    Tensor potential, field, field_grad;

    potential = at::real(phi_expanded).sum(0) / (M_PI * V_t) - (2.0 * alpha_over_root_pi) * q;
    if ( rank.toInt() > 0 ) {
        field = (2.0 / V_t) * at::real(at::matmul(phi_expanded.t(), at::complex(at::zeros_like(kvectors), kvectors))) + alpha_over_root_pi * (4.0 * alpha2 / 3.0) * p;
    }
    if ( rank.toInt() > 1 ) {
        Tensor k_outer = at::einsum("bi,bj->bij", {kvectors, kvectors}).reshape({-1, 9});  // (K',9)
        field_grad = (4.0 * M_PI / V_t) * at::real(at::matmul(phi_expanded.t(), at::complex(k_outer, at::zeros_like(k_outer)))).reshape({-1, 3, 3}); // (N,3,3)
        field_grad = field_grad + alpha_over_root_pi * (16.0 * alpha2 * alpha2 / 5.0) / 3.0 * t;
    }
    return {potential, field, field_grad};
}


TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("ewald_long_range_potential", ewald_long_range_potential);
}