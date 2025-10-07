#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
  m.def(
    "cmm_non_elec_nonbonded_interaction_from_pairs("
      "Tensor dist_vecs, Tensor pairs, Tensor multipoles, "
      "Tensor q_pauli, Tensor Kdipo_pauli, Tensor Kquad_pauli, Tensor b_pauli_ij, "
      "Tensor q_xpol, Tensor Kdipo_xpol, Tensor Kquad_xpol, Tensor b_xpol_ij, "
      "Tensor q_ct_don, Tensor Kdipo_ct_don, Tensor Kquad_ct_don, "
      "Tensor q_ct_acc, Tensor Kdipo_ct_acc, Tensor Kquad_ct_acc, Tensor b_ct_ij, Tensor eps_ct_ij, "
      "Tensor C6_disp_ij, Tensor b_disp_ij, "
      "Scalar rcut_sr, Scalar rcut_lr, Scalar rcut_switch_buf"
    ") -> (Tensor, Tensor)"
  );
  m.def(
    "cmm_elec_from_pairs("
      "Tensor dist_vecs, Tensor pairs, "
      "Tensor dist_vecs_excl, Tensor pairs_excl, "
      "Tensor multipoles, "
      "Tensor Z, Tensor b_elec_ij, Tensor b_elec, "
      "Scalar ewald_alpha, "
      "Scalar rcut_sr, Scalar rcut_lr, Scalar rcut_switch_buf"
    ") -> (Tensor, Tensor, Tensor)"
  );
}

PYBIND11_MODULE(torchff_cmm, m) {
    m.doc() = "torchff CMM CUDA extension";
}
