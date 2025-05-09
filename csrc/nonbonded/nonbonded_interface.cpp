#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_nonbonded_energy_from_cluster_pairs(Tensor coords, Tensor box, Tensor sigma, Tensor epsilon, Tensor charges, Tensor coul_constant, Scalar cutoff, Tensor sorted_atom_indices, Tensor interacting_clusters, Tensor bitmask_exclusions, bool do_shift) -> Tensor");
    m.def("compute_nonbonded_forces_from_cluster_pairs(Tensor coords, Tensor box, Tensor sigma, Tensor epsilon, Tensor charges, Tensor coul_constant, Scalar cutoff, Tensor sorted_atom_indices, Tensor interacting_clusters, Tensor bitmask_exclusions, Tensor forces) -> ()");
    m.def("compute_nonbonded_energy_from_atom_pairs(Tensor coords, Tensor pairs, Tensor box, Tensor sigma, Tensor epsilon, Tensor charges, Tensor coul_constant, Scalar cutoff, bool do_shift) -> Tensor");
    m.def("compute_nonbonded_forces_from_atom_pairs(Tensor coords, Tensor pairs, Tensor box, Tensor sigma, Tensor epsilon, Tensor charges, Tensor coul_constant, Scalar cutoff, Tensor forces) -> ()");
}

PYBIND11_MODULE(torchff_nb, m) {
    m.doc() = "torchff nonbonded CUDA extension";
}

