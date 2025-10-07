#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_multipolar_energy_from_atom_pairs(Tensor coords, Tensor pairs, Tensor multipoles) -> Tensor");
}

PYBIND11_MODULE(torchff_multipoles, m) {
    m.doc() = "torchff multipolar CUDA extension";
}
