#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_harmonic_bond_energy(Tensor coords, Tensor pairs, Tensor b0, Tensor k) -> Tensor");
    m.def("compute_harmonic_bond_energy_and_forces(Tensor coords, Tensor pairs, Tensor b0, Tensor k, Tensor (a!) ene, Tensor (a!) forces) -> ()");
}

PYBIND11_MODULE(torchff_harmonic_bond, m) {
    m.doc() = "torchff harmonic bond CUDA extension";
}
