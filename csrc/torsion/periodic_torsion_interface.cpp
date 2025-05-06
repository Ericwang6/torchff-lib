#include <pybind11/pybind11.h>
#include <torch/library.h>


TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_periodic_torsion_energy(Tensor coords, Tensor torsions, Tensor fc, Tensor per, Tensor phase) -> (Tensor)");
    m.def("compute_periodic_torsion_forces(Tensor coords, Tensor torsions, Tensor fc, Tensor per, Tensor phase, Tensor (a!) forces) -> ()");
}

PYBIND11_MODULE(torchff_periodic_torsion, m) {
    m.doc() = "torchff periodic torsion CUDA extension";
}