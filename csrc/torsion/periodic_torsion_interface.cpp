#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY(torchff, m) {
    m.def("compute_periodic_torsion(Tensor coords, Tensor torsions, Tensor fc, Tensor per, Tensor phase) -> Tensor, Tensor, Tensor, Tensor");
}

PYBIND11_MODULE(torchff_periodic_torsion, m) {
    m.doc() = "torchff periodic torsion CUDA extension";
}