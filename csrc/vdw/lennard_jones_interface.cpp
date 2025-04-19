#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_lennard_jones(Tensor coords, Tensor pairs, Tensor box, Tensor sigma, Tensor epsilon, float cutoff) -> Tensor[]");
}

PYBIND11_MODULE(torchff_vdw, m) {
    m.doc() = "torchff vdw CUDA extension";
}