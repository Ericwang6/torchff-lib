#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_lennard_jones_energy(Tensor coords, Tensor pairs, Tensor box, Tensor sigma, Tensor epsilon, Scalar cutoff) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff vdw CUDA extension";
}