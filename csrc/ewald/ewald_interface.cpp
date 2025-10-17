#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("ewald_long_range_potential(Tensor coords, Tensor box, Tensor q, Tensor p, Tensor t, Tensor all_hkl, Tensor sym_factors, Scalar alpha, Scalar rank) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff multipolar Ewald CUDA extension";
}