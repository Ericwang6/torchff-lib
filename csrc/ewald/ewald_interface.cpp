#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
  m.def("ewald_long_range(Tensor coords, Tensor box, "
        "Tensor q, Tensor p, Tensor t, int K, int rank, float alpha) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "torchff multipolar Ewald CUDA extension";
}

