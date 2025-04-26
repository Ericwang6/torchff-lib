#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("coulomb_cuda_kernel(Tensor coords, Tensor charges, Tensor pairs, Tensor box, Tensor epsilon, float cutoff) -> Tensor[]");
}

PYBIND11_MODULE(torchff_coulomb, m) {
    m.doc() = "torchff Coulomb CUDA extension";
}
