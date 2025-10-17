#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_coulomb_energy(Tensor coords, Tensor pairs, Tensor box, Tensor charges, Tensor prefac, Scalar cutoff, bool do_shift) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff Coulomb CUDA extension";
}
