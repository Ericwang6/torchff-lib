#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY(torchff, m) {
    m.def("compute_harmonic_angle_energy(Tensor coords, Tensor triplets, Tensor theta0, Tensor k) -> Tensor");
    m.def("compute_harmonic_angle_energy_grad(Tensor coords, Tensor triplets, Tensor theta0, Tensor k) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(torchff_harmonic_angle, m) {
    m.doc() = "torchff harmonic angle CUDA extension";
}
