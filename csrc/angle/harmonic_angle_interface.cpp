#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_harmonic_angle_energy(Tensor coords, Tensor angles, Tensor theta0, Tensor k) -> Tensor");
    m.def("compute_harmonic_angle_forces(Tensor coords, Tensor angles, Tensor theta0, Tensor k, Tensor (a!) forces) -> ()");
}

PYBIND11_MODULE(torchff_harmonic_angle, m) {
    m.doc() = "torchff harmonic angle CUDA extension";
}
