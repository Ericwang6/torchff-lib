#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("build_neighbor_list_nsquared(Tensor coords, Tensor box, float cutoff, int max_npairs) -> Tensor");
    m.def("build_neighbor_list_cell_list(Tensor coords, Tensor box, float cutoff, int max_npairs) -> Tensor");
}

PYBIND11_MODULE(torchff_nblist, m) {
    m.doc() = "torchff neighbor list extension";
}