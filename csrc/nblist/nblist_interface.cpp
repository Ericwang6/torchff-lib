#include <pybind11/pybind11.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("build_neighbor_list_nsquared(Tensor coords, Tensor box, Scalar cutoff, Scalar max_npairs, bool padding) -> (Tensor, Tensor)");
    m.def("build_neighbor_list_cell_list(Tensor coords, Tensor box, Scalar cutoff, Scalar max_npairs, Scalar cell_size, bool padding) -> (Tensor, Tensor)");
    m.def("build_neighbor_list_cell_list_shared(Tensor coords, Tensor box, Scalar cutoff, Scalar max_npairs, Scalar cell_size, bool padding) -> (Tensor, Tensor)");
    m.def("build_cluster_pairs(Tensor coords, Tensor box, Scalar cutoff, Tensor exclusions, Scalar cell_size, Scalar max_num_interacting_clusters) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("decode_cluster_pairs(Tensor coords, Tensor box, Tensor sorted_atom_indices, Tensor interacting_clusters, Tensor bitmask_exclusions, Scalar cutoff, Scalar max_npairs, Scalar num_interacting_clusters, bool padding) -> (Tensor, Tensor)");
    // m.def("build_neighbor_list_nnpop(Tensor positions, Scalar cutoff, Scalar max_num_neighbors, Tensor box_vectors, bool checkErrors) -> (Tensor neighbors, Tensor deltas, Tensor distances, Tensor num_pairs)");
    // m.def("build_neighbor_list_torchmdnet(Tensor positions, Tensor batch, Tensor in_box_size, bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool loop, bool include_transpose) -> (Tensor, Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(torchff_nblist, m) {
    m.doc() = "torchff neighbor list extension";
}