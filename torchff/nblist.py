from typing import Optional, Tuple
import torch
import torchff_nblist


def build_neighbor_list_nsquared(coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_npairs: int = -1, padding: bool = False):
    return torch.ops.torchff.build_neighbor_list_nsquared(coords, box, cutoff, max_npairs, padding)


def build_neighbor_list_cell_list(coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_npairs: int = -1, cell_size: float = 0.4, padding: bool = False, shared: bool = False):
    if shared:
        return torch.ops.torchff.build_neighbor_list_cell_list_shared(coords, box, cutoff, max_npairs, cell_size, padding)
    else:
        return torch.ops.torchff.build_neighbor_list_cell_list(coords, box, cutoff, max_npairs, cell_size, padding)


def build_cluster_pairs(
    coords: torch.Tensor, box: torch.Tensor,
    cutoff: float, 
    exclusions: Optional[torch.Tensor] = None,
    cell_size: float = 0.4,
    max_num_interacting_clusters: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if exclusions is None:
        exclusions = torch.full((coords.shape[0], 1), -1, dtype=torch.int32, device=coords.device)
    
    sorted_atom_indices, interacting_clusters, bitmask_exclusions, num_interacting_clusters = torch.ops.torchff.build_cluster_pairs(
        coords,
        box,
        cutoff,
        exclusions,
        cell_size,
        max_num_interacting_clusters
    )
    return (
        sorted_atom_indices, interacting_clusters, 
        bitmask_exclusions, num_interacting_clusters
    )


def decode_cluster_pairs(
    coords: torch.Tensor, 
    box: torch.Tensor,
    sorted_atom_indices: torch.Tensor,
    interacting_clusters: torch.Tensor,
    bitmask_exclusions: torch.Tensor,
    cutoff: float,
    max_npairs: int = -1,
    num_interacting_clusters: int = -1,
    padding: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torchff.decode_cluster_pairs(
        coords, box, 
        sorted_atom_indices, interacting_clusters, bitmask_exclusions,
        cutoff, max_npairs, num_interacting_clusters, padding
    )


def build_neighbor_list_cluster_pairs(
    coords: torch.Tensor,
    box: torch.Tensor,
    cutoff: float,
    exclusions: Optional[torch.Tensor] = None,
    cell_size: float = 0.45,
    max_num_interacting_clusters: int = -1,
    max_npairs: int = -1,
    padding: bool = False
):
    sorted_atom_indices, interacing_clusters, bitmask_exclusions, num_interacting_clusters = build_cluster_pairs(
        coords, box,
        cutoff, exclusions,
        cell_size, max_num_interacting_clusters
    )
    # print(interacing_clusters)
    # print(bitmask_exclusions)
    # print("Found number of interacting clusters:", num_interacting_clusters.item())
    return decode_cluster_pairs(
        coords, box, sorted_atom_indices, interacing_clusters,
        bitmask_exclusions, cutoff,
        max_npairs, num_interacting_clusters.item(),
        padding
    )
