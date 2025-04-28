import torch
import torchff_vdw


def compute_lennard_jones_energy(
    coords: torch.Tensor, 
    pairs: torch.Tensor, 
    box: torch.Tensor, 
    sigma: torch.Tensor, 
    epsilon: torch.Tensor,
    cutoff: float
) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return torch.ops.torchff.compute_lennard_jones_energy(coords, pairs, box, sigma, epsilon, cutoff)