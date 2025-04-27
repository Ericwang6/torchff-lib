import torch
import torchff_harmonic_bond


def compute_harmonic_bond_energy(coords: torch.Tensor, pairs: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """Compute harmonic bond energies"""
    return torch.ops.torchff.compute_harmonic_bond_energy(coords, pairs, b0, k)
