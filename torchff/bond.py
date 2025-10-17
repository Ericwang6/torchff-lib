import torch
import torchff_bond


def compute_harmonic_bond_energy(coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """Compute harmonic bond energies"""
    return torch.ops.torchff.compute_harmonic_bond_energy(coords, bonds, b0, k)


def compute_harmonic_bond_forces(
    coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor,
    forces: torch.Tensor
):
    """
    Compute harmonic bond forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_harmonic_bond_forces(coords, bonds, b0, k, forces)