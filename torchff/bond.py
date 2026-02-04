import torch
import torch.nn as nn
import torchff_bond


def compute_harmonic_bond_energy(coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """Compute harmonic bond energies"""
    return torch.ops.torchff.compute_harmonic_bond_energy(coords, bonds, b0, k)


def compute_harmonic_bond_energy_ref(coords, bonds, b0, k):
    r = torch.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], dim=1)
    ene = (r - b0) ** 2 * k / 2
    return torch.sum(ene)


class HarmonicBond(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, bonds, b0, k):
        if self.use_customized_ops:
            return compute_harmonic_bond_energy(coords, bonds, b0, k)
        else:
            return compute_harmonic_bond_energy_ref(coords, bonds, b0, k)


def compute_harmonic_bond_forces(
    coords: torch.Tensor, bonds: torch.Tensor, b0: torch.Tensor, k: torch.Tensor,
    forces: torch.Tensor
):
    """
    Compute harmonic bond forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_harmonic_bond_forces(coords, bonds, b0, k, forces)