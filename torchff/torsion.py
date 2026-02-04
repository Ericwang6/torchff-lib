import torch
import torch.nn as nn
import torchff_torsion


def compute_periodic_torsion_energy(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    periodicity: torch.Tensor,
    phase: torch.Tensor
) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return torch.ops.torchff.compute_periodic_torsion_energy(coords, torsions, fc, periodicity, phase)


def compute_periodic_torsion_energy_ref(coords, torsions, fc, periodicity, phase):
    b1 = coords[torsions[:, 1]] - coords[torsions[:, 0]]
    b2 = coords[torsions[:, 2]] - coords[torsions[:, 1]]
    b3 = coords[torsions[:, 3]] - coords[torsions[:, 2]]
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    norm_n1 = torch.norm(n1, dim=1)
    norm_n2 = torch.norm(n2, dim=1)
    cosval = torch.clamp(
        torch.sum(n1 * n2, dim=1) / (norm_n1 * norm_n2), 
        -0.999999999, 0.999999999
    )
    phi = torch.acos(cosval) * torch.sign(torch.sum(n1 * b3, axis=1))
    ene = fc * (1 + torch.cos(periodicity * phi - phase))
    return torch.sum(ene)


class PeriodicTorsion(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, torsions, fc, periodicity, phase):
        if self.use_customized_ops:
            return compute_periodic_torsion_energy(coords, torsions, fc, periodicity, phase)
        else:
            return compute_periodic_torsion_energy_ref(coords, torsions, fc, periodicity, phase)


def compute_periodic_torsion_forces(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    periodicity: torch.Tensor,
    phase: torch.Tensor,
    forces: torch.Tensor
) -> None:
    """
    Compute periodic torsion forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_periodic_torsion_forces(coords, torsions, fc, periodicity, phase, forces)