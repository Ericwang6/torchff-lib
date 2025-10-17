import torch
import torchff_torsion


def compute_periodic_torsion_energy(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    per: torch.Tensor,
    phase: torch.Tensor
) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return torch.ops.torchff.compute_periodic_torsion_energy(coords, torsions, fc, per, phase)


def compute_periodic_torsion_forces(
    coords: torch.Tensor, 
    torsions: torch.Tensor, 
    fc: torch.Tensor, 
    per: torch.Tensor,
    phase: torch.Tensor,
    forces: torch.Tensor
) -> None:
    """
    Compute periodic torsion forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_periodic_torsion_forces(coords, torsions, fc, per, phase, forces)