import torch
import torch.nn as nn
import torchff_angle


def compute_harmonic_angle_energy(coords: torch.Tensor, angles: torch.Tensor, theta0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute harmonic angle energies"""
    return torch.ops.torchff.compute_harmonic_angle_energy(coords, angles, theta0, k)


def compute_harmonic_angle_energy_ref(coords, angles, theta0, k):
    v1 = coords[angles[:, 0]] - coords[angles[:, 1]]
    v2 = coords[angles[:, 2]] - coords[angles[:, 1]]
    dot_product = torch.sum(v1 * v2, dim=1)
    mag_v1 = torch.norm(v1, dim=1)
    mag_v2 = torch.norm(v2, dim=1)
    cos_theta = dot_product / (mag_v1 * mag_v2)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical issues
    theta = torch.acos(cos_theta)
    ene = (theta - theta0) ** 2 * k / 2
    return torch.sum(ene)


class HarmonicAngle(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(self, coords, angles, theta0, k):
        if self.use_customized_ops:
            return compute_harmonic_angle_energy(coords, angles, theta0, k)
        else:
            return compute_harmonic_angle_energy_ref(coords, angles, theta0, k)


def compute_harmonic_angle_forces(
    coords: torch.Tensor, 
    angles: torch.Tensor, 
    theta0: torch.Tensor, 
    k: torch.Tensor,
    forces: torch.Tensor
) -> torch.Tensor:
    """
    Compute harmonic angle forces in-place, backward calculation does not supported, used for fast MD
    """
    return torch.ops.torchff.compute_harmonic_angle_forces(coords, angles, theta0, k, forces)