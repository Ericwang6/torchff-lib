import torch
import torchff_harmonic_angle


def compute_harmonic_angle_energy(coords: torch.Tensor, angles: torch.Tensor, theta0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute harmonic angle energies"""
    return torch.ops.torchff.compute_harmonic_angle_energy(coords, angles, theta0, k)
