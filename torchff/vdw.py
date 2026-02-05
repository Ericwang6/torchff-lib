import torch
import torch.nn as nn
import torchff_vdw


def compute_lennard_jones_energy(
    coords: torch.Tensor, 
    pairs: torch.Tensor, 
    box: torch.Tensor, 
    sigma: torch.Tensor, 
    epsilon: torch.Tensor,
    cutoff: float
) -> torch.Tensor :
    """Compute Lennard-Jones energies using customized CUDA/C++ ops"""
    return torch.ops.torchff.compute_lennard_jones_energy(coords, pairs, box, sigma, epsilon, cutoff)


def compute_lennard_jones_energy_ref(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Reference Lennard-Jones implementation using native PyTorch ops"""
    # box in row major: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
    dr_vecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    box_inv = torch.linalg.inv(box)
    ds_vecs = torch.matmul(dr_vecs, box_inv)
    ds_vecs_pbc = ds_vecs - torch.floor(ds_vecs + 0.5)
    dr_vecs_pbc = torch.matmul(ds_vecs_pbc, box)
    dr = torch.norm(dr_vecs_pbc, dim=1)
    mask = dr <= cutoff
    sigma_ij = (sigma[pairs[:, 0]] + sigma[pairs[:, 1]]) / 2
    epsilon_ij = torch.sqrt(epsilon[pairs[:, 0]] * epsilon[pairs[:, 1]])
    tmp = (sigma_ij / dr) ** 6
    ene = 4 * epsilon_ij * tmp * (tmp - 1)
    return torch.sum(ene * mask)


class LennardJones(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        cutoff: float,
    ) -> torch.Tensor:
        if self.use_customized_ops:
            return compute_lennard_jones_energy(coords, pairs, box, sigma, epsilon, cutoff)
        else:
            return compute_lennard_jones_energy_ref(coords, pairs, box, sigma, epsilon, cutoff)