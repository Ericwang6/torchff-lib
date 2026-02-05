
import torch
import torch.nn as nn
import torchff_coulomb


def compute_coulomb_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    charges: torch.Tensor,
    coulomb_constant: float,
    cutoff: float,
    do_shift: bool = True,
) -> torch.Tensor:
    """Compute Coulomb energies using customized CUDA/C++ ops"""
    return torch.ops.torchff.compute_coulomb_energy(
        coords, pairs, box, charges, coulomb_constant, cutoff, do_shift
    )


def compute_coulomb_energy_ref(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    charges: torch.Tensor,
    coulomb_constant: float,
    cutoff: float,
    do_shift: bool = True,
) -> torch.Tensor:
    """Reference Coulomb implementation using native PyTorch ops"""
    # box in row major: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
    dr_vecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    box_inv = torch.linalg.inv(box)
    ds_vecs = torch.matmul(dr_vecs, box_inv)
    ds_vecs_pbc = ds_vecs - torch.floor(ds_vecs + 0.5)
    dr_vecs_pbc = torch.matmul(ds_vecs_pbc, box)
    dr = torch.norm(dr_vecs_pbc, dim=1)
    mask = dr <= cutoff
    rinv = 1.0 / dr
    if do_shift:
        rinv = rinv - 1.0 / cutoff
    ene = charges[pairs[:, 0]] * charges[pairs[:, 1]] * rinv
    return torch.sum(ene * mask) * coulomb_constant


class Coulomb(nn.Module):
    def __init__(self, use_customized_ops: bool = False):
        super().__init__()
        self.use_customized_ops = use_customized_ops

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        charges: torch.Tensor,
        coulomb_constant: float,
        cutoff: float,
        do_shift: bool = True,
    ) -> torch.Tensor:
        if self.use_customized_ops:
            return compute_coulomb_energy(
                coords, pairs, box, charges, coulomb_constant, cutoff, do_shift
            )
        else:
            return compute_coulomb_energy_ref(
                coords, pairs, box, charges, coulomb_constant, cutoff, do_shift
            )
