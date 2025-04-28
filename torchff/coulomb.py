
import torch
import torchff_coulomb


def compute_coulomb_energy(
    coords: torch.Tensor, 
    pairs: torch.Tensor, 
    box: torch.Tensor, 
    charges: torch.Tensor, 
    prefac: torch.Tensor,
    cutoff: float,
    do_shift: bool = True
) -> torch.Tensor :
    """Compute coulomb energies"""
    return torch.ops.torchff.compute_coulomb_energy(coords, pairs, box, charges, prefac, cutoff, do_shift)

