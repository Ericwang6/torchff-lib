import torch
import torchff_multipoles


def compute_multipolar_energy_from_atom_pairs(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    # box: torch.Tensor,
    multipoles: torch.Tensor
):
    '''
    Compute nonbonded interaction energies (fixed charge Coulomb and Lennard-Jones)
    '''
    return torch.ops.torchff.compute_multipolar_energy_from_atom_pairs(
        coords, pairs, multipoles
    )