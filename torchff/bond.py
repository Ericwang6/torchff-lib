import torch
import torchff_harmonic_bond


class HarmonicBondFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, pairs, b0, k):
        ctx.save_for_backward(coords, pairs, b0, k)
        ene = torch.ops.torchff.compute_harmonic_bond_energy(coords, pairs, b0, k)
        return ene

    @staticmethod
    def backward(ctx, grad_outputs):
        coords, pairs, b0, k = ctx.saved_tensors
        grad_outputs = torch.ops.torchff.compute_harmonic_bond_energy_grad(coords, pairs, b0, k)
        return grad_outputs[0], None, grad_outputs[1], grad_outputs[2]



def compute_harmonic_bond_energy(coords: torch.Tensor, pairs: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """Compute harmonic bond energies"""
    return HarmonicBondFunction.apply(coords, pairs, b0, k)