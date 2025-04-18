import torch
import torchff_periodic_torsion


class PeriodicTorsionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, torsions, fc, per, phase):
        outputs = torch.ops.torchff.compute_harmonic_bond_energy(coords, torsions, fc, per, phase)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        coord_grad, fc_grad, phase_grad = ctx.saved_tensors
        return coord_grad, fc_grad, None, phase_grad


def compute_harmonic_bond_energy(coords: torch.Tensor, pairs: torch.Tensor, b0: torch.Tensor, k: torch.Tensor) -> torch.Tensor :
    """Compute harmonic bond energies"""
    return PeriodicTorsionFunction.apply(coords, pairs, b0, k)