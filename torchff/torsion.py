import torch
import torchff_periodic_torsion


class PeriodicTorsionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, torsions, fc, per, phase):
        outputs = torch.ops.torchff.compute_periodic_torsion(coords, torsions, fc, per, phase)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        coord_grad, fc_grad, phase_grad = ctx.saved_tensors
        return coord_grad * grad_outputs, None, fc_grad * grad_outputs, None, phase_grad * grad_outputs


def compute_periodic_torsion_energy(coords: torch.Tensor, torsions: torch.Tensor, fc: torch.Tensor, per: torch.Tensor, phase: torch.Tensor) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return PeriodicTorsionFunction.apply(coords, torsions, fc, per, phase)