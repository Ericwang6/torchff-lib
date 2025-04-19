import torch
import torchff_vdw


class LennardJonesFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, pairs, box, sigma, epsilon, cutoff):
        outputs = torch.ops.torchff.compute_lennard_jones(coords, pairs, box, sigma, epsilon, cutoff)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        coord_grad, sigma_grad, epsilon_grad = ctx.saved_tensors
        return (
            coord_grad * grad_outputs, 
            None, None,
            sigma_grad * grad_outputs, 
            epsilon_grad * grad_outputs,
            None
        )


def compute_lennard_jones_energy(
    coords: torch.Tensor, pairs: torch.Tensor, box: torch.Tensor, 
    sigma: torch.Tensor, epsilon: torch.Tensor,
    cutoff: float
) -> torch.Tensor :
    """Compute periodic torsion energies"""
    return LennardJonesFunction.apply(coords, pairs, box, sigma, epsilon, cutoff)