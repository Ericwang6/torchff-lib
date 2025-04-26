
import torch
import torchff_coulomb


class CoulombFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, charges, pairs, box, epsilon, cutoff):
        outputs = torch.ops.torchff.compute_coulomb_energy(coords, charges, pairs, box, epsilon, cutoff)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        coord_grad, charge_grad, epsilon_grad = ctx.saved_tensors
        return (
            coord_grad * grad_outputs, 
            None, None,
            charge_grad * grad_outputs, 
            epsilon_grad * grad_outputs,
            None
        )


def compute_coulomb_energy(
        coords: torch.Tensor, charges: torch.Tensor, pairs: torch.Tensor, box: torch.Tensor, epsilon: torch.Tensor,
    cutoff: float
) -> torch.Tensor :
    """Compute coulomb energies"""
    return CoulombFunction.apply(coords, charges, pairs, box, epsilon, cutoff)

