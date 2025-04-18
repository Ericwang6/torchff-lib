import torch
import torchff_harmonic_angle

class HarmonicAngleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, triplets, theta0, k):
        ctx.save_for_backward(coords, triplets, theta0, k)
        ene = torch.ops.torchff.compute_harmonic_angle_energy(coords, triplets, theta0, k)
        return ene
        #return torch.Tensor(0);

    @staticmethod
    def backward(ctx, grad_outputs):
        coords, triplets, theta0, k = ctx.saved_tensors
        grad_outputs = torch.ops.torchff.compute_harmonic_angle_energy_grad(coords, triplets, theta0, k)
        return grad_outputs[0], None, grad_outputs[1], grad_outputs[2]
        #return torch.Tensor(0);

def compute_harmonic_angle_energy(coords: torch.Tensor, triplets: torch.Tensor, theta0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute harmonic angle energies"""
    return HarmonicAngleFunction.apply(coords, triplets, theta0, k)
