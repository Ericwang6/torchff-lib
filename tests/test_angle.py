import pytest
import time
import torch
import torch.nn as nn
import torchff

from .utils import perf_op, check_op


@torch.compile
class HarmonicAngle(nn.Module):
    def forward(self, coords, angles, theta0, k):
        v1 = coords[angles[:, 0]] - coords[angles[:, 1]]
        v2 = coords[angles[:, 2]] - coords[angles[:, 1]]
        dot_product = torch.sum(v1 * v2, dim=1)
        mag_v1 = torch.norm(v1, dim=1)
        mag_v2 = torch.norm(v2, dim=1)
        cos_theta = dot_product / (mag_v1 * mag_v2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical issues
        theta = torch.acos(cos_theta)
        ene = (theta - theta0) ** 2 * k / 2
        return torch.sum(ene)


@pytest.mark.parametrize("device, dtype", [
    #('cpu', torch.float64), 
    #('cpu', torch.float32), 
    ('cuda', torch.float32),
    ('cuda', torch.float64)
])
def test_harmonic_angle(device, dtype):
    requires_grad = True
    N = 1000
    Nangles = 10000
    angles = torch.randint(0, N-3, (Nangles, 3), device=device, dtype=torch.int32)
    angles[:, 1] = angles[:, 0] + 1
    angles[:, 2] = angles[:, 1] + 1
    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    theta0 = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)

    harmonic_angle_ref = HarmonicAngle()
    perf_op(
        harmonic_angle_ref, 
        coords, angles, theta0, k, 
        desc='ref-harmonic-angle', 
        run_backward=True, use_cuda_graph=True
    )
    perf_op(
        torchff.compute_harmonic_angle_energy, 
        coords, angles, theta0, k, 
        desc='torchff-harmonic-angle', 
        run_backward=True, use_cuda_graph=True
    )
    check_op(
        torchff.compute_harmonic_angle_energy,
        harmonic_angle_ref,
        coords, angles, theta0, k, 
        check_grad=True,
        atol=1e-2 if dtype is torch.float32 else 1e-5
    )
    
    forces = torch.zeros_like(coords, requires_grad=False)
    torchff.compute_harmonic_angle_forces(coords, angles, theta0, k, forces)
    coords.grad = None
    e = harmonic_angle_ref(coords, angles, theta0, k)
    e.backward()
    assert torch.allclose(
        forces, 
        -coords.grad.clone().detach(), 
        atol=1e-2 if dtype is torch.float32 else 1e-5
    )

    perf_op(
        torchff.compute_harmonic_bond_forces, 
        coords, angles, theta0, k, forces,
        desc='torchff-harmonic-angle-forces', 
        run_backward=False, use_cuda_graph=True
    )