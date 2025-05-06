import pytest
import random
import math
import torch
import torch.nn as nn
import torchff

from .utils import perf_op, check_op

@torch.compile
class PeriodicTorsion(nn.Module):

    def forward(self, coords, torsions, fc, per, phase):
        b1 = coords[torsions[:, 1]] - coords[torsions[:, 0]]
        b2 = coords[torsions[:, 2]] - coords[torsions[:, 1]]
        b3 = coords[torsions[:, 3]] - coords[torsions[:, 2]]
        n1 = torch.cross(b1, b2, dim=1)
        n2 = torch.cross(b2, b3, dim=1)
        norm_n1 = torch.norm(n1, dim=1)
        norm_n2 = torch.norm(n2, dim=1)
        cosval = torch.clamp(
            torch.sum(n1 * n2, dim=1) / (norm_n1 * norm_n2), 
            -0.999999999, 0.999999999
        )
        phi = torch.acos(cosval) * torch.sign(torch.sum(n1 * b3, axis=1))
        ene = fc * (1 + torch.cos(per * phi - phase))
        return torch.sum(ene)


@pytest.mark.parametrize("device, dtype", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    ('cuda', torch.float32),
    ('cuda', torch.float64), 
])
def test_periodic_torsion(device, dtype):
    requires_grad = True
    N = 10000
    Ntors = 10000
    pairs = torch.randint(0, max(1, N-4), (Ntors, 4), device=device, dtype=torch.int32)
    pairs[:, 1] = pairs[:, 0] + 1
    pairs[:, 2] = pairs[:, 1] + 1
    pairs[:, 3] = pairs[:, 2] + 1

    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    fc = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)
    phase = torch.tensor([random.randint(0, 1) * math.pi for _ in range(Ntors)], dtype=dtype, requires_grad=requires_grad, device=device)
    per = torch.tensor([random.randint(1, 6) for _ in range(Ntors)], dtype=torch.int32, requires_grad=False, device=device)

    periodic_torsion_ref = PeriodicTorsion()

    perf_op(
        periodic_torsion_ref, 
        coords, pairs, fc, per, phase, 
        desc='ref-periodic-torsion', 
        run_backward=True, use_cuda_graph=True
    )
    perf_op(
        torchff.compute_periodic_torsion_energy, 
        coords, pairs, fc, per, phase, 
        desc='torchff-periodic-torsion', 
        run_backward=True, use_cuda_graph=True
    )
    check_op(
        torchff.compute_periodic_torsion_energy,
        periodic_torsion_ref,
        coords, pairs, fc, per, phase, 
        check_grad=True,
        atol=1e-2 if dtype is torch.float32 else 5e-4
    )
    
    forces = torch.zeros_like(coords, requires_grad=False)
    torchff.compute_periodic_torsion_forces(coords, pairs, fc, per, phase, forces)
    coords.grad = None
    e = periodic_torsion_ref(coords, pairs, fc, per, phase)
    e.backward()
    assert torch.allclose(
        forces, 
        -coords.grad.clone().detach(), 
        atol=1e-2 if dtype is torch.float32 else 5e-4
    ), 'Force not the same'

    perf_op(
        torchff.compute_periodic_torsion_forces, 
        coords, pairs, fc, per, phase, forces,
        desc='torchff-periodic-torsion-forces', 
        run_backward=False, use_cuda_graph=True
    )
    