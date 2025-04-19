import pytest
import time
import random
import math
import torch
import torch.nn as nn
import torchff

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
            -1.0, 1.0
        )
        phi = torch.acos(cosval) * torch.sign(torch.sum(n1 * b3, axis=1))
        ene = fc * (1 + torch.cos(per * phi - phase))
        return torch.sum(ene)


@pytest.mark.parametrize("device, dtype", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    ('cuda', torch.float64), 
    # ('cuda', torch.float32)
])
def test_periodic_torsion(device, dtype):
    requires_grad = True
    N = 10000
    Ntors = 10000
    pairs = torch.randint(0, max(1, N-4), (Ntors, 4), device=device)
    pairs[:, 1] = pairs[:, 0] + 1
    pairs[:, 2] = pairs[:, 1] + 1
    pairs[:, 3] = pairs[:, 2] + 1

    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    fc = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)
    phase = torch.tensor([random.randint(0, 1) * math.pi for _ in range(Ntors)], dtype=dtype, requires_grad=requires_grad, device=device)
    per = torch.tensor([random.randint(1, 6) for _ in range(Ntors)], dtype=torch.long, requires_grad=False, device=device)

    periodic_torsion = PeriodicTorsion()
    ene_ref = periodic_torsion(coords, pairs, fc, per, phase)
    ene = torchff.compute_periodic_torsion_energy(coords, pairs, fc, per, phase)

    assert torch.allclose(ene_ref, ene), 'Energy not the same'

    ene_ref.backward()
    grads_ref = [coords.grad.clone().detach(), fc.grad.clone().detach(), phase.grad.clone().detach()]

    coords.grad.zero_()
    fc.grad.zero_()
    phase.grad.zero_()

    ene.backward()
    grads = [coords.grad.clone().detach(), fc.grad.clone().detach(), phase.grad.clone().detach()]

    for c, g, gref in zip(['coord', 'fc', 'phase'], grads, grads_ref):
        assert torch.allclose(gref, g, atol=1e-5), f'Gradient {c} not the same'

    # Test time
    Ntimes = 1000
    start = time.time()
    for _ in range(Ntimes):
        ene = torchff.compute_periodic_torsion_energy(coords, pairs, fc, per, phase)
        ene.backward()
    end = time.time()
    print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")

    start = time.time()
    for _ in range(Ntimes):
        ene_ref = periodic_torsion(coords, pairs, fc, per, phase)
        ene_ref.backward()
    end = time.time()
    print(f"torch time: {(end-start)/Ntimes*1000:.5f} ms")