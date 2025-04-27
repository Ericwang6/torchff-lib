import pytest
import time
import torch
import torch.nn as nn
import torchff

@torch.compile
class HamronicBond(nn.Module):

    def forward(self, coords, pairs, r0, k):
        r = torch.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], dim=1)
        ene = (r - r0) ** 2 * k / 2
        return torch.sum(ene)


@pytest.mark.parametrize("device, dtype", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    ('cuda', torch.float64), 
    ('cuda', torch.float32)
])
def test_harmonic_bond(device, dtype):
    requires_grad = True
    N = 100000
    Nbonds = 100000
    pairs = torch.randint(0, N-2, (Nbonds, 2), device=device, dtype=torch.int32)
    pairs[:, 1] = pairs[:, 0] + 1
    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)

    harmonic_bond = HamronicBond()
    ene_ref = harmonic_bond(coords, pairs, r0, k)
    ene = torchff.compute_harmonic_bond_energy(coords, pairs, r0, k)

    assert torch.allclose(ene_ref, ene), 'Energy not the same'

    ene_ref.backward()
    grads_ref = [coords.grad.clone().detach(), r0.grad.clone().detach(), k.grad.clone().detach()]

    coords.grad.zero_()
    r0.grad.zero_()
    k.grad.zero_()

    ene.backward()
    grads = [coords.grad.clone().detach(), r0.grad.clone().detach(), k.grad.clone().detach()]

    for c, g, gref in zip(['coord', 'b0', 'k'], grads, grads_ref):
        assert torch.allclose(g, gref, atol=1e-5), f'Gradient {c} not the same'

    # Test time
    Ntimes = 1000
    start = time.perf_counter()
    for _ in range(Ntimes):
        ene = torchff.compute_harmonic_bond_energy(coords, pairs, r0, k)
        ene.backward()
    end = time.perf_counter()
    print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")


    start = time.perf_counter()
    for _ in range(Ntimes):
        ene_ref = harmonic_bond(coords, pairs, r0, k)
        ene_ref.backward()
    end = time.perf_counter()
    print(f"torch time: {(end-start)/Ntimes*1000:.5f} ms")
