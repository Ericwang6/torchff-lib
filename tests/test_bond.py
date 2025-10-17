import pytest
import random
import torch
import torch.nn as nn
import torchff

from .utils import perf_op, check_op


@torch.compile
class HamronicBond(nn.Module):
    def forward(self, coords, bonds, b0, k):
        r = torch.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], dim=1)
        ene = (r - b0) ** 2 * k / 2
        return torch.sum(ene)
harmonic_bond_ref = HamronicBond()


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32), 
    ('cuda', torch.float64), 
])
def test_harmonic_bond(device, dtype):
    requires_grad = True
    N = 100
    Nbonds = N * 2
    arange = list(range(N))
    pairs = torch.tensor([random.sample(arange, 2) for _ in range(Nbonds)], device=device, dtype=torch.int32)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)
    check_op(
        torchff.compute_harmonic_bond_energy,
        harmonic_bond_ref,
        {'coords': coords, 'bonds': pairs, 'b0': r0, 'k': k},
        check_grad=True
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    torchff.compute_harmonic_bond_forces(coords, pairs, r0, k, forces)
    coords.grad = None
    e = harmonic_bond_ref(coords, pairs, r0, k)
    e.backward()
    assert torch.allclose(forces, -coords.grad.clone().detach(), atol=1e-5)



@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32), 
    ('cuda', torch.float64), 
])
def test_perf_harmonic_bond(device, dtype):
    requires_grad = True
    N = 10000
    Nbonds = N * 2
    arange = list(range(N))
    pairs = torch.tensor([random.sample(arange, 2) for _ in range(Nbonds)], device=device, dtype=torch.int32)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=False)
    k = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=False)
    perf_op(
        harmonic_bond_ref, 
        coords, pairs, r0, k, 
        desc='ref-harmonic-bond', 
        run_backward=True, use_cuda_graph=True
    )
    perf_op(
        torchff.compute_harmonic_bond_energy, 
        coords, pairs, r0, k, 
        desc='torchff-harmonic-bond', 
        run_backward=True, use_cuda_graph=True
    )
