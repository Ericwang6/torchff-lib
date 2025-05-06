import pytest
import time
import torch
import torch.nn as nn
import torchff

from .utils import perf_op, check_op

@torch.compile
class HamronicBond(nn.Module):
    def forward(self, coords, pairs, r0, k):
        r = torch.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], dim=1)
        ene = (r - r0) ** 2 * k / 2
        return torch.sum(ene)


@pytest.mark.parametrize("device, dtype", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32),
    ('cuda', torch.float32), 
    ('cuda', torch.float64), 
])
def test_harmonic_bond(device, dtype):
    requires_grad = True
    N = 10000
    Nbonds = 10000
    pairs = torch.randint(0, N-2, (Nbonds, 2), device=device, dtype=torch.int32)
    pairs[:, 1] = pairs[:, 0] + 1
    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)

    harmonic_bond_ref = HamronicBond()
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
    check_op(
        torchff.compute_harmonic_bond_energy,
        harmonic_bond_ref,
        coords, pairs, r0, k,
        check_grad=True,
        atol=1e-2 if dtype is torch.float32 else 1e-5
    )
    
    forces = torch.zeros_like(coords, requires_grad=False)
    torchff.compute_harmonic_bond_forces(coords, pairs, r0, k, forces)
    coords.grad = None
    e = harmonic_bond_ref(coords, pairs, r0, k)
    e.backward()
    assert torch.allclose(
        forces, 
        -coords.grad.clone().detach(), 
        atol=1e-2 if dtype is torch.float32 else 1e-5
    )

    perf_op(
        torchff.compute_harmonic_bond_forces, 
        coords, pairs, r0, k, forces,
        desc='torchff-harmonic-bond-forces', 
        run_backward=False, use_cuda_graph=True
    )
