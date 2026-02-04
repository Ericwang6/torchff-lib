import pytest
import random
import torch
from torchff.bond import *

from .utils import perf_op, check_op


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
    pairs = torch.tensor([random.sample(arange, 2) for _ in range(Nbonds)], device=device)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nbonds, device=device, dtype=dtype, requires_grad=requires_grad)

    func = HarmonicBond(use_customized_ops=True)
    func_ref = HarmonicBond(use_customized_ops=False)

    check_op(
        func,
        func_ref,
        {'coords': coords, 'bonds': pairs, 'b0': r0, 'k': k},
        check_grad=True
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    compute_harmonic_bond_forces(coords, pairs, r0, k, forces)
    coords.grad = None
    e = func_ref(coords, pairs, r0, k)
    e.backward()
    assert torch.allclose(forces, -coords.grad.clone().detach(), atol=1e-5)



@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32), 
    ('cuda', torch.float64), 
])
def test_perf_harmonic_bond(device, dtype):
    requires_grad = True
    N = 10000
    pairs = []
    for i in range(N):
        pairs.append([i*3, i*3+1])
        pairs.append([i*3, i*3+2])
    pairs = torch.tensor(pairs, device=device)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    r0 = torch.rand(pairs.shape[0], device=device, dtype=dtype, requires_grad=False)
    k = torch.rand(pairs.shape[0], device=device, dtype=dtype, requires_grad=False)

    func = HarmonicBond(use_customized_ops=True)
    func_ref = torch.compile(HarmonicBond(use_customized_ops=False))

    perf_op(
        func_ref, 
        coords, pairs, r0, k, 
        desc='ref-harmonic-bond', 
        run_backward=True, use_cuda_graph=True
    )
    perf_op(
        func, 
        coords, pairs, r0, k, 
        desc='torchff-harmonic-bond', 
        run_backward=True, use_cuda_graph=True
    )
