import pytest
import random
import math
import torch
from torchff.torsion import *

from .utils import perf_op, check_op


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64), 
])
def test_periodic_torsion(device, dtype):
    requires_grad = True
    N = 100
    Ntors = N * 2
    arange = list(range(N))
    pairs = torch.tensor([random.sample(arange, 4) for _ in range(Ntors)], device=device)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    fc = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=requires_grad)
    phase = torch.tensor([random.randint(0, 1) * math.pi for _ in range(Ntors)], dtype=dtype, requires_grad=requires_grad, device=device)
    periodicity = torch.tensor([random.randint(1, 6) for _ in range(Ntors)], dtype=torch.int64, requires_grad=False, device=device)

    func = PeriodicTorsion(use_customized_ops=True)
    func_ref = PeriodicTorsion(use_customized_ops=False)

    check_op(
        func,
        func_ref,
        {'coords': coords, 'torsions': pairs, 'fc': fc, 'periodicity': periodicity, 'phase': phase},
        check_grad=True,
        atol=1e-2 if dtype is torch.float32 else 5e-4
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    compute_periodic_torsion_forces(coords, pairs, fc, periodicity, phase, forces)
    coords.grad = None
    e = func_ref(coords, pairs, fc, periodicity, phase)
    e.backward()
    assert torch.allclose(
        forces, 
        -coords.grad.clone().detach(), 
        atol=1e-2 if dtype is torch.float32 else 5e-4
    )


@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64), 
])
def test_perf_periodic_torsion(device, dtype):
    requires_grad = True
    N = 10000
    Ntors = N * 2
    arange = list(range(N))
    pairs = torch.tensor([random.sample(arange, 4) for _ in range(Ntors)], device=device)

    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    fc = torch.rand(Ntors, device=device, dtype=dtype, requires_grad=False)
    phase = torch.tensor([random.randint(0, 1) * math.pi for _ in range(Ntors)], dtype=dtype, requires_grad=False, device=device)
    periodicity = torch.tensor([random.randint(1, 6) for _ in range(Ntors)], dtype=torch.int64, requires_grad=False, device=device)

    func = PeriodicTorsion(use_customized_ops=True)
    func_ref = torch.compile(PeriodicTorsion(use_customized_ops=False))

    perf_op(
        func_ref, 
        coords, pairs, fc, periodicity, phase, 
        desc='ref-periodic-torsion', 
        run_backward=True, use_cuda_graph=True
    )
    perf_op(
        func, 
        coords, pairs, fc, periodicity, phase, 
        desc='torchff-periodic-torsion', 
        run_backward=True, use_cuda_graph=True
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    perf_op(
        compute_periodic_torsion_forces, 
        coords, pairs, fc, periodicity, phase, forces,
        desc='torchff-periodic-torsion-forces', 
        run_backward=False, use_cuda_graph=True
    )
    