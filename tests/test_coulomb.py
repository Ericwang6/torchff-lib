import pytest
import time
import numpy as np
import torch
import torch.nn as nn
import torchff


@torch.compile
class Coulomb(nn.Module):

    def forward(self, coords, pairs, box, charges, prefac, cutoff, do_shift=True):
        # box in row major: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
        drVecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        boxInv = torch.linalg.inv(box)
        dsVecs = torch.matmul(drVecs, boxInv)
        dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
        drVecsPBC = torch.matmul(dsVecsPBC, box)
        dr = torch.norm(drVecsPBC, dim=1)
        mask = dr <= cutoff
        rinv = 1 / dr
        if do_shift:
            rinv -= 1 / cutoff
        ene = charges[pairs[:, 0]] * charges[pairs[:, 1]]* rinv
        return torch.sum(ene * mask) * prefac


@pytest.mark.parametrize("device, dtype, requires_grad", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    ('cuda', torch.float32, True),
    ('cuda', torch.float64, True), 
])
def test_coulomb(device, dtype, requires_grad):
    cutoff = 4.0

    box = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ], requires_grad=False, dtype=dtype, device=device)
    
    N = 100000
    coords = (np.random.rand(N, 3) * 10.0).tolist()
    coords = torch.tensor(coords, requires_grad=requires_grad, device=device, dtype=dtype)

    charges = (np.random.rand(N) + 0.001).tolist()
    charges = torch.tensor(charges, requires_grad=requires_grad, device=device, dtype=dtype)
    # coords = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], requires_grad=requires_grad, device=device, dtype=dtype)

    Npairs = 100000
    pairs = torch.randint(0, N-2, (Npairs, 2), device=device, dtype=torch.int32)
    pairs[:, 1] = pairs[:, 0] + 1
    # pairs = torch.tensor([[0, 1]], dtype=torch.long, device=device)

    prefac = torch.tensor(20.0, device=device, requires_grad=requires_grad, dtype=dtype)

    coul = Coulomb()
    ene_ref = coul(coords, pairs, box, charges, prefac, cutoff)
    ene = torchff.compute_coulomb_energy(coords, pairs, box, charges, prefac, cutoff)

    assert torch.allclose(ene_ref, ene), 'Energy not the same'

    if requires_grad:
        ene_ref.backward()
        grads_ref = [coords.grad.clone().detach(), charges.grad.clone().detach() , prefac.grad.clone().detach()]

        coords.grad.zero_()
        charges.grad.zero_()
        prefac.grad.zero_()

        ene.backward()
        grads = [coords.grad.clone().detach(), charges.grad.clone().detach(), prefac.grad.clone().detach()]

        for c, g, gref in zip(['coord', 'charge', 'prefac'], grads, grads_ref):
            assert torch.allclose(g, gref, atol=1e-5), f'Gradient {c} not the same (max deviation {torch.max(torch.abs(g - gref))})'

    # Test times
    Ntimes = 1000
    
    if requires_grad:
        start = time.time()
        for _ in range(Ntimes):
            ene = torchff.compute_coulomb_energy(coords, pairs, box, charges, prefac, cutoff)
            ene.backward()
        end = time.time()
        print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")

        start = time.time()
        for _ in range(Ntimes):
            ene_ref = coul(coords, pairs, box, charges, prefac, cutoff)
            ene_ref.backward()
        end = time.time()
        print(f"torch time: {(end-start)/Ntimes*1000:.5f} ms")
    else:
        start = time.time()
        for _ in range(Ntimes):
            ene = torchff.compute_coulomb_energy(coords, pairs, box, charges, prefac, cutoff)
        end = time.time()
        print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")

        start = time.time()
        for _ in range(Ntimes):
            ene_ref = coul(coords, pairs, box, charges, prefac, cutoff)
        end = time.time()
        print(f"torch time: {(end-start)/Ntimes*1000:.5f} ms")
