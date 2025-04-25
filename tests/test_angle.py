import pytest
import time
import torch
import torch.nn as nn
import torchff

def harmonic_angle(coords, angles, theta0, k):
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
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    ('cuda', torch.float64), 
    ('cuda', torch.float32)
])
def test_harmonic_angle(device, dtype):
    requires_grad = True
    N = 100000
    Nangles = 100000
    angles = torch.randint(0, N-3, (Nangles, 3), device=device)
    angles[:, 1] = angles[:, 0] + 1
    angles[:, 2] = angles[:, 1] + 1
    coords = torch.rand(N, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    theta0 = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)

    ene_ref = harmonic_angle(coords, angles, theta0, k)
    ene = torchff.compute_harmonic_angle_energy(coords, angles, theta0, k)

    assert torch.allclose(ene_ref, ene, atol=1e-5), 'Energy not the same'

    ene_ref.backward()
    grads_ref = [coords.grad.clone().detach()]

    coords.grad.zero_()

    ene.backward()
    grads = [coords.grad.clone().detach()]

    grad_atol = 1e-5 if dtype is torch.float64 else 1e-2
    for c, g, gref in zip(['coord', 'theta0', 'k'], grads, grads_ref):
        assert torch.allclose(g, gref, atol=grad_atol), f'Gradient {c} not the same'

    # Test time
    Ntimes = 1000
    start = time.time()
    for _ in range(Ntimes):
        ene = torchff.compute_harmonic_angle_energy(coords, angles, theta0, k)
        ene.backward()
    end = time.time()
    print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")

    start = time.time()
    for _ in range(Ntimes):
        ene_ref = harmonic_angle(coords, angles, theta0, k)
        ene_ref.backward()
    end = time.time()
    print(f"torch time: {(end-start)/Ntimes*1000:.5f} ms")
