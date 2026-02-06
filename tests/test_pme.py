import math
import numpy as np
import pytest
import torch

from torchff.pme import PME


torch.set_printoptions(precision=8)


def create_test_data(num: int, rank: int = 2, device: str = "cuda", dtype: torch.dtype = torch.float64):
    """Create random test data for PME tests."""
    # Set a physically reasonable box length scaling with number of atoms
    boxLen = float((num * 10.0) ** (1.0 / 3.0))

    # Random coordinates in [0, boxLen)
    coords_np = np.random.rand(num, 3) * boxLen

    # Random charges, shifted so that the total charge is zero
    q_np = np.random.randn(num)
    q_np -= q_np.mean()

    # Random dipoles
    d_np = np.random.randn(num, 3)

    # Random symmetric, traceless quadrupoles per atom
    t_np = np.empty((num, 3, 3), dtype=float)
    for i in range(num):
        A = np.random.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace  # make traceless
        t_np[i] = sym

    # Cubic box
    box_np = np.eye(3) * boxLen

    # Convert to torch tensors with the requested device and dtype
    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.tensor(d_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 1
        else None
    )
    t = (
        torch.tensor(t_np, device=device, dtype=dtype, requires_grad=True)
        if rank >= 2
        else None
    )

    p = torch.zeros_like(p)
    t = torch.zeros_like(t)

    # Find appropriate PME parameters.
    alpha_pme = math.sqrt(-math.log10(2 * 1e-6)) / 9.0
    max_hkl = 10  # Reasonable default for PME grid

    return coords, box, q, p, t, alpha_pme, max_hkl


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [2])
def test_pme_execution(device, dtype, rank):
    """Test that PME with use_customized_ops=True can execute successfully."""
    N = 1000
    coords, box, q, p, t, alpha, max_hkl = create_test_data(N, rank, device=device, dtype=dtype)

    func = PME(alpha, max_hkl, rank, use_customized_ops=True).to(
        device=device, dtype=dtype
    )
    func_ref = PME(alpha, max_hkl, rank, use_customized_ops=False).to(
        device=device, dtype=dtype
    )

    # Test forward pass - should execute without errors
    result = func(coords, box, q, p, t)
    result_ref = func_ref(coords, box, q, p, t)
    print(result[3])
    print(result_ref[3])
    
    # # Verify that result is a tuple with expected components
    # assert isinstance(result, tuple), "PME should return a tuple"
    # assert len(result) == 5, "PME should return 5 components: (phi, E, EG, energy, forces)"
    
    # pot, field, EG, energy, forces = result
    # print(energy)
    
    # # Verify all components are tensors (or None for forces)
    # assert isinstance(pot, torch.Tensor), "pot should be a tensor"
    # assert isinstance(energy, torch.Tensor), "energy should be a tensor"
    
    # if rank >= 1:
    #     assert isinstance(field, torch.Tensor), "field should be a tensor for rank >= 1"
    # if rank >= 2:
    #     assert isinstance(EG, torch.Tensor), "EG should be a tensor for rank >= 2"
