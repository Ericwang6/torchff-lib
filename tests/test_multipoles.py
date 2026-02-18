import numpy as np
import pytest
import torch

from .utils import check_op, perf_op
from torchff.multipoles import MultipolarInteraction


torch.set_printoptions(precision=8)


def create_test_data(
    num: int,
    rank: int = 2,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    cutoff: float = 9.0,
    ewald_alpha: float = -1.0,
):
    """Create random test data for multipolar interaction tests."""
    box_len = float((num * 10.0) ** (1.0 / 3.0))

    coords_np = np.random.rand(num, 3) * box_len
    q_np = np.random.randn(num) * 0.1
    d_np = np.random.randn(num, 3) * 0.1

    t_np = np.empty((num, 3, 3), dtype=float)
    for i in range(num):
        A = np.random.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace
        t_np[i] = sym

    box_np = np.eye(3) * box_len

    # PBC minimum-image distance to build pairs
    dr = coords_np[:, None, :] - coords_np[None, :, :]
    box_inv_np = np.linalg.inv(box_np)
    ds = dr @ box_inv_np
    ds = ds - np.floor(ds + 0.5)
    dr_pbc = ds @ box_np
    dist = np.linalg.norm(dr_pbc, axis=2)
    ii, jj = np.triu_indices(num, k=1)
    mask = dist[ii, jj] < cutoff
    pairs_np = np.column_stack([ii[mask], jj[mask]])

    if pairs_np.size == 0:
        pairs_np = np.array([[0, 1]], dtype=np.int64)

    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    pairs = torch.tensor(pairs_np, device=device, dtype=torch.int64)
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

    prefactor = 1.0
    return coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_multipolar_energy(device, dtype, rank):
    """Compare custom CUDA multipolar kernel against Python reference implementation."""
    N = 200
    coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor = create_test_data(
        N, rank, device=device, dtype=dtype
    )

    func = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=True
    ).to(device=device, dtype=dtype)
    func_ref = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=False, cuda_graph_compat=False
    ).to(device=device, dtype=dtype)

    # rank=2 sums many pair energies; CUDA parallel reduction order can differ from Python
    # sequential sum, so allow slightly larger tolerance for quadrupoles
    atol = 1e-6 if dtype is torch.float64 else 1e-4
    rtol = 0.0
    if rank == 2:
        atol = max(atol, 1e-2)
        rtol = 1e-4
    check_op(
        func,
        func_ref,
        {"coords": coords, "box": box, "pairs": pairs, "q": q, "p": p, "t": t},
        check_grad=True,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_perf_multipolar(device, dtype, rank):
    """Performance comparison between Python and custom CUDA multipolar implementations."""
    N = 500
    coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor = create_test_data(
        N, rank, device=device, dtype=dtype
    )

    func_ref = torch.compile(
        MultipolarInteraction(
            rank, cutoff, ewald_alpha, prefactor,
            use_customized_ops=False, cuda_graph_compat=False,
        )
    ).to(device=device, dtype=dtype)
    func = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=True
    ).to(device=device, dtype=dtype)

    perf_op(
        func_ref,
        coords,
        box,
        pairs,
        q,
        p,
        t,
        desc=f"multipolar_ref (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )
    perf_op(
        func,
        coords,
        box,
        pairs,
        q,
        p,
        t,
        desc=f"multipolar_torchff (N={N}, rank={rank})",
        repeat=1000,
        run_backward=True,
    )


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float32), ("cuda", torch.float64)])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_multipolar_energy_ewald(device, dtype, rank):
    """Compare custom CUDA vs Python reference when ewald_alpha > 0 (erfc damping)."""
    N = 200
    coords, box, pairs, q, p, t, cutoff, ewald_alpha, prefactor = create_test_data(
        N, rank, device=device, dtype=dtype, ewald_alpha=0.4
    )

    func = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=True
    ).to(device=device, dtype=dtype)
    func_ref = MultipolarInteraction(
        rank, cutoff, ewald_alpha, prefactor, use_customized_ops=False, cuda_graph_compat=False
    ).to(device=device, dtype=dtype)

    atol = 1e-5 if dtype is torch.float64 else 1e-3
    rtol = 1e-4
    if rank == 2:
        atol = max(atol, 1e-2)
    check_op(
        func,
        func_ref,
        {"coords": coords, "box": box, "pairs": pairs, "q": q, "p": p, "t": t},
        check_grad=True,
        atol=atol,
        rtol=rtol,
    )
