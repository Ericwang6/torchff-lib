"""
B-spline utilities for PME, matching the logic in :file:`csrc/pme/bsplines.cuh`.
"""

import torch

TWOPI = 2.0 * 3.14159265358979323846


def get_bspline_coeff_order6(i: int) -> float:
    """
    B-spline coefficient for order-6 PME at index i.

    Matches :func:`get_bspline_coeff_order6` in :file:`csrc/pme/bsplines.cuh`.
    """
    coeffs = {1: 1.0 / 120.0, 2: 26.0 / 120.0, 3: 66.0 / 120.0, 4: 26.0 / 120.0, 5: 1.0 / 120.0}
    return coeffs.get(i, 0.0)


def get_bspline_modulus(k: torch.Tensor, K: int, order: int = 6) -> torch.Tensor:
    """
    B-spline modulus for PME reciprocal-space scaling.

    For each wave vector index k, computes
    sum_m b(m) * cos(2*pi*m*k/K) where m runs over the order-6 B-spline range
    and b(m) are the B-spline coefficients. Matches
    :func:`get_bspline_modulus_device` in :file:`csrc/pme/bsplines.cuh`.

    Parameters
    ----------
    k : torch.Tensor
        1D tensor of integer indices (e.g. 0 .. K-1 or 0 .. K//2).
    K : int
        Grid size along this dimension.
    order : int
        B-spline order; only 6 is implemented.

    Returns
    -------
    torch.Tensor
        Modulus values, same shape and device as k, dtype float.
    """
    if order != 6:
        raise ValueError("Only order=6 is implemented.")
    half = order // 2
    k_f = k.double() if k.dtype != torch.float64 else k
    if k_f.dtype != torch.float64:
        k_f = k_f.double()
    sum_val = torch.zeros(k_f.shape, dtype=torch.float64, device=k.device)
    for m in range(-half, half):
        b_val = get_bspline_coeff_order6(m + half)
        arg = (TWOPI * m * k_f) / K
        sum_val += b_val * torch.cos(arg)
    return sum_val


def compute_bspline_moduli_1d(K: int, dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:
    """
    Precompute the 1D B-spline modulus array for grid size K.

    Returns moduli for indices 0, 1, ..., K-1. For use as xmoduli or ymoduli
    in PME (length K).

    Parameters
    ----------
    K : int
        Grid size along this dimension.
    dtype : torch.dtype
        Output tensor dtype.
    device : torch.device, optional
        Output device; default is CPU.

    Returns
    -------
    torch.Tensor
        Shape (K,), dtype and device as requested.
    """
    k = torch.arange(K, dtype=torch.int64, device=device)
    out = get_bspline_modulus(k, K, order=6)
    return out.to(dtype=dtype, device=device)


def compute_bspline_moduli_z(K3: int, dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:
    """
    Precompute the z-dimension B-spline modulus array for real FFT.

    Returns moduli for z indices 0, 1, ..., K3//2 (length K3//2 + 1),
    as used in the conjugate-even (rfft) reciprocal grid.

    Parameters
    ----------
    K3 : int
        Grid size along the z dimension.
    dtype : torch.dtype
        Output tensor dtype.
    device : torch.device, optional
        Output device; default is CPU.

    Returns
    -------
    torch.Tensor
        Shape (K3//2 + 1,), dtype and device as requested.
    """
    K3_complex = K3 // 2 + 1
    k = torch.arange(K3_complex, dtype=torch.int64, device=device)
    out = get_bspline_modulus(k, K3, order=6)
    return out.to(dtype=dtype, device=device)
