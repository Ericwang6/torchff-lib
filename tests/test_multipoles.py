import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist
import pytest
from .utils import perf_op, check_op
from .get_reference import get_water_data

import torch
from torchff.multipoles import compute_multipolar_energy_from_atom_pairs


def computeInteractionTensor(drVec: torch.Tensor, dampFactors: torch.Tensor, drInv: Optional[torch.Tensor] = None, rank: int = 2):
    """
    drVec: N x 3
    mPoles: N x 10
    dampFactors: 5 x N

    eData: N x 
    """
    if drInv is None:
        drInv = 1 / torch.norm(drVec, dim=1)
    
    # calculate inversions
    drInv2 = torch.pow(drInv, 2)
    drInv3 = drInv2 * drInv
    drInv5 = drInv3 * drInv2

    drVec2 = torch.pow(drVec, 2)
    x, y, z = drVec[:, 0], drVec[:, 1], drVec[:, 2]
    x2, y2, z2 = drVec2[:, 0], drVec2[:, 1], drVec2[:, 2]
    xy, xz, yz = x * y, x * z, y * z

    drInv7 = drInv5 * drInv2
    drInv9 = drInv7 * drInv2

    if len(dampFactors):
        drInv = drInv * dampFactors[0]
        if rank > 0:
            drInv3 = drInv3 * dampFactors[1]
            drInv5 = drInv5 * dampFactors[2]
        if rank > 1:
            drInv7 = drInv7 * dampFactors[3]
            drInv9 = drInv9 * dampFactors[4]

    tx, ty, tz = -x * drInv3, -y * drInv3, -z * drInv3
    
    txx = 3 * x2 * drInv5 - drInv3
    txy = 3 * xy * drInv5
    txz = 3 * xz * drInv5
    tyy = 3 * y2 * drInv5 - drInv3
    tyz = 3 * yz * drInv5
    tzz = 3 * z2 * drInv5 - drInv3     

    txxx = -15 * x2 * x * drInv7 + 9 * x * drInv5
    txxy = -15 * x2 * y * drInv7 + 3 * y * drInv5
    txxz = -15 * x2 * z * drInv7 + 3 * z * drInv5
    tyyy = -15 * y2 * y * drInv7 + 9 * y * drInv5
    tyyx = -15 * y2 * x * drInv7 + 3 * x * drInv5
    tyyz = -15 * y2 * z * drInv7 + 3 * z * drInv5
    tzzz = -15 * z2 * z * drInv7 + 9 * z * drInv5
    tzzx = -15 * z2 * x * drInv7 + 3 * x * drInv5
    tzzy = -15 * z2 * y * drInv7 + 3 * y * drInv5
    txyz = -15 * x * y * z * drInv7

    txxxx = 105 * x2 * x2 * drInv9 - 90 * x2 * drInv7 + 9 * drInv5
    txxxy = 105 * x2 * xy * drInv9 - 45 * xy * drInv7
    txxxz = 105 * x2 * xz * drInv9 - 45 * xz * drInv7
    txxyy = 105 * x2 * y2 * drInv9 - 15 * (x2 + y2) * drInv7 + 3 * drInv5
    txxzz = 105 * x2 * z2 * drInv9 - 15 * (x2 + z2) * drInv7 + 3 * drInv5
    txxyz = 105 * x2 * yz * drInv9 - 15 * yz * drInv7

    tyyyy = 105 * y2 * y2 * drInv9 - 90 * y2 * drInv7 + 9 * drInv5
    tyyyx = 105 * y2 * xy * drInv9 - 45 * xy * drInv7
    tyyyz = 105 * y2 * yz * drInv9 - 45 * yz * drInv7
    tyyzz = 105 * y2 * z2 * drInv9 - 15 * (y2 + z2) * drInv7 + 3 * drInv5
    tyyxz = 105 * y2 * xz * drInv9 - 15 * xz * drInv7

    tzzzz = 105 * z2 * z2 * drInv9 - 90 * z2 * drInv7 + 9 * drInv5
    tzzzx = 105 * z2 * xz * drInv9 - 45 * xz * drInv7
    tzzzy = 105 * z2 * yz * drInv9 - 45 * yz * drInv7                
    tzzxy = 105 * z2 * xy * drInv9 - 15 * xy * drInv7

    
    if rank == 0:
        iTensor = drInv
    elif rank == 1:
        iTensor = torch.vstack((
            drInv, -tx,   -ty,   -tz,   
            tx,    -txx,  -txy,  -txz,  
            ty,    -txy,  -tyy,  -tyz,  
            tz,    -txz,  -tyz,  -tzz,  
        )).T.reshape(-1, 4, 4)
    elif rank == 2:
        iTensor = torch.vstack((
            drInv, -tx,   -ty,   -tz,   txx,   txy,   txz,   tyy,   tyz,   tzz,
            tx,    -txx,  -txy,  -txz,  txxx,  txxy,  txxz,  tyyx,  txyz,  tzzx,
            ty,    -txy,  -tyy,  -tyz,  txxy,  tyyx,  txyz,  tyyy,  tyyz,  tzzy,
            tz,    -txz,  -tyz,  -tzz,  txxz,  txyz,  tzzx,  tyyz,  tzzy,  tzzz,
            txx,   -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
            txy,   -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
            txz,   -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
            tyy,   -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
            tyz,   -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
            tzz,   -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz
        )).T.reshape(-1, 10, 10)
    else:
        raise NotImplementedError(f"Rank >= {rank} not supported")
    
    return iTensor


@torch.compile
def compute_multipolar_energy_ref(coords: torch.Tensor, pairs: torch.Tensor, multipoles: torch.Tensor):
    drVecs = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    drInv = 1 / torch.norm(drVecs, dim=1)
    iTensor = computeInteractionTensor(drVecs, [], drInv, 2)
    mPoles_j_p = multipoles[pairs[:, 1]]
    mPoles_i_p = multipoles[pairs[:, 0]]
    ene = torch.sum(torch.bmm(mPoles_j_p.unsqueeze(1), torch.bmm(iTensor, mPoles_i_p.unsqueeze(2))))
    return ene


@pytest.mark.parametrize("device, dtype", [
    # ('cpu', torch.float64), 
    # ('cpu', torch.float32), 
    # ('cuda', torch.float32),
    ('cuda', torch.float64), 
])
def test_multipolar(device, dtype):
    coords_numpy = get_water_data(100, 0.5).coords.numpy(force=True) * 10 # in atomic units
    dist_mat = cdist(coords_numpy, coords_numpy) 
    dist_mat[np.triu_indices(coords_numpy.shape[0])] = np.inf
    pairs = np.argwhere(dist_mat < 9.0).tolist()
    print("Num pairs:", len(pairs))

    coords = torch.tensor(coords_numpy.tolist(), dtype=dtype, device=device, requires_grad=True)
    pairs = torch.tensor(pairs, dtype=torch.int32)

    mutlipoles_numpy = np.random.rand(coords.shape[0], 10) * 10
    multipoles = torch.tensor(mutlipoles_numpy.tolist(), device=device, dtype=dtype, requires_grad=True)

    # perf_op(
    #     compute_multipolar_energy_ref, coords, pairs, multipoles, desc='ref', run_backward=True
    # )
    # perf_op(
    #     compute_multipolar_energy_from_atom_pairs, coords, pairs, multipoles, desc='torchff', run_backward=True
    # )

    check_op(
        compute_multipolar_energy_from_atom_pairs,
        compute_multipolar_energy_ref,
        coords, pairs, multipoles,
        check_grad=True
    )

