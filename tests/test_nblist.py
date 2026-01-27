import os
import pytest
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import torchff
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .utils import perf_op


def test_nblist_cell_list():
    dirname = os.path.dirname(__file__)
    pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
    top = pdb.getTopology()
    boxVectors = top.getPeriodicBoxVectors()
    print(boxVectors)
    box = torch.tensor([
        [boxVectors[0].x, boxVectors[0].y, boxVectors[0].z],
        [boxVectors[1].x, boxVectors[1].y, boxVectors[1].z],
        [boxVectors[2].x, boxVectors[2].y, boxVectors[2].z]
    ], dtype=torch.float32, device='cuda', requires_grad=False)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value.tolist(),
        dtype=torch.float32, device='cuda', requires_grad=True
    )

    cutoff = 0.4

    pairs_nsquared, num = torchff.build_neighbor_list_nsquared(coords, box, cutoff, -1, False)
    print(pairs_nsquared.shape, num)
    mask = torch.floor_divide(pairs_nsquared[:, 0], 3) != torch.floor_divide(pairs_nsquared[:, 1], 3)
    pairs_nsquared = pairs_nsquared[mask]

    

    # pairs_clist, _ = torchff.build_neighbor_list_cell_list(coords, box, cutoff, coords.shape[0] * 500, 0.5, False, True)
    excl_i, excl_j = [], []
    for n in range(top.getNumAtoms()//3):
        for i in range(3):
            for j in range(3):
                excl_i.append(n*3+i)
                excl_j.append(n*3+j)
    exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

    pairs_clist, _ = torchff.build_neighbor_list_cluster_pairs(
        coords, box, cutoff, exclusions, 0.4, -1, -1, False
    )

    # pairs_clist_set = set()
    # for p in pairs_clist.detach().cpu().numpy().tolist():
    #     if (min(p), max(p)) in pairs_clist_set:
    #         print(f"Duplicated: {(min(p), max(p))}")
    #     pairs_clist_set.add((min(p), max(p)))
    
    pairs_nsquared_set = set((min(p), max(p)) for p in pairs_nsquared.detach().cpu().numpy().tolist())
    print(len(pairs_nsquared_set))
    pairs_clist_set = set((min(p), max(p)) for p in pairs_clist.detach().cpu().numpy().tolist())
    print(len(pairs_clist_set))
    clist_diff_set = pairs_clist_set.difference(pairs_nsquared_set)
    if len(clist_diff_set) > 0:
        print("The fist 10 pairs are in cell-list nblist but not in nsquared:", list(clist_diff_set)[:10])
        clist_diff = torch.tensor(list(clist_diff_set), dtype=torch.int32)
        print(clist_diff)
        diff_vec = coords[clist_diff[:, 0]] - coords[clist_diff[:, 1]]
        print(torch.norm(((diff_vec / box[0][0]) - torch.round(diff_vec / box[0][0])) * box[0][0], dim=1))
    nsq_diff_set = pairs_nsquared_set.difference(pairs_clist_set)

    if len(nsq_diff_set) > 0:
        print("The first 10 pairs are in nsquared nblist but not in cell-list:", list(nsq_diff_set)[:10])
        nsq_diff = torch.tensor(list(nsq_diff_set), dtype=torch.int32)
        diff_vec = coords[nsq_diff[:, 0]] - coords[nsq_diff[:, 1]]
        print(torch.norm(((diff_vec / box[0][0]) - torch.round(diff_vec / box[0][0])) * box[0][0], dim=1))

    assert pairs_nsquared.shape[0] == pairs_clist.shape[0]
           
    # perf_op(
    #     torchff.build_neighbor_list_nsquared,
    #     coords, box, cutoff, coords.shape[0]*500,
    #     desc='O(N^2) NBList',
    #     run_backward=False,
    #     use_cuda_graph=False,
    #     explicit_sync=True
    # )
    # perf_op(
    #     torchff.build_neighbor_list_cluster_pairs,
    #     coords, box, cutoff, None, 0.5, -1, -1, False,
    #     desc='Cluster Pair NBList',
    #     run_backward=False,
    #     use_cuda_graph=False,
    #     explicit_sync=True
    # )
    

def test_build_cluster_pair_perf():
    dirname = os.path.dirname(__file__)
    pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
    cutoff = 1.2

    top = pdb.getTopology()
    boxVectors = top.getPeriodicBoxVectors()
    print(boxVectors)
    box = torch.tensor([
        [boxVectors[0].x, boxVectors[0].y, boxVectors[0].z],
        [boxVectors[1].x, boxVectors[1].y, boxVectors[1].z],
        [boxVectors[2].x, boxVectors[2].y, boxVectors[2].z]
    ], dtype=torch.float32, device='cuda', requires_grad=False)
    coords = torch.tensor(
        pdb.getPositions(asNumpy=True)._value.tolist(),
        dtype=torch.float32, device='cuda', requires_grad=True
    )
    ff = app.ForceField('tip3p.xml')
    system: mm.System = ff.createSystem(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff*unit.nanometer,
        constraints=None,
        rigidWater=False
    )

   
    # water excls
    excl_i, excl_j = [], []
    for n in range(system.getNumParticles()//3):
        for i in range(3):
            for j in range(3):
                excl_i.append(n*3+i)
                excl_j.append(n*3+j)
    exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

    perf_op(
        torchff.build_cluster_pairs,
        coords, box, cutoff, exclusions, 0.6, -1,
        use_cuda_graph=False,
        run_backward=False
    )