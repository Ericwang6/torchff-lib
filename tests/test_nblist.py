import os
import pytest
import time
import numpy as np
import torch
import torch.nn as nn
import torchff
import openmm as mm
import openmm.app as app


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

    cutoff = 1.2

    # warm up
    for _ in range(10):
        pairs_nsquared = torchff.build_neighbor_list_nsquared(coords, box, cutoff, coords.shape[0] * 500)
        pairs_clist = torchff.build_neighbor_list_cell_list(coords, box, cutoff, coords.shape[0] * 500, cutoff, False)
        # print(pairs_clist.shape, pairs_nsquared.shape)
        assert pairs_nsquared.shape[0] == pairs_clist.shape[0]
    
    n = 1000
    time_nsq = []
    time_clist = []
    for _ in range(n):
        start = time.perf_counter()
        pairs = torchff.build_neighbor_list_nsquared(coords, box, cutoff, coords.shape[0] * 500)
        end = time.perf_counter()
        time_nsq.append((end-start)*1000)

        torch.cuda.synchronize()

        start = time.perf_counter()
        pairs = torchff.build_neighbor_list_cell_list(coords, box, cutoff, coords.shape[0] * 500, cutoff, False)
        end = time.perf_counter()
        time_clist.append((end-start)*1000)
    
    print("O(N^2) Time: ", f'{np.mean(time_nsq):.5f}+-{np.std(time_nsq):.5f} ms')
    print("Cell List Time: ", f'{np.mean(time_clist):.5f}+-{np.std(time_clist):.5f} ms')


        