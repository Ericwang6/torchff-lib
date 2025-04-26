import torch
import torchff_nblist


if __name__ == '__main__':
    import numpy as np
    import time
    from NNPOps.neighbors import getNeighborPairs

    cutoff = 1.2

    dtype = torch.float64
    device = 'cuda'
    requires_grad = False

    N = 1000
    box_len = (N / 100) ** (1/3)
    box = torch.tensor([
        [box_len, 0.0, 0.0],
        [0.0, box_len, 0.0],
        [0.0, 0.0, box_len]
    ], requires_grad=False, dtype=dtype, device=device)
    print("Box size:", box_len)

    coords = (np.random.rand(N, 3) * box_len).tolist()
    coords = torch.tensor(coords, requires_grad=requires_grad, device=device, dtype=dtype)
    
    Ntimes = 1000
    start = time.time()
    for _ in range(Ntimes):
        pairs = getNeighborPairs(coords, cutoff, -1, box, False)[0]
    end = time.time()
    print(f"NNPOp time: {(end-start)/Ntimes*1000:.5f} ms")

    npairs_ref = int(pairs[pairs != -1].shape[0] / 2)

    start = time.time()
    for _ in range(Ntimes):
        pairs = torch.ops.torchff.build_neighbor_list_nsquared(coords, box, cutoff, -1)
    end = time.time()
    print(f"torchff time: {(end-start)/Ntimes*1000:.5f} ms")

    npairs = pairs.shape[0]

    print(npairs_ref, npairs)