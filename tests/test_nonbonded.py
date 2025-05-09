import os
from tqdm import tqdm
import pytest
import openmm as mm
import openmm.unit as unit
import openmm.app as app

import torch
import torch.nn as nn
import numpy as np
import torchff

from .utils import perf_op, check_op


@torch.compile
class Nonbonded(nn.Module):
    def forward(self, coords, pairs, box, sigma, epsilon, charges, coul_constant, cutoff, do_shift):
        drVecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        boxInv = torch.linalg.inv(box)
        dsVecs = torch.matmul(drVecs, boxInv)
        dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
        drVecsPBC = torch.matmul(dsVecsPBC, box)
        dr = torch.norm(drVecsPBC, dim=1)
        mask = dr <= cutoff
        rinv = 1 / dr
        ene = charges[pairs[:, 0]] * charges[pairs[:, 1]] * (rinv - 1 / cutoff * do_shift) * coul_constant

        sigma_ij = (sigma[pairs[:, 0]] + sigma[pairs[:, 1]]) / 2
        epsilon_ij = torch.sqrt(epsilon[pairs[:, 0]] * epsilon[pairs[:, 1]])
        tmp = (sigma_ij / dr) ** 6
        ene += 4 * epsilon_ij * tmp * (tmp - 1)

        return torch.sum(ene * mask)
    

@pytest.mark.parametrize("device, dtype, requires_grad", [
    ('cuda', torch.float32, True),
    ('cuda', torch.float64, True), 
])
def test_nonbonded_atom_pairs(device, dtype, requires_grad):
    cutoff = 4.0

    box = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ], requires_grad=False, dtype=dtype, device=device)
    
    N = 30000
    coords = (np.random.rand(N, 3) * 10.0).tolist()
    coords = torch.tensor(coords, requires_grad=requires_grad, device=device, dtype=dtype)

    Npairs = 12000000
    pairs = torch.randint(0, N-2, (Npairs, 2), device=device, dtype=torch.int32, requires_grad=False)
    pairs[:, 1] = pairs[:, 0] + 1

    sigma = (np.random.rand(N) * 0.3 + 0.001).tolist()
    sigma = torch.tensor(sigma, device=device, dtype=dtype, requires_grad=requires_grad)

    epsilon = (np.random.rand(N) * 10 + 0.05).tolist()
    epsilon = torch.tensor(epsilon, device=device, dtype=dtype, requires_grad=requires_grad)

    charges = (np.random.rand(N) + 0.001).tolist()
    charges = torch.tensor(charges, requires_grad=requires_grad, device=device, dtype=dtype)
    coul_constant = torch.tensor(20.0, device=device, requires_grad=False, dtype=dtype)

    nb = Nonbonded()

    perf_op(
        torchff.compute_nonbonded_energy_from_atom_pairs,
        coords, pairs, box, sigma, epsilon, charges, coul_constant, cutoff, True,
        desc='torchff-nb',
        run_backward=True,
        use_cuda_graph=True,
    )

    perf_op(
        nb,
        coords, pairs, box, sigma, epsilon, charges, coul_constant, cutoff, True,
        desc='ref-nb',
        run_backward=True,
        use_cuda_graph=True,
    )

    check_op(
        torchff.compute_nonbonded_energy_from_atom_pairs,
        nb,
        coords, pairs, box, sigma, epsilon, charges, coul_constant, cutoff, True,
        check_grad=True,
        atol=1e-5 if dtype is torch.float64 else 1e-2
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    perf_op(
        torchff.compute_nonbonded_forces_from_atom_pairs,
        coords, pairs, box, sigma, epsilon, charges, coul_constant, cutoff, forces,
        desc='torchff-nb-forces',
        run_backward=False,
        use_cuda_graph=True,
    )



# def test_nonbonded():
#     dirname = os.path.dirname(__file__)
#     pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
#     top = pdb.getTopology()
#     pos = pdb.getPositions()

#     cutoff = 1.2

#     ff = app.ForceField('tip3p.xml')
#     system: mm.System = ff.createSystem(
#         top,
#         nonbondedMethod=app.PME,
#         nonbondedCutoff=cutoff*unit.nanometer,
#         constraints=None,
#         rigidWater=False
#     )
#     coords = torch.tensor([[v.x, v.y, v.z] for v in pos], dtype=torch.float32, device='cuda', requires_grad=True)
#     box = torch.tensor([[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()], dtype=torch.float32, device='cuda', requires_grad=True)
#     print(box)
#     # water excls
#     exclusions = []
#     for i in range(system.getNumParticles() // 3):
#         exclusions.append([i*3+1, i*3+2])
#         exclusions.append([i*3, i*3+2])
#         exclusions.append([i*3, i*3+1])
#     exclusions = torch.tensor(exclusions, dtype=torch.int32, requires_grad=False, device='cuda')

#     omm_force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
#     charges, sigma, epsilon = [], [], []
#     for i in range(omm_force.getNumParticles()):
#         param = omm_force.getParticleParameters(i)
#         charges.append(param[0].value_in_unit(unit.elementary_charge))
#         # charges.append(0.0)
#         sigma.append(param[1].value_in_unit(unit.nanometer))
#         # epsilon.append(param[2].value_in_unit(unit.kilojoules_per_mole))
#         epsilon.append(0.0)


#     charges = torch.tensor(charges, dtype=torch.float32, device='cuda')
#     sigma = torch.tensor(sigma, dtype=torch.float32, device='cuda')
#     epsilon = torch.tensor(epsilon, dtype=torch.float32, device='cuda')
#     cutoff = omm_force.getCutoffDistance().value_in_unit(unit.nanometer)
#     prefac = torch.tensor(138.93544539709033, dtype=torch.float32, device='cuda')

#     # print(sigma, epsilon, charges)
#     sorted_atom_indices, interacting_clusters, bitmask_exclusions, num_interacting_clusters = torchff.build_cluster_pairs(
#         coords, box,
#         cutoff, exclusions,
#         0.5, 150000
#     )
#     print(torch.sum(interacting_clusters >= 0)/2)
#     print(num_interacting_clusters)

#     # prune interacting clusters
#     # boxInv = torch.linalg.inv(box)
#     # indices = []
#     # for i in tqdm(range(num_interacting_clusters.item())):
#     #     if interacting_clusters[i, 0] == interacting_clusters[i, 1]:
#     #         indices.append(i)
#     #         continue
#     #     x, y = interacting_clusters[i, 0], interacting_clusters[i, 1]
#     #     atomsx = sorted_atom_indices[x*32:(x+1)*32]
#     #     atomsy = sorted_atom_indices[y*32:(y+1)*32]
#     #     pairs = torch.cartesian_prod(atomsx, atomsy)
#     #     drVecs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
#     #     dsVecs = torch.matmul(drVecs, boxInv)
#     #     dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
#     #     drVecsPBC = torch.matmul(dsVecsPBC, box)
#     #     dr = torch.norm(drVecsPBC, dim=1)
#     #     mask = dr <= cutoff
#     #     if torch.sum(mask) > 0:
#     #         indices.append(i)

#     # print(interacting_clusters.shape, num_interacting_clusters, len(indices))


#     static_sorted_atom_indices = torch.zeros_like(sorted_atom_indices, requires_grad=False)
#     # static_interacting_clusters = interacting_clusters[torch.tensor(indices)]
#     static_interacting_clusters = torch.zeros_like(interacting_clusters, requires_grad=False)
#     static_interacting_clusters.copy_(interacting_clusters)
#     # static_interacting_clusters = torch.full_like(interacting_clusters, -1, requires_grad=False)
#     static_bitmask_exclusions = torch.zeros_like(bitmask_exclusions, requires_grad=False)
    
#     static_sorted_atom_indices.copy_(sorted_atom_indices)
#     # static_interacting_clusters.copy_(interacting_clusters)
#     static_bitmask_exclusions.copy_(bitmask_exclusions)

#     ene = torchff.compute_nonbonded_energy_from_cluster_pairs(
#         coords, box, sigma, epsilon, charges, prefac, cutoff, 
#         static_sorted_atom_indices,
#         static_interacting_clusters, 
#         static_bitmask_exclusions, 
#         True
#     )
#     print(ene)

#     # g = torch.cuda.CUDAGraph()
#     # with torch.cuda.graph(g):
#     #     ene = torchff.compute_nonbonded_energy_from_cluster_pairs(
#     #         coords, box, sigma, epsilon, charges, prefac, cutoff, 
#     #         static_sorted_atom_indices,
#     #         static_interacting_clusters, 
#     #         static_bitmask_exclusions, 
#     #         True
#     #     )
#     # g.replay()


#     forces = torch.zeros_like(coords, requires_grad=False)
#     print(forces.dtype)
#     perf_op(
#         torchff.compute_nonbonded_forces_from_cluster_pairs,
#         coords, box, sigma, epsilon, charges, prefac, cutoff, 
#         static_sorted_atom_indices, static_interacting_clusters, static_bitmask_exclusions, forces,
#         desc='nonbonded',
#         run_backward=False,
#         use_cuda_graph=True
#     )