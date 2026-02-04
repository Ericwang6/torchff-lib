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



def test_nonbonded_cluster_pairs():
    dirname = os.path.dirname(__file__)
    pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
    top = pdb.getTopology()
    pos = pdb.getPositions()

    cutoff = 1.2

    ff = app.ForceField('tip3p.xml')
    system: mm.System = ff.createSystem(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff*unit.nanometer,
        constraints=None,
        rigidWater=False
    )
    coords = torch.tensor([[v.x, v.y, v.z] for v in pos], dtype=torch.float32, device='cuda', requires_grad=True)
    box = torch.tensor([[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()], dtype=torch.float32, device='cuda', requires_grad=False)

    # water excls
    excl_i, excl_j = [], []
    for n in range(system.getNumParticles()//3):
        for i in range(3):
            for j in range(3):
                excl_i.append(n*3+i)
                excl_j.append(n*3+j)
    exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

    omm_force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    charges, sigma, epsilon = [], [], []
    for i in range(omm_force.getNumParticles()):
        param = omm_force.getParticleParameters(i)
        # charges.append(param[0].value_in_unit(unit.elementary_charge))
        charges.append(0.0)
        sigma.append(param[1].value_in_unit(unit.nanometer))
        epsilon.append(param[2].value_in_unit(unit.kilojoules_per_mole))
        # epsilon.append(0.0)

    charges = torch.tensor(charges, dtype=torch.float32, device='cuda', requires_grad=False)
    sigma = torch.tensor(sigma, dtype=torch.float32, device='cuda', requires_grad=False)
    epsilon = torch.tensor(epsilon, dtype=torch.float32, device='cuda', requires_grad=False)
    cutoff = omm_force.getCutoffDistance().value_in_unit(unit.nanometer)
    prefac = torch.tensor(138.93544539709033, dtype=torch.float32, device='cuda')

    sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms = torchff.build_cluster_pairs(
        coords, box, cutoff, exclusions, 0.7, -1
    )

    print("Num cluster pairs w/  exclusions:", cluster_exclusions.shape[1])
    print("Num cluster pairs w/o exclusions:", torch.sum(interacting_clusters != -1))


    # ene = torchff.compute_nonbonded_energy_from_cluster_pairs(
    #     coords, box, sigma, epsilon, charges, prefac, cutoff, 
    #     sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
    #     True
    # )
    # print(ene)


    forces = torch.zeros_like(coords, requires_grad=False)
    # perf_op(
    #     torchff.compute_nonbonded_forces_from_cluster_pairs,
    #     coords, box, sigma, epsilon, charges, prefac, cutoff, 
    #     sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
    #     forces,
    #     desc='nonbonded-forces-cluster-pairs',
    #     run_backward=False,
    #     use_cuda_graph=True
    # )

    perf_op(
        torchff.compute_nonbonded_energy_from_cluster_pairs,
        coords, box, sigma, epsilon, charges, prefac, cutoff, 
        sorted_atom_indices, cluster_exclusions, bitmask_exclusions, interacting_clusters, interacting_atoms,
        True,
        desc='nonbonded-forces(backward)-cluster-pairs',
        run_backward=True,
        use_cuda_graph=True
    )

    pairs, _ = torchff.build_neighbor_list_nsquared(coords, box, cutoff, -1, False)
    mask = torch.floor_divide(pairs[:, 0], 3) != torch.floor_divide(pairs[:, 1], 3)
    pairs = pairs[mask, :].clone()
    print(f"Num pairs: {pairs.shape[0]}")

    # perf_op(
    #     torchff.compute_nonbonded_forces_from_atom_pairs,
    #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff-0.1, 
    #     forces,
    #     desc='nonbonded-forces-atom-pairs',
    #     run_backward=False,
    #     use_cuda_graph=True
    # )


    # perf_op(
    #     torchff.compute_nonbonded_energy_from_atom_pairs,
    #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff, 
    #     True,
    #     desc='nonbonded-forces(backward)-atom-pairs',
    #     run_backward=True,
    #     use_cuda_graph=True
    # )

    # nb = Nonbonded()
    # perf_op(
    #     nb,
    #     coords, pairs, box, sigma, epsilon, charges, prefac, cutoff, 
    #     True,
    #     desc='nonbonded-forces-atom-pairs',
    #     run_backward=True,
    #     use_cuda_graph=True
    # )
