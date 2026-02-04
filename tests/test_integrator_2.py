import os, sys
import time
import torch
import torch.nn as nn
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from dataclasses import fields
import torchff

from .get_reference import get_water_data



def test_md():
    cutoff = 1.2
    water_data = get_water_data(1000, cutoff+0.2, torch.float32, 'cuda', True, False, False)
    coords_start = water_data.coords.detach().clone()

    ene = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=False)
    forces = torch.zeros_like(water_data.coords, dtype=torch.float32, device='cuda', requires_grad=False)
    velocities = torch.zeros_like(water_data.coords, dtype=torch.float32, device='cuda', requires_grad=False)
    
    dt = 0.001
    coul_constant = torch.tensor(138.93544539709033, dtype=torch.float32, device='cuda')
    nblist_method = 'cluster-pairs'
    rebuid_freq = 20
    total_steps = 5000

    if nblist_method == 'atom-pairs':
        nblist_trial, num_pairs = torchff.build_neighbor_list_nsquared(water_data.coords, water_data.box, water_data.cutoff, -1, False)
        mask = torch.floor_divide(nblist_trial[:, 0], 3) != torch.floor_divide(nblist_trial[:, 1], 3)
        nblist_trial = nblist_trial[mask]
        nblist = torch.full((int(nblist_trial.shape[0]*1.2), 2), -1, dtype=nblist_trial.dtype, device='cuda')
        nblist[:nblist_trial.shape[0]].copy_(nblist_trial)
    else:
        nblist_trial = torchff.build_cluster_pairs(
            water_data.coords, water_data.box, water_data.cutoff, water_data.exclusions, 0.6
        )
        nexcl = int(nblist_trial[1].shape[1] * 2)
        nint = int(torch.sum(nblist_trial[-1] != -1).cpu().item()*2)
        nblist_trial = torchff.build_cluster_pairs(
            water_data.coords, water_data.box, water_data.cutoff, water_data.exclusions, 0.6, nint
        )
        cluster_exclusions = torch.full((2, nexcl), -1, dtype=nblist_trial[1].dtype, device='cuda')
        cluster_exclusions[:, :nblist_trial[1].shape[1]].copy_(nblist_trial[1])
        bitmask_exclusions = torch.full((nexcl, 32), -1, dtype=nblist_trial[2].dtype, device='cuda')
        bitmask_exclusions[:nblist_trial[2].shape[0]].copy_(nblist_trial[2])
        nblist = (nblist_trial[0], cluster_exclusions, bitmask_exclusions, nblist_trial[-2], nblist_trial[-1])
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fields(water_data):
            t = getattr(water_data, f.name)
            if isinstance(t, torch.Tensor):
                t.grad = None
        ebond = torchff.compute_harmonic_bond_energy(water_data.coords, water_data.bonds, water_data.b0, water_data.kb)
        eangle = torchff.compute_harmonic_angle_energy(water_data.coords, water_data.angles, water_data.th0, water_data.kth)
        # enb = torchff.compute_nonbonded_energy_from_atom_pairs(water_data.coords, nblist, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges, coul_constant, cutoff, True)
        enb = torchff.compute_nonbonded_energy_from_cluster_pairs(
            water_data.coords, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges,
            coul_constant, cutoff, *nblist, do_shift=True 
        )
        ene = ebond + eangle + enb
        ene.backward()

        velocities.add_(-water_data.coords.grad / water_data.mass * dt)
        with torch.no_grad():
            water_data.coords.add_(velocities * dt)
            water_data.coords.grad.zero_()

    torch.cuda.current_stream().wait_stream(s)
    
    # Record operations
    g = torch.cuda.CUDAGraph()
    for f in fields(water_data):
        t = getattr(water_data, f.name)
        if isinstance(t, torch.Tensor):
            t.grad = None
    with torch.cuda.graph(g):

        ebond = torchff.compute_harmonic_bond_energy(water_data.coords, water_data.bonds, water_data.b0, water_data.kb)
        eangle = torchff.compute_harmonic_angle_energy(water_data.coords, water_data.angles, water_data.th0, water_data.kth)
        # enb = torchff.compute_nonbonded_energy_from_atom_pairs(water_data.coords, nblist, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges, coul_constant, cutoff, True)
        enb = torchff.compute_nonbonded_energy_from_cluster_pairs(
            water_data.coords, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges,
            coul_constant, cutoff, *nblist, do_shift=True 
        )
        ene = ebond + eangle + enb
        ene.backward()

        velocities.add_(-water_data.coords.grad / water_data.mass * dt)
        with torch.no_grad():
            water_data.coords.add_(velocities * dt)
            water_data.coords.grad.zero_()
    

    # Start simulation
    with torch.no_grad():
        water_data.coords.copy_(coords_start)
        forces.zero_()
        velocities.zero_()

    # torchff.compute_harmonic_bond_forces(water_data.coords, water_data.bonds, water_data.b0, water_data.kb, forces)
    # torchff.compute_harmonic_angle_forces(water_data.coords, water_data.angles, water_data.th0, water_data.kth, forces)
    # # torchff.compute_nonbonded_forces_from_atom_pairs(water_data.coords, nblist, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges, coul_constant, cutoff, forces)
    # torchff.compute_nonbonded_forces_from_cluster_pairs(
    #     water_data.coords, water_data.box, water_data.sigma, water_data.epsilon, water_data.charges,
    #     coul_constant, cutoff, *nblist, forces=forces 
    # )
    # print(forces)
    # exit(0)

    coords_last_rebuild = torch.zeros_like(water_data.coords)
    coords_last_rebuild.copy_(water_data.coords)

    start = time.perf_counter()

    for _ in range(total_steps//rebuid_freq):
        for _ in range(rebuid_freq):
            g.replay()

        # print(water_data.coords)
        with torch.no_grad():
            # print("Max deviation:", torch.max(torch.norm(water_data.coords-coords_last_rebuild, dim=1)))
            # coords_last_rebuild.copy_(water_data.coords)
            if nblist_method == 'atom-pairs':
                nblist_trial, num_pairs = torchff.build_neighbor_list_nsquared(water_data.coords, water_data.box, water_data.cutoff, -1, False)
                mask = torch.floor_divide(nblist_trial[:, 0], 3) != torch.floor_divide(nblist_trial[:, 1], 3)
                nblist_trial = nblist_trial[mask]
                torch.full(nblist.shape, -1, out=nblist)
                nblist[:nblist_trial.shape[0]].copy_(nblist_trial)
            else:
                nblist_trial = torchff.build_cluster_pairs(
                    water_data.coords, water_data.box, water_data.cutoff, water_data.exclusions, 0.6, nint
                )
                torch.full(nblist[1].shape, -1, out=nblist[1], dtype=nblist[1].dtype)
                torch.full(nblist[2].shape, -1, out=nblist[2], dtype=nblist[2].dtype)
                nblist[0].copy_(nblist_trial[0])
                nblist[1][:, :nblist_trial[1].shape[1]].copy_(nblist_trial[1])
                nblist[2][:nblist_trial[2].shape[0]].copy_(nblist_trial[2])
                nblist[3].copy_(nblist_trial[3])
                nblist[4].copy_(nblist_trial[4])


        # torchff.build_cluster_pairs(coords, water_data.box, water_data.cutoff, water_data.exclusions, 0.6)
    torch.cuda.synchronize()

    end = time.perf_counter()

    time_per_step = (end - start) / total_steps
    print(f"Time per step: {time_per_step*1000:.3f} ms")
    perf = 24 * 3600 / time_per_step / 1e6
    print(f"Performance: {perf:.3f} ns/day")