import os, time
import sys
import torch
import torch.nn as nn
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmtorch import TorchForce

import torchff


class HarmonicBondTorchForce(nn.Module):
    def __init__(self, omm_force: mm.HarmonicBondForce):
        super(HarmonicBondTorchForce, self).__init__()
        self.bonds = []
        self.k = []
        self.b0 = []
        for i in range(omm_force.getNumBonds()):
            bond = omm_force.getBondParameters(i)
            self.bonds.append([bond[0], bond[1]])
            self.b0.append(bond[2].value_in_unit(unit.nanometer))
            self.k.append(bond[3].value_in_unit(unit.kilojoule_per_mole/(unit.nanometer**2)))
        
        self.bonds = torch.tensor(self.bonds, dtype=torch.int32, device='cuda')
        self.k = torch.tensor(self.k, dtype=torch.float32, device='cuda')
        self.b0 = torch.tensor(self.b0, dtype=torch.float32, device='cuda')

    def forward(self, coords: torch.Tensor, box: torch.Tensor):
        return torchff.compute_harmonic_bond_energy(coords, self.bonds, self.b0, self.k)
        # r = torch.norm(coords[self.bonds[:, 0]] - coords[self.bonds[:, 1]], dim=1)
        # ene = (r - self.b0) ** 2 * self.k / 2
        # return torch.sum(ene)


class HarmonicAngleTorchForce(nn.Module):
    def __init__(self, omm_force: mm.HarmonicAngleForce):
        super(HarmonicAngleTorchForce, self).__init__()
        self.angles = []
        self.k = []
        self.th0 = []
        for i in range(omm_force.getNumAngles()):
            angle = omm_force.getAngleParameters(i)
            self.angles.append([angle[0], angle[1], angle[2]])
            self.th0.append(angle[3].value_in_unit(unit.radian))
            self.k.append(angle[4].value_in_unit(unit.kilojoule_per_mole/(unit.radian**2)))
        
        self.angles = torch.tensor(self.angles, dtype=torch.int32, device='cuda')
        self.k = torch.tensor(self.k, dtype=torch.float32, device='cuda')
        self.th0 = torch.tensor(self.th0, dtype=torch.float32, device='cuda')

    def forward(self, coords: torch.Tensor, box: torch.Tensor):
        return torchff.compute_harmonic_angle_energy(coords, self.angles, self.th0, self.k)
        # v1 = coords[self.angles[:, 0]] - coords[self.angles[:, 1]]
        # v2 = coords[self.angles[:, 2]] - coords[self.angles[:, 1]]
        # dot_product = torch.sum(v1 * v2, dim=1)
        # mag_v1 = torch.norm(v1, dim=1)
        # mag_v2 = torch.norm(v2, dim=1)
        # cos_theta = dot_product / (mag_v1 * mag_v2)
        # cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical issues
        # theta = torch.acos(cos_theta)
        # ene = (theta - self.th0) ** 2 * self.k / 2
        # return torch.sum(ene)


class NonbondedTorchForce(nn.Module):
    def __init__(self, omm_force: mm.NonbondedForce):
        super(NonbondedTorchForce, self).__init__()
        self.charges = []
        self.sigma = []
        self.epsilon = []

        for i in range(omm_force.getNumParticles()):
            param = omm_force.getParticleParameters(i)
            self.charges.append(param[0].value_in_unit(unit.elementary_charge))
            self.sigma.append(param[1].value_in_unit(unit.nanometer))
            self.epsilon.append(param[2].value_in_unit(unit.kilojoules_per_mole))
        
        self.charges = torch.tensor(self.charges, dtype=torch.float32, device='cuda')
        self.sigma = torch.tensor(self.sigma, dtype=torch.float32, device='cuda')
        self.epsilon = torch.tensor(self.epsilon, dtype=torch.float32, device='cuda')
        self.cutoff = omm_force.getCutoffDistance().value_in_unit(unit.nanometer)
        self.prefac = torch.tensor(138.93544539709033, dtype=torch.float32, device='cuda')

        self.counter = 0
        self.nsteps_rebuild = 1
        self.pairs = torch.tensor([[0, 1]], dtype=torch.int32, device='cuda')

        self.max_npairs = omm_force.getNumParticles() * 500
        self.prev_coords = torch.zeros((omm_force.getNumParticles(), 3), dtype=torch.float32, device='cuda')
    
    def forward(self, coords: torch.Tensor, box: torch.Tensor):
        # if self.counter % self.nsteps_rebuild == 0:
            # pairs = torchffs.build_neighbor_list_nsquared(coords, box, self.cutoff, self.max_npairs)
        
        # if self.counter == 0 or torch.norm(coords - self.prev_coords, dim=1) > 0.2:
        pairs = torchff.build_neighbor_list_cell_list(coords, box, self.cutoff, self.max_npairs, self.cutoff, False)
        mask = torch.floor_divide(pairs[:, 0], 3) != torch.floor_divide(pairs[:, 1], 3)
        self.pairs = pairs[mask, :]
            # self.counter += 1

        coul = torchff.compute_coulomb_energy(coords, self.pairs, box, self.charges, self.prefac, self.cutoff)
        lj = torchff.compute_lennard_jones_energy(coords, self.pairs, box, self.sigma, self.epsilon, self.cutoff)
        # self.prev_coords = coords
        return coul + lj
        
        # drVecs = coords[self.pairs[:, 0]] - coords[self.pairs[:, 1]]
        # boxInv = torch.linalg.inv(box)
        # dsVecs = torch.matmul(drVecs, boxInv)
        # dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
        # drVecsPBC = torch.matmul(dsVecsPBC, box)
        # dr = torch.norm(drVecsPBC, dim=1)
        # mask = dr <= self.cutoff
        # ene = self.charges[self.pairs[:, 0]] * self.charges[self.pairs[:, 1]] / dr
        # ene_0 = self.charges[self.pairs[:, 0]] * self.charges[self.pairs[:, 1]] / self.cutoff
        # coul = torch.sum((ene - ene_0) * mask) * self.prefac

        # sigma_ij = (self.sigma[self.pairs[:, 0]] + self.sigma[self.pairs[:, 1]]) / 2
        # epsilon_ij = torch.sqrt(self.epsilon[self.pairs[:, 0]] * self.epsilon[self.pairs[:, 1]])
        # tmp = (sigma_ij / dr) ** 6
        # lj = torch.sum(4 * epsilon_ij * tmp * (tmp - 1) * mask)

        # return lj + coul
            

class TorchSystem(nn.Module):
    def __init__(self, omm_system: mm.System):
        super(TorchSystem, self).__init__()
        self.forces = nn.ModuleList()
        for force in omm_system.getForces():
            if isinstance(force, mm.HarmonicBondForce):
                self.forces.append(HarmonicBondTorchForce(force))
            if isinstance(force, mm.HarmonicAngleForce):
                self.forces.append(HarmonicAngleTorchForce(force))
            # if isinstance(force, mm.NonbondedForce):
            #     self.forces.append(NonbondedTorchForce(force))

        self.ene = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        self.forces_tensor = torch.zeros((omm_system.getNumParticles(), 3), dtype=torch.float32, device='cuda')
    
    def forward(self, coords: torch.Tensor, box: torch.Tensor):
        return self.forces[0](coords, box) + self.forces[1](coords, box)
        # ene = torch.tensor(0.0, dtype=coords.dtype, device=coords.device)
        # for force in self.forces:
        #     ene += force(coords, box)
        # return ene
        # return self.fake_zero, self.fake_zeros
        # self.ene.zero_()
        # self.forces_tensor.zero_()
        # torchff.compute_harmonic_bond_energy_and_forces(
        #     coords, 
        #     self.forces[0].bonds,
        #     self.forces[0].b0,
        #     self.forces[0].k,
        #     self.ene,
        #     self.forces_tensor
        # )
        # return self.ene, self.forces_tensor


dirname = os.path.dirname(__file__)
pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
top = pdb.getTopology()
pos = pdb.getPositions()

ff = app.ForceField('tip3p.xml')
system: mm.System = ff.createSystem(
    top,
    nonbondedMethod=app.CutoffPeriodic,
    nonbondedCutoff=0.4*unit.nanometer,
    constraints=None,
    rigidWater=False
)
# system.removeForce([i for i in range(system.getNumForces()) if isinstance(system.getForce(i), mm.NonbondedForce)][0])

for idx in range(system.getNumForces()):
    f = system.getForce(idx)
    if isinstance(f, mm.NonbondedForce):
        f.setUseDispersionCorrection(False)
        f.setUseSwitchingFunction(False)
        f.setReactionFieldDielectric(1.0)

t_system = mm.System()
for i in range(system.getNumParticles()):
    t_system.addParticle(system.getParticleMass(i))

model = torch.jit.script(TorchSystem(system))
# model = TorchSystem(system)

# # benchmark pure inference
# pos_tensor = torch.tensor([[v.x, v.y, v.z] for v in pos], dtype=torch.float32, device='cuda', requires_grad=True)
# box_tensor = torch.tensor([[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()], dtype=torch.float32, device='cuda', requires_grad=True)
# for _ in range(10):
#     ene = model(pos_tensor, box_tensor)
#     ene.backward()
#     # pos_tensor = pos_tensor + 1.0
#     # pos_tensor.grad.zero_()

# start = time.perf_counter()

# for i in range(1000):
#     ene = model(pos_tensor, box_tensor)
#     ene.backward()
#         # pos_tensor = pos_tensor + 1.0
#         # pos_tensor.grad.zero_()
# end = time.perf_counter()
# print(f"Pure inference time: {(end-start):.4f} ms")

# exit(0)

tforce = TorchForce(model)
tforce.setOutputsForces(False)
tforce.setProperty("useCUDAGraphs", "true")
tforce.setUsesPeriodicBoundaryConditions(True)

t_system.addForce(tforce)

context = mm.Context(system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName('CUDA'))
context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
context.setPositions(pos)
state = context.getState(getEnergy=True, getForces=True)
ene_ref = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
forces_ref = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
print(f'OpenMM Energy: {ene_ref} kJ/mol')


# context = mm.Context(t_system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName('CUDA'))
# context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
# context.setPositions(pos)
# state = context.getState(getEnergy=True, getForces=True)
# ene = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
# forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
# print(f'TorchFF Energy: {ene} kJ/mol')


integrator = mm.VerletIntegrator(0.001)
simulation = app.Simulation(
    top, 
    # t_system, 
    system,
    integrator,
    mm.Platform.getPlatformByName('CUDA')
)
simulation.context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
simulation.context.setPositions(pos)
simulation.context.setVelocities(np.zeros((top.getNumAtoms(), 3)))
# print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
# simulation.minimizeEnergy()


# Configure a reporter to print to the console every 0.1 ps (100 steps)
reporter = app.StateDataReporter(file=sys.stdout, reportInterval=5000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)
simulation.reporters.append(reporter)

simulation.step(10000)