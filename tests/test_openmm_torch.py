import os

import torch
import torch.nn as nn
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
    

class TorchSystem(nn.Module):
    def __init__(self, omm_system: mm.System):
        super(TorchSystem, self).__init__()
        self.forces = nn.ModuleList()
        for force in omm_system.getForces():
            if isinstance(force, mm.HarmonicBondForce):
                self.forces.append(HarmonicBondTorchForce(force))
    
    def forward(self, coords: torch.Tensor, box: torch.Tensor):
        ene = torch.tensor(0.0, dtype=coords.dtype, device=coords.device)
        for force in self.forces:
            ene += force(coords, box)
        return ene


dirname = os.path.dirname(__file__)
pdb = app.PDBFile(os.path.join(dirname, 'water/water_100.pdb'))
top = pdb.getTopology()
pos = pdb.getPositions()

ff = app.ForceField('tip3p.xml')
system = ff.createSystem(
    top,
    nonbondedMethod=app.CutoffPeriodic,
    nonbondedCutoff=0.5*unit.nanometer,
    constraints=None,
    rigidWater=False
)

t_system = mm.System()
for i in range(system.getNumParticles()):
    t_system.addParticle(system.getParticleMass(i))

model = TorchSystem(system)
tforce = TorchForce(torch.jit.script(model))
tforce.setOutputsForces(False)
tforce.setUsesPeriodicBoundaryConditions(True)

t_system.addForce(tforce)

context = mm.Context(t_system, mm.VerletIntegrator(1.0), mm.Platform.getPlatformByName('CUDA'))
context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
context.setPositions(pos)
state = context.getState(getEnergy=True)
print(state.getPotentialEnergy())
