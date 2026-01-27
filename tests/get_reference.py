import os
from typing import Dict
from dataclasses import dataclass
import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


@dataclass
class WaterData:
    num: int
    coords: torch.Tensor
    box: torch.Tensor
    cutoff: float
    energies: Dict[str, float]
    bonds: torch.Tensor
    b0: torch.Tensor
    kb: torch.Tensor
    angles: torch.Tensor
    th0: torch.Tensor
    kth: torch.Tensor
    sigma: torch.Tensor
    epsilon: torch.Tensor
    charges: torch.Tensor
    exclusions: torch.Tensor
    mass: torch.Tensor
    forces: np.ndarray


def get_water_data(n, cutoff, dtype=torch.float32, device='cuda', 
                   coord_grad=True, box_grad=True, param_grad=True):
    dirname = os.path.dirname(__file__)
    pdb = app.PDBFile(os.path.join(dirname, f'water/water_{n}.pdb'))
    top = pdb.getTopology()
    pos = pdb.getPositions()

    ff = app.ForceField('tip3p.xml')
    system = ff.createSystem(
        top,
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff*unit.nanometer,
        constraints=None,
        rigidWater=False
    )

    for idx in range(system.getNumForces()):
        f = system.getForce(idx)
        f.setForceGroup(idx)
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(False)
            f.setUseSwitchingFunction(False)
            f.setReactionFieldDielectric(1.0)
    
    # Torch inputs
    coords = torch.tensor([[v.x, v.y, v.z] for v in pos], dtype=dtype, device=device, requires_grad=coord_grad)
    box = torch.tensor(
        [[v.x, v.y, v.z] for v in top.getPeriodicBoxVectors()], 
        dtype=dtype, device=device, requires_grad=box_grad
    )
    mass = torch.tensor([system.getParticleMass(i)._value for i in range(top.getNumAtoms())], dtype=dtype, device=device, requires_grad=False).reshape(-1, 1)
    bonds, b0, kb = [], [], []
    angles, th0, kth = [], [], []
    sigma, epsilon, charges = [], [], []
    
    for omm_force in system.getForces():
        if isinstance(omm_force, mm.HarmonicBondForce):
            for i in range(omm_force.getNumBonds()):
                bond = omm_force.getBondParameters(i)
                bonds.append([bond[0], bond[1]])
                b0.append(bond[2].value_in_unit(unit.nanometer))
                kb.append(bond[3].value_in_unit(unit.kilojoule_per_mole/(unit.nanometer**2)))
        elif isinstance(omm_force, mm.HarmonicAngleForce):
            for i in range(omm_force.getNumAngles()):
                angle = omm_force.getAngleParameters(i)
                angles.append([angle[0], angle[1], angle[2]])
                th0.append(angle[3].value_in_unit(unit.radian))
                kth.append(angle[4].value_in_unit(unit.kilojoule_per_mole/(unit.radian**2)))
        elif isinstance(omm_force, mm.NonbondedForce):
            for i in range(omm_force.getNumParticles()):
                param = omm_force.getParticleParameters(i)
                charges.append(param[0].value_in_unit(unit.elementary_charge))
                sigma.append(param[1].value_in_unit(unit.nanometer))
                epsilon.append(param[2].value_in_unit(unit.kilojoules_per_mole))
    
    bonds = torch.tensor(bonds, dtype=torch.int32, device=device)
    kb = torch.tensor(kb, dtype=dtype, device=device, requires_grad=param_grad)
    b0 = torch.tensor(b0, dtype=dtype, device=device, requires_grad=param_grad)

    angles = torch.tensor(angles, dtype=torch.int32, device=device)
    kth = torch.tensor(kth, dtype=dtype, device=device, requires_grad=param_grad)
    th0 = torch.tensor(th0, dtype=dtype, device=device, requires_grad=param_grad)

    charges = torch.tensor(charges, dtype=dtype, device=device, requires_grad=param_grad)
    sigma = torch.tensor(sigma, dtype=dtype, device=device, requires_grad=param_grad)
    epsilon = torch.tensor(epsilon, dtype=dtype, device=device, requires_grad=param_grad)
    
    excl_i, excl_j = [], []
    for n in range(system.getNumParticles()//3):
        for i in range(3):
            for j in range(3):
                excl_i.append(n*3+i)
                excl_j.append(n*3+j)
    exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device=device)
    
    # Reference OpenMM energies
    integrator = mm.LangevinMiddleIntegrator(298.15, 1.0, 0.0005)
    simulation = app.Simulation(top, system, integrator)

    simulation.context.reinitialize(preserveState=True)
    simulation.context.setPositions(pos)

    forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value

    energies = {}
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if isinstance(force, mm.CMMotionRemover):
            continue
        state = simulation.context.getState(getEnergy=True, groups={idx})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies[force.__class__.__name__] = energy


    # seperate charges
    for idx in range(system.getNumForces()):
        f = system.getForce(idx)
        if isinstance(f, mm.NonbondedForce):
            for i in range(f.getNumParticles()):
                p = f.getParticleParameters(i)
                f.setParticleParameters(i, 0.0, p[1], p[2])

    simulation.context.reinitialize(preserveState=True)
    simulation.context.setPositions(pos)

    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if not isinstance(force, mm.NonbondedForce):
            continue
        state = simulation.context.getState(getEnergy=True, groups={idx})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies['LennardJones'] = energy

    energies['Coulomb'] = energies['NonbondedForce'] - energies['LennardJones']

    return WaterData(
        n, coords, box, cutoff, energies,
        bonds, b0, kb,
        angles, th0, kth,
        sigma, epsilon, charges,
        exclusions,
        mass, forces
    )


if __name__ == '__main__':
    water_data = get_water_data(100, 0.4)
    for key, energy in water_data.energies.items():
        print(f"{key:<25s}  {energy:14.6f}")