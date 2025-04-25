import os
import openmm as mm
import openmm.app as app
import openmm.unit as unit


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

print("Num Atoms:", top.getNumAtoms())
print("Box: ", top.getPeriodicBoxVectors())

for idx in range(system.getNumForces()):
    f = system.getForce(idx)
    f.setForceGroup(idx)

integrator = mm.LangevinMiddleIntegrator(298.15, 1.0, 0.0005)
simulation = app.Simulation(top, system, integrator)

simulation.context.reinitialize(preserveState=True)
simulation.context.setPositions(pos)

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

energies.pop('NonbondedForce')
for key, energy in energies.items():
    print(f"{key:<25s}  {energy:14.6f}")
