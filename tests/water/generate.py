import openmm as mm
import openmm.app as app
import openmm.unit as unit


num_waters = 1000000
box_size = (num_waters / (6.02 * 100 / 18)) ** (1/3)

ff = app.ForceField('tip3p.xml')
top = app.Topology()
pos = []
modeller = app.Modeller(top, pos)
modeller.addSolvent(ff, 'tip3p', numAdded=num_waters)

top = modeller.getTopology()
pos = modeller.getPositions()

print(top.getPeriodicBoxVectors())

integrator = mm.LangevinMiddleIntegrator(298.15, 1.0, 0.0005)
system = ff.createSystem(top, nonbondedMethod=app.PME, nonbondedCutoff=0.5*unit.nanometer, rigidWater=False, constraints=None)

simulation = app.Simulation(top, system, integrator)
simulation.context.setPositions(pos)
simulation.minimizeEnergy()

opt_pos = simulation.context.getState(getPositions=True).getPositions()

with open(f'water_{num_waters}.pdb', 'w') as f:
    app.PDBFile.writeFile(top, opt_pos, f)