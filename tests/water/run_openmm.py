#!/usr/bin/env python

import argparse
import sys
import time

try:
    import openmm as mm
    from openmm import app, unit
except ImportError:
    # Fall back to the old simtk namespace if needed
    from simtk import openmm as mm  # type: ignore
    from simtk.openmm import app, unit  # type: ignore


def _select_platform(requested: str | None) -> mm.Platform | None:
    """Select an OpenMM platform, preferring CUDA if available."""
    if requested is not None:
        return mm.Platform.getPlatformByName(requested)

    for name in ("CUDA", "ROCM", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an NVE simulation of a water box with PME (8 Å cutoff) using OpenMM.\n"
            "Water is modeled as non-rigid; timestep is 0.001 ps."
        )
    )
    parser.add_argument(
        "pdb",
        help="Path to water PDB file (e.g. water_1000.pdb)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of MD steps to run (default: 100000)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="OpenMM platform to use (e.g. CUDA, CPU). If not set, choose automatically.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Initial temperature in Kelvin for velocity initialization (default: 300 K)",
    )

    args = parser.parse_args()

    # Load the PDB (assumed to be a pure water box)
    pdb = app.PDBFile(args.pdb)

    # Use a flexible water model: PME with 8 Å cutoff, no constraints, rigidWater=False
    # tip3p.xml is bundled with OpenMM; it defines water nonbonded parameters.
    forcefield = app.ForceField("tip3p.xml")

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.Ewald,
        nonbondedCutoff=8.0 * unit.angstrom,
        constraints=None,         # No constraints -> water not rigid
        rigidWater=False,
        removeCMMotion=True,
    )

    # NVE integrator: plain Verlet with 0.001 ps timestep
    timestep = 0.001 * unit.picoseconds
    integrator = mm.VerletIntegrator(timestep)

    platform = _select_platform(args.platform)
    if platform is not None:
        simulation = app.Simulation(pdb.topology, system, integrator, platform)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator)

    simulation.context.setPositions(pdb.positions)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)

    # Run the simulation and time it
    start_time = time.time()
    simulation.step(args.steps)
    end_time = time.time()

    wall_seconds = end_time - start_time
    simulated_ps = args.steps * 0.001  # timestep is 0.001 ps

    # Convert to ns/day
    simulated_ns = simulated_ps / 1000.0
    ns_per_day = simulated_ns * 86400.0 / wall_seconds if wall_seconds > 0 else float("inf")

    print(
        f"Simulated {simulated_ns:.3f} ns in {wall_seconds:.1f} s "
        f"-> speed = {ns_per_day:.2f} ns/day ({wall_seconds/args.steps*1000:.2f} ms/step)"
    )


if __name__ == "__main__":
    main()

