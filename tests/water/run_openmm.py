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


def _get_ewald_params(system: mm.System, context: mm.Context) -> tuple[float, int]:
    """Return (alpha, kmax) for the Ewald/PME NonbondedForce.

    alpha is returned in units of 1/nm.  kmax is derived from the PME grid
    dimensions as max(nx, ny, nz), which is a reasonable proxy for the
    reciprocal-space resolution.
    """
    nb_force: mm.NonbondedForce | None = None
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            nb_force = f
            break
    if nb_force is None:
        return float("nan"), 0

    try:
        alpha, nx, ny, nz = nb_force.getPMEParametersInContext(context)
    except Exception:
        alpha, nx, ny, nz = nb_force.getPMEParameters()

    # alpha has dimensions of 1/length.
    # alpha_inv_nm = alpha.value_in_unit(1.0 / unit.nanometer)
    kmax = int(max(nx, ny, nz))
    return alpha, kmax


def run_openmm_water_md(
    pdb: str,
    steps: int = 100_000,
    platform: str | None = None,
    temperature: float = 300.0,
    use_pme: bool = True,
    alpha: float | None = None,
    kmax: int | None = None,
) -> tuple[float, float, int]:
    """Run a short NVE water simulation and return (ms/step, alpha, kmax).

    This is a programmatic entry point that mirrors the CLI behaviour of this
    script.  It is intended to be re-used by benchmarks such as
    ``examples/fixed_charge_benchmark.py``.

    Parameters
    ----------
    pdb : str
        Path to the water PDB file.
    steps : int
        Number of MD steps to run.
    platform : str | None
        OpenMM platform (e.g. CUDA, CPU). If None, auto-select.
    temperature : float
        Initial temperature in Kelvin for velocity initialization.
    use_pme : bool
        If True, use PME instead of Ewald for long-range electrostatics.
    alpha : float | None
        PME separation parameter in 1/nm. If provided with kmax, overrides
        OpenMM defaults to match TorchFF.
    kmax : int | None
        PME reciprocal-space grid size (max of nx, ny, nz). If provided with
        alpha, overrides OpenMM defaults to match TorchFF.
    """
    # Load the PDB (assumed to be a pure water box)
    pdb_file = app.PDBFile(pdb)

    # Flexible TIP3P (8 A cutoff), no constraints.
    forcefield = app.ForceField("tip3p.xml")

    nonbonded_method = app.PME if use_pme else app.Ewald
    system = forcefield.createSystem(
        pdb_file.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=8.0 * unit.angstrom,
        constraints=None,  # No constraints -> water not rigid
        rigidWater=False,
        removeCMMotion=True,
    )

    # Set PME parameters to match TorchFF when alpha and kmax are provided.
    if use_pme and alpha is not None and kmax is not None:
        for f in system.getForces():
            if isinstance(f, mm.NonbondedForce):
                # alpha in 1/nm; nx=ny=nz=kmax for cubic box
                f.setPMEParameters(alpha, kmax, kmax, kmax)
                break

    timestep = 0.001 * unit.picoseconds
    integrator = mm.VerletIntegrator(timestep)

    plat = _select_platform(platform)
    if plat is not None:
        simulation = app.Simulation(pdb_file.topology, system, integrator, plat)
    else:
        simulation = app.Simulation(pdb_file.topology, system, integrator)

    simulation.context.setPositions(pdb_file.positions)
    simulation.context.setPeriodicBoxVectors(*pdb_file.topology.getPeriodicBoxVectors())
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # Extract Ewald/PME parameters for this configuration.
    alpha, kmax = _get_ewald_params(system, simulation.context)

    start_time = time.time()
    simulation.step(steps)
    end_time = time.time()

    wall_seconds = end_time - start_time
    ms_per_step = wall_seconds / steps * 1000.0 if steps > 0 else float("inf")
    return ms_per_step, alpha, kmax


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
    ms_per_step, alpha_inv_nm, kmax = run_openmm_water_md(
        pdb=args.pdb,
        steps=args.steps,
        platform=args.platform,
        temperature=args.temperature,
    )

    simulated_ps = args.steps * 0.001  # timestep is 0.001 ps
    wall_seconds = ms_per_step * args.steps / 1000.0 if ms_per_step != float("inf") else float("inf")
    simulated_ns = simulated_ps / 1000.0
    ns_per_day = simulated_ns * 86400.0 / wall_seconds if wall_seconds > 0 else float("inf")

    print(
        f"Simulated {simulated_ns:.3f} ns in {wall_seconds:.1f} s "
        f"-> speed = {ns_per_day:.2f} ns/day "
        f"({ms_per_step:.2f} ms/step, alpha={alpha_inv_nm:.6f} 1/nm, kmax={kmax})"
    )


if __name__ == "__main__":
    main()

