import math
import numpy as np
import os


def create_test_data(num: int, rank: int = 2):
    """Create random test data for PME tests and return as numpy arrays."""
    # Set a physically reasonable box length scaling with number of atoms
    boxLen = float((num * 10.0) ** (1.0 / 3.0))

    # Random coordinates in [0, boxLen)
    coords_np = np.random.rand(num, 3) * boxLen

    # Random charges, shifted so that the total charge is zero
    q_np = np.random.randn(num)
    q_np -= q_np.mean()

    # Random dipoles
    d_np = np.random.randn(num, 3)

    # Random symmetric, traceless quadrupoles per atom
    t_np = np.empty((num, 3, 3), dtype=float)
    for i in range(num):
        A = np.random.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace  # make traceless
        t_np[i] = sym

    # Cubic box
    box_np = np.eye(3) * boxLen

    # Find appropriate PME parameters.
    alpha_pme = math.sqrt(-math.log10(2 * 1e-6)) / 9.0
    max_hkl = 10  # Reasonable default for PME grid

    return coords_np, box_np, q_np, d_np, t_np, alpha_pme, max_hkl


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate data for different sizes
    sizes = [100, 300, 1000, 3000, 10000]
    rank = 2
    
    for n in sizes:
        print(f"Generating data for N={n}...")
        coords, box, q, p, t, alpha, max_hkl = create_test_data(n, rank=rank)
        
        # Save to npz file
        output_path = os.path.join(script_dir, f"random_water_{n}.npz")
        np.savez(
            output_path,
            coords=coords,
            box=box,
            q=q,
            p=p,
            t=t,
            alpha=alpha,
            max_hkl=max_hkl,
            rank=rank,
            n=n
        )
        print(f"Saved to {output_path}")
