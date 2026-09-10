#!/usr/bin/env python3
import numpy as np
import os

from . import read_turbomole as rtm
from .constants import amu, a0, au2rcm


# =====================================================
#  Utility: Read XYZ
# =====================================================
def read_xyz(filename):
    """Read atomic symbols and coordinates (Å) from an XYZ file."""
    with open(filename, "r") as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return np.array(symbols), np.array(coords, dtype=float)


def write_xyz(filename, symbols, coords, comment=""):
    """Write atomic symbols and coordinates to XYZ file (Å)."""
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(comment + "\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


# =====================================================
#  Displace geometry along a chosen normal mode
# =====================================================
def read_normal_mode_frequencies(turbomole_dir):
    """Read the normal mode frequencies from Turbomole output directory."""
    data = rtm.turbomole_results(turbomole_dir)
    freqs, Lmat, redmasses = data.get_hessian()
    return freqs


def displace_geometry_along_mode(
        turbomole_dir,
        xyz_path,
        target_freq,
        Q_step=0.01,      # dimensionless displacement
        tolerance=1.0     # allowed mismatch in cm^-1
        ):
    """
    Displace geometry along a normal mode using Turbomole dimensionless modes.

    Args:
        turbomole_dir : Directory with vib_normal_modes and vibspectrum
        xyz_path      : Input XYZ geometry (Å)
        target_freq   : Target vibrational frequency (cm^-1)
        Q_step        : Dimensionless displacement Q
        tolerance     : Allowed mismatch in cm^-1 between target and nearest mode

    Returns:
        symbols, displaced_coords (Å), matched_freq
    """
    symbols, coords = read_xyz(xyz_path)
    n_atoms = len(symbols)

    # Read frequencies and normal modes
    data = rtm.turbomole_results(turbomole_dir)
    freqs, Lmat, redmasses = data.get_hessian()

    # Remove mass-weighting:
    masses = data.get_masses()
    nAtoms = len(symbols)
    for mode in range(3 * nAtoms):
                for row in range(3 * nAtoms):
                    atom: int = row // 3
                    Lmat[row, mode] /= np.sqrt(masses[atom])

    # Find nearest frequency
    idx = np.argmin(np.abs(freqs - target_freq))
    matched_freq = freqs[idx]
    redmass = redmasses[idx]

    if abs(matched_freq - target_freq) > tolerance:
        raise ValueError(
            f"No mode within {tolerance} cm^-1 of {target_freq}. "
            f"Closest: {matched_freq:.2f} cm^-1"
        )

    # Extract dimensionless displacement vector
    mode_vector_not_normed = Lmat[:, idx].reshape((-1,3))
    mode_vector = mode_vector_not_normed * (1./np.linalg.norm(mode_vector_not_normed))

    # Convert displacement into Angstrom:
    Q_step = Q_step / np.sqrt(redmass * amu * matched_freq / au2rcm)
    Q_step *= a0 * 1e10
    
    # Displace geometry
    displaced_coords = coords + Q_step * mode_vector

    return symbols, displaced_coords, matched_freq, Q_step

