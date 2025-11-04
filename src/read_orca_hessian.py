import re
import numpy as np

def parse_orca_hess(filename):
    """Parse ORCA .hess file to extract masses, coordinates, frequencies, and orthogonalized normal modes."""
    with open(filename) as f:
        text = f.read()

    # ----------------------------------------------------------
    # $atoms  → masses and coordinates
    # ----------------------------------------------------------
    atoms_block = re.search(r'\$atoms\s+(\d+)\s+(.*?)\$', text, re.S)
    if not atoms_block:
        raise ValueError("No $atoms block found.")
    natoms = int(atoms_block.group(1))
    atom_lines = atoms_block.group(2).strip().splitlines()
    masses, coords = [], []
    for line in atom_lines[:natoms]:
        parts = line.split()
        masses.append(float(parts[1]))
        coords.append([float(x) for x in parts[2:5]])
    masses = np.array(masses)
    coords = np.array(coords)

    # ----------------------------------------------------------
    # $vibrational_frequencies  → frequencies
    # ----------------------------------------------------------
    freq_block = re.search(r'\$vibrational_frequencies\s+(\d+)\s+(.*?)\$', text, re.S)
    if not freq_block:
        raise ValueError("No $vibrational_frequencies block found.")
    nfreq = int(freq_block.group(1))
    freq_lines = freq_block.group(2).strip().splitlines()
    freqs = np.array([float(line.split()[1]) for line in freq_lines[:nfreq]])

    # ----------------------------------------------------------
    # $normal_modes  → normal coordinate vectors
    # ----------------------------------------------------------
    nm_block = re.search(r'\$normal_modes\s+(.*?)\$', text, re.S)
    if not nm_block:
        raise ValueError("No $normal_modes block found.")
    lines = nm_block.group(1).strip().splitlines()
    nrow, ncol = map(int, lines[0].split())
    normal_modes = np.zeros((nrow, ncol))

    col_indices = []
    for line in lines[1:]:
        parts = line.split()
        # Column index header lines
        if all(re.fullmatch(r'\d+', p) for p in parts):
            col_indices = list(map(int, parts))
        elif col_indices:
            row = int(parts[0])
            vals = [float(x.replace('D', 'E')) for x in parts[1:]]
            normal_modes[row, col_indices[:len(vals)]] = vals

    # ----------------------------------------------------------
    # Reassemble and orthogonalize normal coordinate matrix
    # ----------------------------------------------------------
    n_dof = 3 * natoms
    assert n_dof == normal_modes.shape[0] == normal_modes.shape[1], "Matrix size mismatch."

    Lmat = np.copy(normal_modes)

    # Undo ORCA's mass-weighting: multiply each coordinate by sqrt(mass)
    for mode in range(n_dof):
        for row in range(n_dof):
            atom = row // 3
            Lmat[row, mode] *= np.sqrt(masses[atom])

    # Normalize each mode vector
    for mode in range(n_dof):
        norm = np.linalg.norm(Lmat[:, mode])
        if norm > 0:
            Lmat[:, mode] /= norm

    # Reapply mass weighting (back to ORCA convention) to compute effective masses
    Lmat2 = np.copy(Lmat)
    for mode in range(n_dof):
        for row in range(n_dof):
            atom = row // 3
            Lmat2[row, mode] /= np.sqrt(masses[atom])

    # Compute reduced masses
    red_masses = 1.0 / np.diag(Lmat2.T @ Lmat2)

    # Sanity check: Lmat should now be orthogonal (diagonal Gram matrix)
    product = np.matmul(Lmat.T, Lmat)
    if np.sum(np.abs(product - np.diag(np.diag(product))) > 1e-4):
        raise ValueError("Lmat * Lmat^T not diagonal; check orthogonalization.")

    return masses, coords, freqs, Lmat, red_masses


# ----------------------------------------------------------
# Example usage
# ----------------------------------------------------------
# masses, coords, freqs, Lmat, red_masses = parse_orca_hess("PBE0_TZVP.hess")
# print("Atoms:", len(masses))
# print("Frequencies (cm^-1):", freqs[:10])
# print("Reduced masses (amu):", red_masses[:10])
# print(red_masses)
# print("Lmat shape:", Lmat.shape)

