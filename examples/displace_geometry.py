#!/usr/bin/env python3
"""Displace a geometry along a chosen normal mode read from a Turbomole calculation."""

from nonradiative_rates.vibtools import displace_geometry_along_mode, write_xyz

# adjust these paths to point at your own Turbomole aoforce directory and starting geometry:
hessian_dir = "/home/linux3_i1/toews/Documents/phd/quantum_sensors/calculations/pentacene_0/b3lyp/svp/optimization/jobex/aoforce"
xyz_path = "/home/linux3_i1/toews/Documents/phd/quantum_sensors/optimized/pentacene_0_b3lyp_svp_opt.xyz"


def main():
    symbols, displaced_coords, matched_freq, Q_step = displace_geometry_along_mode(
        hessian_dir,
        xyz_path,
        target_freq=997.18,
        Q_step=0.1,       # dimensionless displacement
        tolerance=0.01,   # allowed mismatch in cm^-1
        )

    print(f"matched mode at {matched_freq:.2f} cm^-1, displaced by Q = {Q_step:.4f}")
    write_xyz("displaced.xyz", symbols, displaced_coords, comment=f"displaced along {matched_freq:.2f} cm^-1 mode")


if __name__ == "__main__":
    main()
