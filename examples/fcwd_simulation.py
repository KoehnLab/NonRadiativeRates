#!/usr/bin/env python3
"""Demonstrates an FCWD simulation using the Turbomole data bundled for the unit tests."""

import os

from nonradiative_rates.fcwd import run_simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "testmolecule")


def main():
    # run with the defaults, against the bundled test data:
    result = run_simulation(
        hessian_calculation=DATA_DIR,
        egrad_calculation=DATA_DIR,
        fcwd_file="fcwd.dat",
    )

    print(f"\nharmonic FCWD average:   {result.val_avg:.4e}")
    print(f"anharmonic FCWD average: {result.vala_avg:.4e}")

    # example of overriding a few parameters, e.g. to treat low-frequency
    # modes classically and use a tighter anharmonicity threshold:
    # run_simulation(
    #     hessian_calculation=DATA_DIR,
    #     egrad_calculation=DATA_DIR,
    #     fcwd_file="fcwd_low_freq_classical.dat",
    #     anh_thr=2900,
    #     damp_thr=100.,
    #     low_freq_approx=200,
    # )


if __name__ == "__main__":
    main()
