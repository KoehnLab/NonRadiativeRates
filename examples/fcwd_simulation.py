#!/usr/bin/env python3
"""Demonstrates an FCWD simulation using the bundled example Turbomole data."""

from nonradiative_rates.fcwd import run_simulation


def main():
    # run with the defaults, against the bundled example data:
    run_simulation(
        hessian_calculation="testmolecule",
        egrad_calculation="testmolecule",
        fcwd_file="fcwd.dat",
    )

    # example of overriding a few parameters, e.g. to treat low-frequency
    # modes classically and use a tighter anharmonicity threshold:
    # run_simulation(
    #     hessian_calculation="testmolecule",
    #     egrad_calculation="testmolecule",
    #     fcwd_file="fcwd_low_freq_classical.dat",
    #     anh_thr=2900,
    #     damp_thr=100.,
    #     low_freq_approx=200,
    # )


if __name__ == "__main__":
    main()
