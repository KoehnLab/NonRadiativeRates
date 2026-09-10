#!/usr/bin/env python3
"""Computes an internal conversion rate constant from an FCWD simulation and a NAC vector."""

import numpy as np

from nonradiative_rates import read_turbomole as rtm
from nonradiative_rates.fcwd import run_simulation

au2rcm = 219474.63068  # cm-1 / E_h
amu = 1./5.485799090441e-4 # me
hbar = 1.054571817e-34
a0 = 5.29177210544e-11
cc = 299792458
hh = 6.62607015e-34

# Specify directories:
gamma = 0.211
hessian_dir = f"/home/linux3_i1/toews/Documents/phd/organic_radical_emitters/calculations/ttm-1cz_1/cam-b3lyp/svp/optimization/rsh_optimization_0.190_0.460_{gamma:.3f}/jobex/aoforce"
egrad_dir = f"/home/linux3_i1/toews/Documents/phd/organic_radical_emitters/calculations/ttm-1cz_1/cam-b3lyp/svp/excited_state_1/rsh_state_1_egrad_0.190_0.460_{gamma:.3f}/ridft/egrad"
nac_dir = f"/home/linux3_i1/toews/Documents/phd/organic_radical_emitters/calculations/ttm-1cz_1/cam-b3lyp/svp/excited_state_1_exeq_1/rsh_state_1_exeq_1_egrad_0.190_0.460_{gamma:.3f}/ridft/egrad"

# Specify excitation and reorganization energy:
vexc = 1.6615e+4
reo = 1479.72


def non_radiative(nac, fcwd):
    """
    Computes internal conversion rate constant from provided norm of the NAC vector (rcm) and the FCWD (1/rcm).
    """
    fcwd_si = fcwd / (100 * hh * cc)
    nac_si = (nac * 100 * hh * cc)
    knr = (np.pi/hbar) * nac_si**2 * fcwd_si

    return knr


def main():
    val_avg, vala_avg, osci_list, mode_fcf_list, mode_fcf_a_list = run_simulation(
        hessian_calculation=hessian_dir,
        egrad_calculation=egrad_dir,
        fcwd_file="FCWD.dat",
        e_trans=vexc-reo,
        low_freq_approx=300.,
        max_e=25000.,
    )

    nac_dat = rtm.turbomole_results(nac_dir)
    nacv = nac_dat.get_couplingvector(True)
    Mvect = nac_dat.get_sqrtMvector(True)

    hessian_dat = rtm.turbomole_results(hessian_dir)
    freqs, Lmat, redmass = hessian_dat.get_hessian()

    # Mass weighting and projection:
    nacv_m = nacv / Mvect * (1./np.sqrt(amu))
    nacv_p = nacv_m.T @ Lmat

    # Frequency weighting:
    nacv_pf = []
    for nacme, frq in zip(nacv_p, freqs):
        if frq < 1.:
            nacpf = 0.
        else:
            frqau = frq / au2rcm
            nacpf = nacme / np.sqrt(frqau)
        nacv_pf.append(nacpf)
    nacv_pf = freqs * np.array(nacv_pf)
    nac = np.linalg.norm(nacv_pf)
    print(f"\nNAC: {nac:.2f} rcm")

    # Computes non-radiative rate:
    knr = non_radiative(nac, val_avg)
    knr_a = non_radiative(nac, vala_avg)
    print(f"\nNon-radiative rates:\n{knr:.3e} (H)   {knr_a:.3e} (A)\n")

    # Writes output to file:
    with open(f"TTM-1Cz_{gamma}.dat", "w") as _f:
        _f.write(f"{'FREQ':>20} {'HR':>20} {'NAC':>20}\n")
        for osci,nac in zip(osci_list, nacv_pf[7:]):
            _f.write(f"{osci[0]:>20.2f} {osci[1]:>20.4e} {nac:>20.2f}\n")


if __name__ == "__main__":
    main()
