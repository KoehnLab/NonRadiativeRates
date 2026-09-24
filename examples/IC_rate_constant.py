#!/usr/bin/env python3

import numpy as np

from nonradiative_rates.fcwd import run_simulation
from nonradiative_rates.tools import non_radiative
from nonradiative_rates import read_turbomole as rtm
from nonradiative_rates.constants import au2rcm, amu

# Specify directories:
hessian_dir = "./relaxed-gradient_model/TTM-1Cz_GRAD-HESSIAN/"
grad_dir = "./relaxed-gradient_model/TTM-1Cz_GRAD-HESSIAN/"
nac_dir = "./relaxed-gradient_model/TTM-1Cz_D1-EGRAD/"

# Specify energy gap of the non-radiative transition
ad_exc = 1.5924e+4

#def run_simulation(
#        hessian_calculation=hessian_dir,
#        egrad_calculation=grad_dir,
#        i_state_gradient = False,   # gradient refers to initial state? T/F
#        fcwd_file="FCWD.dat",       # output file for FCWD
#        D = 30000,                  # diss. energy (for estimate of anharm.)
#        anh_thr = 400.,             # use anh. approx for modes larger than this
#        e_trans = ad_exc,         # energy where FCWD is measured
#        sigma = 100.,               # width of Gaussian energy window centered at e_trans
#        damp = 10.,                 # damping for low-frequency modes
#        damp_thr = 100.,            # threshold for low-frequency modes
#        low_freq_approx = 300.,     # classical approx. for modes < this val.
#        thrmod = 1e-24,             # cutoff threshold for modes
#        maxquanta = 200,            # max. quanta (cutoff should lead to smaller value)
#        max_e = 25000.,             # maximum energy for comp. FCWD (e_trans + several sigma)
#        e_bin = 1.,                 # binning for FCWD
#        run_mode_tests = False       # see end of this routine
#        ):


def main():
    result = run_simulation(
                 hessian_calculation=hessian_dir,
                 egrad_calculation=grad_dir,
                 i_state_gradient = False,
                 fcwd_file="FCWD_new.dat",
                 D = 30000,                  # diss. energy (for estimate of anharm.)
                 anh_thr = 400.,             # use anh. approx for modes larger than this
                 e_trans = ad_exc,         # energy where FCWD is measured
                 sigma = 100.,               # width of Gaussian energy window centered at e_trans
                 damp = 10.,                 # damping for low-frequency modes
                 damp_thr = 100.,            # threshold for low-frequency modes
                 low_freq_approx = 300.,     # classical approx. for modes < this val.
                 thrmod = 1e-24,             # cutoff threshold for modes
                 maxquanta = 200,            # max. quanta (cutoff should lead to smaller value)
                 max_e = 25000.,             # maximum energy for comp. FCWD (e_trans + several sigma)
                 e_bin = 1.,                 # binning for FCWD
                 run_mode_tests = True,     # check for mode contributions
                 mode_contrib_file = "FCWD_contr.dat",
                 verbose = True
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
    knr = non_radiative(nac, result.val_avg)
    knr_a = non_radiative(nac, result.vala_avg)
    print(f"\nNon-radiative rates:\n{knr:.3e} (H)   {knr_a:.3e} (A)\n")

    # Writes output to file:
    with open(f"rates_new.dat", "w") as _f:
        _f.write(f"{'FREQ':>20} {'HR':>20} {'NAC':>20}\n")
        for osci,nac in zip(result.osci_list, nacv_pf[7:]):
            _f.write(f"{osci[0]:>20.2f} {osci[1]:>20.4e} {nac:>20.2f}\n")
    


if __name__ == "__main__":
    main()
