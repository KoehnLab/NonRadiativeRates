#!/usr/bin/env python3

# Internal conversion D1 -> D0 of the TTM-1Cz radical in the relaxed-gradient model
# (CAM-B3LYP/def2-SVP): Hessian and gradient of D0 at the D1 geometry, derivative couplings
# (NACs) at the D1 geometry; k_IC = pi/hbar |NAC|^2 FCWD.
# The FCWD is folded in Fourier space; with FCWD_analysis = True, the analysis tools are run
# (single modes and pairs omitted, minimal mode set, dominant final states). Single-mode results
# are given as enhancement factors FCWD/FCWD(-k), as in the previous version (FCWD_contr.dat).

import os
import numpy as np

from nonradiative_rates.fcwd import run_simulation
from nonradiative_rates.fcwd import minimal_mode_set, dominant_states, write_mode_set, write_dominant_states
from nonradiative_rates.fcwd import strongest_pairs
from nonradiative_rates.ic import non_radiative, effective_nac
from nonradiative_rates import read_turbomole as rtm

here = os.path.dirname(os.path.abspath(__file__))

# Specify directories (files needed by the readers; the complete data can be found on the data
# repository of the University of Stuttgart (DaRUS): doi:10.18419/DARUS-6477):
hessian_dir = os.path.join(here, "relaxed-gradient_model/TTM-1Cz_GRAD-HESSIAN/")
grad_dir = os.path.join(here, "relaxed-gradient_model/TTM-1Cz_GRAD-HESSIAN/")
nac_dir = os.path.join(here, "relaxed-gradient_model/TTM-1Cz_D1-EGRAD/")

# Specify energy gap of the non-radiative transition (value of the original example; note that
# the relaxed-gradient model would suggest e_vert + lambda = 19007 + 1554 = 20561 cm-1, with the
# vertical D1-D0 gap e_vert at the D1 geometry, cf. fcwd.relaxed_gradient_gap)
ad_exc = 1.5924e+4

# Analysis tools of the FCWD (single modes and pairs omitted, minimal mode set, dominant states):
FCWD_analysis = True


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
                 fcwd_analysis = FCWD_analysis,  # FCWD analysis: remaining FCWD with single modes omitted
                 fcwd_analysis_file = "FCWD_analysis.dat",
                 fcwd_analysis_pairs = FCWD_analysis,  # pair analysis: synergy factors of all pairs of modes
                 fcwd_analysis_pairs_file = "FCWD_pair_analysis.dat",
                 verbose = True,
                 fcwd_method = "fft"         # folding in Fourier space (analyses in a single call)
              )

    nac, nacv_pf = effective_nac(nac_dir, hessian_dir)
    print(f"\nNAC: {nac:.2f} rcm")

    # Computes non-radiative rate:
    knr = non_radiative(nac, result.val_avg)
    knr_a = non_radiative(nac, result.vala_avg)
    print(f"\nNon-radiative rates:\n{knr:.3e} (H)   {knr_a:.3e} (A)\n")

    if not FCWD_analysis:
        write_rates(result, nacv_pf)
        return

    # Enhancement factors FCWD / FCWD(without mode k) of the most important single modes, as in
    # the previous version of this example (file FCWD_contr.dat, columns "rel"); the remaining
    # FCWD in % of the full FCWD (FCWD_analysis.dat) is the inverse:
    an = result.analysis
    for label, enh in (("H", an.enhancement_h), ("A", an.enhancement_a)):
        order = np.argsort(-enh)[:5]
        print(f"Enhancement factors FCWD/FCWD(-k) ({label}): "
              + ", ".join(f"{an.freqs[i]:.0f} cm-1: {enh[i]:.3f}" for i in order))
    print()
    write_fcwd_contr("FCWD_contr.dat", an)

    # Combination effects: pairs of modes that can partly replace each other (Q < 1) or
    # reinforce each other (Q > 1), Q = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j)):
    for anharmonic, label in ((False, "H"), (True, "A")):
        for kind in ("replacing", "reinforcing"):
            pairs = strongest_pairs(result.pair_analysis, anharmonic=anharmonic, npairs=5, kind=kind)
            print(f"Most {kind} pairs ({label}), Q (FCWD without both): "
                  + ", ".join(f"{fi:.0f}/{fj:.0f}: {q:.2f} ({ret*100:.1f} %)" for fi, fj, q, ret in pairs))
    print()

    # Modes promoting the transition (greedy search, combination effects included) and
    # dominant final vibrational states, for harmonic (H) and anharmonic (A) FCFs:
    for anharmonic, label in ((False, "H"), (True, "A")):
        mode_set = minimal_mode_set(result, fraction=0.9, anharmonic=anharmonic)
        print(f"FCWD with the mode set only ({label}): {mode_set.retained*100:.1f} %, modes "
              + ", ".join(f"{frq:.0f}" for frq in mode_set.freqs) + " cm-1")
        write_mode_set(f"mode_set_{label}.dat", mode_set)

        states = dominant_states(result, nstates=10, anharmonic=anharmonic)
        print(f"Dominant final states ({label}), contribution to the FCWD (sum: {sum(st.fraction for st in states)*100:.1f} %):")
        for st in states[:5]:
            print(f"  {st.fraction*100:6.2f} %  {st.energy:8.1f} cm-1  "
                  + " ".join(f"{frq:.0f}({nq})" for frq, nq in st.quanta.items()))
        write_dominant_states(f"dominant_states_{label}.dat", states)
        print()

    write_rates(result, nacv_pf)


def write_fcwd_contr(fname, an):
    # Enhancement factors in the format of the previous version (run_mode_tests, FCWD_contr.dat):
    # FCWD with mode k omitted and FCWD/FCWD(-k), modes in descending order of frequency
    with open(fname, "w") as outf:
        print(f"{'FREQ':>16} {'FCWDh':>16} {'FCWDh rel':>16}  {'FCWDa':>16} {'FCWDa rel':>16}", file=outf)
        for k in range(len(an.freqs)-1, -1, -1):
            print(f"{an.freqs[k]:>16.2f} {an.fcwd_h_omit[k]:>16.5e} {an.enhancement_h[k]:>16.6f}  "
                  f"{an.fcwd_a_omit[k]:>16.5e} {an.enhancement_a[k]:>16.6f} ", file=outf)


def write_rates(result, nacv_pf):
    # Writes frequencies, Huang-Rhys factors and projected NACs of all modes entering the FCWD
    # (those with frequencies >= 1e-3 cm-1, as in run_simulation(); translations, rotations and
    # imaginary modes excluded) to file:
    freqs, _, _ = rtm.turbomole_results(hessian_dir).get_hessian()
    nac_modes = np.asarray(nacv_pf)[np.asarray(freqs) >= 1e-3]
    with open("rates_new.dat", "w") as _f:
        _f.write(f"{'FREQ':>20} {'HR':>20} {'NAC':>20}\n")
        for osci, nac in zip(result.osci_list, nac_modes):
            _f.write(f"{osci[0]:>20.2f} {osci[1]:>20.4e} {nac:>20.2f}\n")


if __name__ == "__main__":
    main()
