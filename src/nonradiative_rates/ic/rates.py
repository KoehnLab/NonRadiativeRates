"""Internal conversion (IC) rate constants from nonadiabatic couplings (NACs)."""
import numpy as np

from .. import read_turbomole as rtm
from ..constants import hbar, cc, hh, au2rcm, amu


def non_radiative(nac, fcwd):
    """
    Computes internal conversion rate constant from provided norm of the NAC vector (rcm) and the FCWD (1/rcm).
    """
    fcwd_si = fcwd / (100 * hh * cc)
    nac_si = (nac * 100 * hh * cc)
    knr = (np.pi/hbar) * nac_si**2 * fcwd_si

    return knr


def effective_nac(nac_calculation, hessian_calculation):
    """
    Projects the derivative-coupling vector of the Turbomole calculation nac_calculation onto
    the normal modes of hessian_calculation (mass weighting, projection, frequency weighting).
    Returns the norm of the projected vector (rcm) and its components per mode (rcm).
    Modes with frequencies below 1 rcm do not contribute.
    """
    nac_dat = rtm.turbomole_results(nac_calculation)
    nacv = nac_dat.get_couplingvector(True)
    Mvect = nac_dat.get_sqrtMvector(True)

    hessian_dat = rtm.turbomole_results(hessian_calculation)
    freqs, Lmat, _ = hessian_dat.get_hessian()

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

    return np.linalg.norm(nacv_pf), nacv_pf
