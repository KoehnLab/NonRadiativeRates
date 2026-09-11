import numpy as np

from .constants import hbar, cc, hh


def non_radiative(nac, fcwd):
    """
    Computes internal conversion rate constant from provided norm of the NAC vector (rcm) and the FCWD (1/rcm).
    """
    fcwd_si = fcwd / (100 * hh * cc)
    nac_si = (nac * 100 * hh * cc)
    knr = (np.pi/hbar) * nac_si**2 * fcwd_si

    return knr
