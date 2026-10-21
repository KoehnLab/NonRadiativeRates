"""Collection of tools to compute the bits and pieces needed for non-radiative rates.

Kept for backward compatibility: the IC rate constant now lives in nonradiative_rates.ic.
"""
from .ic.rates import non_radiative

__all__ = ["non_radiative"]
