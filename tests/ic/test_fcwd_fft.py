#!/usr/bin/env python3

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from nonradiative_rates.fcwd import run_simulation
from nonradiative_rates import fcf_utils as FcfUtils

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "testmolecule")
# realistic case without classically treated modes: TTM-1Cz (IC example), FCWD evaluated at 7000 cm-1
MOL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "ic", "relaxed-gradient_model",
                       "TTM-1Cz_GRAD-HESSIAN")


class TestFCWDFourier(unittest.TestCase):
    """Folding in Fourier space ("fft") has to reproduce folding in energy space ("direct")."""

    def _compare(self, data_dir, i_state_gradient, e_trans, low_freq_approx, rtol):
        fcwd_file = os.path.join(os.path.dirname(__file__), "fcwd_fft_test_output.dat")
        self.addCleanup(lambda: os.remove(fcwd_file) if os.path.exists(fcwd_file) else None)
        res = {}
        for method in ("direct", "fft"):
            res[method] = run_simulation(
                hessian_calculation=data_dir, egrad_calculation=data_dir,
                i_state_gradient=i_state_gradient, fcwd_file=fcwd_file, e_trans=e_trans,
                low_freq_approx=low_freq_approx, max_e=18000., verbose=False, fcwd_method=method)
        d, f = res["direct"], res["fft"]
        assert_allclose(f.val_avg, d.val_avg, rtol=rtol)
        assert_allclose(f.vala_avg, d.vala_avg, rtol=rtol)
        self.assertEqual(f.offset, d.offset)
        # Fourier-space round-off is absolute
        assert_allclose(f.fcwd, d.fcwd, rtol=0., atol=1e-15)
        assert_allclose(f.fcwd_a, d.fcwd_a, rtol=0., atol=1e-15)

    def test_ic_testmolecule(self):
        # FCWD ~ 1e-14: absolute round-off of ~1e-19 gives a relative deviation of ~1e-5
        self._compare(DATA_DIR, True, 15000., 0., rtol=1e-4)

    def test_ic_testmolecule_classical_low_freq(self):
        # classical treatment of low-frequency modes: Gaussian initial FCWD
        self._compare(DATA_DIR, True, 15000., 300., rtol=1e-4)

    def test_ttm1cz(self):
        self._compare(MOL_DIR, False, 7000., 0., rtol=1e-12)

    def test_fine_grid(self):
        # dnu = 0.1 is not exact in single precision; both folds have to bin identically
        mode_list = [[[0.6, 0.], [0.3, 143.7], [0.1, 287.4]], [[0.8, 0.], [0.2, 1436.9]]]
        fd, _ = FcfUtils.FCWD(mode_list, 0., method="direct").get_FCWD(2000., 0.1)
        ff, _ = FcfUtils.FCWD(mode_list, 0., method="fft").get_FCWD(2000., 0.1)
        assert_allclose(ff, fd, rtol=0., atol=1e-15)
        # line at 1436.9 rcm belongs to bin 14369
        self.assertAlmostEqual(fd[14369], 0.6*0.2)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            FcfUtils.FCWD([], 0., method="fourier")


if __name__ == "__main__":
    unittest.main()
