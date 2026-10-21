#!/usr/bin/env python3

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from nonradiative_rates import fcwd as Fcwd
from nonradiative_rates import fcf_utils as FcfUtils
from nonradiative_rates import _fastfold

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "testmolecule")
# realistic case without classically treated modes: TTM-1Cz (IC example), FCWD evaluated at 7000 cm-1
MOL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "ic", "relaxed-gradient_model",
                       "TTM-1Cz_GRAD-HESSIAN")


def _mode_lists(data_dir, i_state_gradient, low_freq_approx, max_e):
    osci_list = Fcwd._read_oscillators(data_dir, data_dir, i_state_gradient, 10., 100., verbose=False)
    mode_fcf_list, mode_fcf_a_list, Ereo_class, _ = Fcwd._compute_mode_fcfs(
        osci_list, 30000., 400., low_freq_approx, 1e-24, 200, max_e, verbose=False)
    return mode_fcf_list, mode_fcf_a_list, Ereo_class


class TestModeContributions(unittest.TestCase):
    """The single-call analysis in Fourier space has to reproduce separate folds in energy space."""

    def _compare(self, data_dir, i_state_gradient, e_trans, low_freq_approx, rtol, max_e=18000., e_bin=1.):
        mode_fcf_list, mode_fcf_a_list, Ereo_class = _mode_lists(data_dir, i_state_gradient, low_freq_approx, max_e)
        for mode_list in (mode_fcf_list, mode_fcf_a_list):
            gen = FcfUtils.FCWD(mode_list, Ereo_class)
            fcwd, offset = gen.get_FCWD(max_e, e_bin)
            grid = offset + e_bin*np.arange(len(fcwd))
            window = np.exp(-0.5*((grid-e_trans)/100.)**2)/(100.*np.sqrt(2.*np.pi))*e_bin

            full, omit = gen.get_mode_contributions(max_e, e_bin, window)
            self.assertEqual(len(omit), len(mode_list))
            assert_allclose(full, np.sum(fcwd*window), rtol=rtol)

            # reference: separate folds in energy space for some of the modes
            for idx in sorted({0, 1, len(mode_list)//2, len(mode_list)-2, len(mode_list)-1}):
                sel = mode_list[:idx] + mode_list[idx+1:]
                fcwd_s, _ = FcfUtils.FCWD(sel, Ereo_class).get_FCWD(max_e, e_bin)
                assert_allclose(omit[idx], np.sum(fcwd_s*window), rtol=rtol)

    def test_ic_testmolecule(self):
        # FCWD ~ 1e-14: absolute Fourier-space round-off gives relative deviations of ~1e-6
        self._compare(DATA_DIR, True, 15000., 0., rtol=1e-4)

    def test_ic_testmolecule_classical_low_freq(self):
        # Gaussian initial FCWD (classical treatment of low-frequency modes)
        self._compare(DATA_DIR, True, 15000., 300., rtol=1e-4)

    def test_ttm1cz(self):
        self._compare(MOL_DIR, False, 7000., 0., rtol=1e-11)

    def test_ttm1cz_fine_grid(self):
        self._compare(MOL_DIR, False, 7000., 0., rtol=1e-11, max_e=10000., e_bin=0.25)

    def test_toy_model(self):
        # two modes, delta initial FCWD: omitting one mode leaves the progression of the other
        mode_list = [[[0.6, 0.], [0.4, 3.]], [[0.7, 0.], [0.3, 5.]]]
        fcwd = np.zeros(10)
        fcwd[0] = 1.
        window = np.zeros(10)
        window[3] = 1.
        window[8] = 2.
        full, omit = _fastfold.mode_contributions_fft(fcwd, mode_list, window, 1.)
        assert_allclose(full, 0.4*0.7 + 2.*0.4*0.3, atol=1e-15)
        assert_allclose(omit, [0., 0.4], atol=1e-15)

    def test_window_on_other_grid(self):
        with self.assertRaises(ValueError):
            _fastfold.mode_contributions_fft(np.ones(10), [[[1., 0.]]], np.ones(9), 1.)

    def test_run_simulation_fcwd_analysis(self):
        # fcwd_analysis: "fft" (single call) and "direct" (separate folds) agree, in the result and in the file
        out = {}
        for method in ("direct", "fft"):
            fcwd_file = os.path.join(os.path.dirname(__file__), f"fcwd_an_{method}.dat")
            analysis_file = os.path.join(os.path.dirname(__file__), f"fcwd_analysis_{method}.dat")
            for f in (fcwd_file, analysis_file):
                self.addCleanup(lambda f=f: os.remove(f) if os.path.exists(f) else None)
            result = Fcwd.run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                         i_state_gradient=False, fcwd_file=fcwd_file, e_trans=7000.,
                                         low_freq_approx=0., max_e=12000., verbose=False,
                                         fcwd_analysis=True, fcwd_analysis_file=analysis_file,
                                         fcwd_method=method)
            an = result.analysis
            assert_allclose(an.fcwd_h, result.val_avg, rtol=1e-12)
            assert_allclose(an.fcwd_a, result.vala_avg, rtol=1e-12)
            assert_allclose(an.enhancement_h, an.fcwd_h/an.fcwd_h_omit)
            assert_allclose(an.retained_h, an.fcwd_h_omit/an.fcwd_h)
            assert_allclose(an.retained_a*an.enhancement_a, 1.)
            self.assertEqual(len(an.freqs), len(result.mode_fcf_list))
            out[method] = (an, np.loadtxt(analysis_file, skiprows=2))
        for attr in ("fcwd_h_omit", "fcwd_a_omit", "enhancement_h", "enhancement_a"):
            assert_allclose(getattr(out["fft"][0], attr), getattr(out["direct"][0], attr), rtol=1e-11)
        assert_allclose(out["fft"][1], out["direct"][1], rtol=1e-5)
        # most important accepting mode of TTM-1Cz at 7000 cm-1
        an = out["fft"][0]
        self.assertEqual(round(an.freqs[np.argmax(an.enhancement_h)]), 1681)

    def test_single_call_for_several_sets(self):
        mode_lists = _mode_lists(MOL_DIR, False, 0., 12000.)[:2]
        fcwd0 = np.zeros(12000)
        fcwd0[0] = 1.
        window = np.exp(-0.5*((np.arange(12000)-7000.)/100.)**2)/(100.*np.sqrt(2.*np.pi))
        full, omit, enh = _fastfold.fcwd_analysis_fft(fcwd0, mode_lists, window, 1.)
        for iset, mode_list in enumerate(mode_lists):
            full_s, omit_s = _fastfold.mode_contributions_fft(fcwd0, mode_list, window, 1.)
            assert_allclose(full[iset], full_s, rtol=1e-14)
            assert_allclose(omit[iset], omit_s, rtol=1e-14)
            assert_allclose(enh[iset], full_s/omit_s, rtol=1e-14)

    def test_deprecated_run_mode_tests(self):
        analysis_file = os.path.join(os.path.dirname(__file__), "fcwd_contr_deprecated.dat")
        self.addCleanup(lambda: os.remove(analysis_file) if os.path.exists(analysis_file) else None)
        with self.assertWarns(DeprecationWarning):
            result = Fcwd.run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                         i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                         low_freq_approx=0., max_e=10000., verbose=False,
                                         run_mode_tests=True, mode_contrib_file=analysis_file,
                                         fcwd_method="fft")
        self.assertIsNotNone(result.analysis)
        self.assertTrue(os.path.exists(analysis_file))


if __name__ == "__main__":
    unittest.main()
