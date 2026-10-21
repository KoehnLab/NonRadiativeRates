#!/usr/bin/env python3

import itertools
import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from nonradiative_rates import _fastfold
from nonradiative_rates import fcf_utils as FcfUtils
from nonradiative_rates.fcwd import (run_simulation, analyze_fcwd, analyze_fcwd_pairs, strongest_pairs,
                                     write_fcwd_pair_analysis)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "testmolecule")
# realistic case without classically treated modes: TTM-1Cz (IC example), FCWD evaluated at 7000 cm-1
MOL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "ic", "relaxed-gradient_model",
                       "TTM-1Cz_GRAD-HESSIAN")


def _direct_pair(result, mode_list, i, j):
    sel = [mode_list[k] for k in range(len(mode_list)) if k not in (i, j)]
    fcwd, _ = FcfUtils.FCWD(sel, result.Ereo_class).get_FCWD(result.max_e, result.e_bin)
    return np.sum(fcwd*result.window)


class TestPairAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                 i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                 low_freq_approx=0., max_e=12000., verbose=False, fcwd_method="fft")
        cls.mol_pairs = analyze_fcwd_pairs(cls.mol)

    def test_toy_model(self):
        # four modes, delta initial FCWD, arbitrary window: compare with brute-force enumeration
        rng = np.random.default_rng(7)
        mode_list = [[[0.5, 0.], [0.3, 3.], [0.2, 6.]], [[0.7, 0.], [0.3, 5.]],
                     [[0.9, 0.], [0.1, 2.]], [[0.6, 0.], [0.25, 4.], [0.15, 8.]]]
        nbins = 30
        fcwd0 = np.zeros(nbins)
        fcwd0[0] = 1.
        window = rng.uniform(0., 1., nbins)

        def brute(modes):
            total = 0.
            for quanta in itertools.product(*[range(len(m)) for m in modes]):
                ebin = sum(int(m[v][1]) for m, v in zip(modes, quanta))
                if ebin < nbins:
                    total += np.prod([m[v][0] for m, v in zip(modes, quanta)])*window[ebin]
            return total

        full, _, pair, pair_err = _fastfold.fcwd_pair_analysis_fft(fcwd0, [mode_list], window, 1.)
        assert_allclose(full[0], brute(mode_list), atol=1e-14)
        for i in range(4):
            for j in range(4):
                sel = [m for k, m in enumerate(mode_list) if k not in (i, j)]
                assert_allclose(pair[0][i, j], brute(sel), atol=1e-14)
        self.assertTrue(np.all(pair_err[0] >= 0.))

    def test_consistency_with_single_analysis(self):
        an = analyze_fcwd(self.mol, "fft")
        pa = self.mol_pairs
        assert_allclose(pa.fcwd_h, self.mol.val_avg, rtol=1e-12)
        assert_allclose(np.diag(pa.pair_h), an.fcwd_h_omit, rtol=1e-12)
        assert_allclose(np.diag(pa.pair_a), an.fcwd_a_omit, rtol=1e-12)
        for mat in (pa.pair_h, pa.pair_a, pa.synergy_h, pa.synergy_a):
            assert_allclose(mat, mat.T)
        assert_allclose(np.diag(pa.synergy_h), 1.)
        single = np.diag(pa.pair_h)
        assert_allclose(pa.synergy_h, pa.fcwd_h*pa.pair_h/np.outer(single, single)*(1-np.eye(len(single))) + np.eye(len(single)))

    def test_against_direct_folds(self):
        pa = self.mol_pairs
        an = analyze_fcwd(self.mol, "fft")
        top = list(np.argsort(-an.enhancement_h)[:4])
        pairs = [(i, j) for a, i in enumerate(top) for j in top[a+1:]] + [(0, 50), (10, 99)]
        for i, j in pairs:
            assert_allclose(pa.pair_h[i, j], _direct_pair(self.mol, self.mol.mode_fcf_list, i, j), rtol=1e-11)
            assert_allclose(pa.pair_a[i, j], _direct_pair(self.mol, self.mol.mode_fcf_a_list, i, j), rtol=1e-11)
        self.assertTrue(np.all(pa.reliable_h) and np.all(pa.reliable_a))

    def test_strongest_pairs(self):
        replacing = strongest_pairs(self.mol_pairs, npairs=3, kind="replacing")
        self.assertEqual([(round(fi), round(fj)) for fi, fj, _, _ in replacing[:1]], [(1394, 1681)])
        self.assertLess(replacing[0][2], 0.7)
        reinforcing = strongest_pairs(self.mol_pairs, npairs=3, kind="reinforcing")
        self.assertTrue(all(q > 1. for _, _, q, _ in reinforcing))
        with self.assertRaises(ValueError):
            strongest_pairs(self.mol_pairs, kind="other")

    def test_reliability_in_the_tail(self):
        # IC test molecule: FCWD ~ 1e-14; pairs of the main acceptors fall below the round-off estimate
        result = run_simulation(hessian_calculation=DATA_DIR, egrad_calculation=DATA_DIR,
                                fcwd_file=os.devnull, e_trans=15000., low_freq_approx=300., verbose=False)
        pa = analyze_fcwd_pairs(result)
        self.assertGreater(np.sum(~pa.reliable_h), 0)
        # where marked reliable, values agree with direct folds
        an = analyze_fcwd(result, "fft")
        top = list(np.argsort(-an.enhancement_h)[:5])
        checked = 0
        for a, i in enumerate(top):
            for j in top[a+1:]:
                ref = _direct_pair(result, result.mode_fcf_list, i, j)
                if pa.reliable_h[i, j]:
                    assert_allclose(pa.pair_h[i, j], ref, rtol=1e-2)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_run_simulation_option(self):
        out = os.path.join(os.path.dirname(__file__), "fcwd_pair_analysis_test_output.dat")
        self.addCleanup(lambda: os.remove(out) if os.path.exists(out) else None)
        result = run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                low_freq_approx=0., max_e=12000., verbose=False, fcwd_method="fft",
                                fcwd_analysis_pairs=True, fcwd_analysis_pairs_file=out)
        assert_allclose(result.pair_analysis.pair_h, self.mol_pairs.pair_h, rtol=1e-14)
        with open(out) as f:
            text = f.read()
        self.assertIn("replacing pairs", text)
        self.assertIn("reinforcing pairs", text)


if __name__ == "__main__":
    unittest.main()
