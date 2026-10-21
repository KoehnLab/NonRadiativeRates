#!/usr/bin/env python3

import itertools
import math
import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from nonradiative_rates import _fastfold
from nonradiative_rates import fcf_utils as FcfUtils
from nonradiative_rates.fcwd import (run_simulation, minimal_mode_set, dominant_states,
                                     write_mode_set, write_dominant_states)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "testmolecule")
# realistic case without classically treated modes: TTM-1Cz (IC example), FCWD evaluated at 7000 cm-1
MOL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "ic", "relaxed-gradient_model",
                       "TTM-1Cz_GRAD-HESSIAN")


def _toy_modes(seed, nmodes=4, nquanta=6):
    rng = np.random.default_rng(seed)
    modes = []
    for _ in range(nmodes):
        S, w = rng.uniform(0.2, 1.5), rng.uniform(80., 400.)
        modes.append([[math.exp(-S)*S**v/math.factorial(v), v*w] for v in range(nquanta)])
    return modes


def _brute_force(mode_list, fcwd0, window):
    """contributions of all states of the quantum modes, with the initial FCWD folded in"""
    nbins = len(window)
    states = {}
    for quanta in itertools.product(*[range(len(m)) for m in mode_list]):
        weight = np.prod([mode_list[k][v][0] for k, v in enumerate(quanta)])
        ebin = sum(int(np.floor(mode_list[k][v][1])) for k, v in enumerate(quanta))
        contrib = sum(fcwd0[b0]*window[ebin+b0] for b0 in np.nonzero(fcwd0)[0] if ebin+b0 < nbins)
        states[quanta] = (weight*contrib, ebin)
    return sorted(states.items(), key=lambda item: -item[1][0])


class TestDominantStates(unittest.TestCase):

    def _check_toy(self, fcwd0):
        mode_list = _toy_modes(3)
        nbins = len(fcwd0)
        window = np.exp(-0.5*((np.arange(nbins)-900.)/60.)**2)
        contrib, bins, quanta = _fastfold.dominant_states(fcwd0, mode_list, window, 1., kbest=60, nstates=8)
        ref = _brute_force(mode_list, fcwd0, window)[:8]
        assert_allclose(contrib, [r[1][0] for r in ref], rtol=1e-12)
        self.assertEqual(list(bins), [r[1][1] for r in ref])
        self.assertEqual([tuple(q) for q in quanta], [r[0] for r in ref])

    def test_toy_delta_start(self):
        fcwd0 = np.zeros(2500)
        fcwd0[0] = 1.
        self._check_toy(fcwd0)

    def test_toy_spread_start(self):
        # initial distribution as for classically treated modes
        fcwd0 = np.zeros(2500)
        fcwd0[:7] = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        self._check_toy(fcwd0)

    def test_ttm1cz(self):
        result = run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                low_freq_approx=0., max_e=15000., verbose=False)
        freqs = {round(o[0]): o for o in result.osci_list}
        for anharmonic in (False, True):
            states = dominant_states(result, nstates=10, anharmonic=anharmonic)
            self.assertEqual(len(states), 10)
            contribs = [st.contribution for st in states]
            self.assertEqual(contribs, sorted(contribs, reverse=True))
            self.assertLess(sum(st.fraction for st in states), 1.)
            # contributions follow from the quantum numbers: prod_k FCF_k(v_k) * window(E)
            mode_list = result.mode_fcf_a_list if anharmonic else result.mode_fcf_list
            nclass = len(result.osci_list) - len(mode_list)
            index = {result.osci_list[nclass+k][0]: k for k in range(len(mode_list))}
            for st in states:
                weight = np.prod([mode_list[k][0][0] for k in range(len(mode_list))
                                  if result.osci_list[nclass+k][0] not in st.quanta])
                ebin = 0
                for frq, nq in st.quanta.items():
                    weight *= mode_list[index[frq]][nq][0]
                    ebin += int(np.floor(mode_list[index[frq]][nq][1]))
                self.assertEqual(ebin, int(st.energy))
                assert_allclose(st.contribution, weight*result.window[ebin], rtol=1e-12)
        # regression: dominant harmonic state
        st = dominant_states(result, nstates=1)[0]
        self.assertEqual({round(f): n for f, n in st.quanta.items()}, {38: 2, 168: 1, 1681: 4})
        assert_allclose(st.fraction, 6.405e-4, rtol=1e-3)
        self.assertIn(1681, freqs)

    def test_kbest_smaller_than_nstates(self):
        result = run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                low_freq_approx=0., max_e=10000., verbose=False)
        with self.assertRaises(ValueError):
            dominant_states(result, nstates=10, kbest=5)


class TestMinimalModeSet(unittest.TestCase):

    def test_toy_model(self):
        # mode 0 bridges the gap, mode 1 is a spectator; the spectator is omitted first
        mode_list = [[[0.5, 0.], [0.5, 10.]], [[0.9, 0.], [0.1, 3.]]]
        fcwd0 = np.zeros(20)
        fcwd0[0] = 1.
        window = np.zeros(20)
        window[10] = 1.
        full, omitted, values, kept = _fastfold.minimal_mode_set_fft(fcwd0, mode_list, window, 1., 0.9)
        assert_allclose(full, 0.45, atol=1e-15)
        self.assertEqual(list(omitted), [1])
        assert_allclose(values, [0.5], atol=1e-15)
        self.assertEqual(list(kept), [0])

    def test_ttm1cz(self):
        result = run_simulation(hessian_calculation=MOL_DIR, egrad_calculation=MOL_DIR,
                                i_state_gradient=False, fcwd_file=os.devnull, e_trans=7000.,
                                low_freq_approx=0., max_e=15000., verbose=False)
        mode_set = minimal_mode_set(result, fraction=0.9)
        assert_allclose(mode_set.fcwd_full, result.val_avg, rtol=1e-12)
        self.assertEqual([round(f) for f in mode_set.freqs],
                         [20, 38, 54, 151, 168, 235, 244, 330, 363, 432, 477, 559, 571, 835, 858, 940, 1010, 1081,
                          1100, 1138, 1177, 1205, 1208, 1214, 1277, 1288, 1342, 1351, 1381, 1394, 1468, 1485, 1525,
                          1656, 1681, 1719])
        self.assertGreaterEqual(mode_set.fcwd_set, 0.9*mode_set.fcwd_full)
        assert_allclose(mode_set.retained, mode_set.fcwd_set/mode_set.fcwd_full)
        assert_allclose(mode_set.omitted_retained, mode_set.omitted_values/mode_set.fcwd_full)

        # the window average of the set agrees with a fold in energy space of these modes only
        nclass = len(result.osci_list) - len(result.mode_fcf_list)
        sel = [result.mode_fcf_list[k] for k in range(len(result.mode_fcf_list))
               if result.osci_list[nclass+k][0] in set(mode_set.freqs)]
        fcwd, _ = FcfUtils.FCWD(sel, 0.).get_FCWD(result.max_e, result.e_bin)
        assert_allclose(mode_set.fcwd_set, np.sum(fcwd*result.window), rtol=1e-11)

        # omitting any further mode drops below the requested fraction
        _, omit = FcfUtils.FCWD(sel, 0.).get_mode_contributions(result.max_e, result.e_bin, result.window)
        self.assertTrue(np.all(omit < 0.9*mode_set.fcwd_full))

    def test_ic_testmolecule_classical_low_freq(self):
        result = run_simulation(hessian_calculation=DATA_DIR, egrad_calculation=DATA_DIR,
                                fcwd_file=os.devnull, e_trans=15000., low_freq_approx=300., verbose=False)
        for anharmonic in (False, True):
            mode_set = minimal_mode_set(result, fraction=0.9, anharmonic=anharmonic)
            full = result.vala_avg if anharmonic else result.val_avg
            assert_allclose(mode_set.fcwd_full, full, rtol=1e-4)
            self.assertGreaterEqual(mode_set.fcwd_set, 0.9*mode_set.fcwd_full)
            self.assertTrue(np.all(mode_set.freqs > 300.))

        out = os.path.join(os.path.dirname(__file__), "mode_analysis_test_output.dat")
        self.addCleanup(lambda: os.remove(out) if os.path.exists(out) else None)
        write_mode_set(out, mode_set)
        write_dominant_states(out, dominant_states(result, nstates=3))


if __name__ == "__main__":
    unittest.main()
