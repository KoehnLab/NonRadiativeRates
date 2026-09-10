#!/usr/bin/env python3

import os
import unittest

from numpy.testing import assert_almost_equal

from nonradiative_rates.fcwd import run_simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "testmolecule")


class TestFCWD(unittest.TestCase):

    def test_run_simulation(self):

        fcwd_file = os.path.join(os.path.dirname(__file__), "fcwd_test_output.dat")
        self.addCleanup(lambda: os.remove(fcwd_file) if os.path.exists(fcwd_file) else None)

        result = run_simulation(
            hessian_calculation=DATA_DIR,
            egrad_calculation=DATA_DIR,
            fcwd_file=fcwd_file,
            verbose=False,
        )

        assert_almost_equal(result.val_avg, 1.7603389356342848e-14)
        assert_almost_equal(result.vala_avg, 1.901868966591623e-11)
        assert_almost_equal(result.offset, 0.0)
        self.assertEqual(len(result.osci_list), 120)
        self.assertEqual(len(result.mode_fcf_list), 120)
        self.assertEqual(len(result.mode_fcf_a_list), 120)


if __name__ == "__main__":
    unittest.main()
