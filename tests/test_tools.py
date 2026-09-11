#!/usr/bin/env python3

import unittest

from numpy.testing import assert_almost_equal

from nonradiative_rates.tools import non_radiative


class TestNonRadiative(unittest.TestCase):

    def test_reference_value(self):
        # regression value, captured from the current implementation
        assert_almost_equal(non_radiative(100.0, 1e-12), 5917.665929406261)

    def test_zero_nac_gives_zero_rate(self):
        assert_almost_equal(non_radiative(0.0, 1e-12), 0.0)

    def test_zero_fcwd_gives_zero_rate(self):
        assert_almost_equal(non_radiative(100.0, 0.0), 0.0)

    def test_scales_quadratically_with_nac(self):
        knr = non_radiative(100.0, 1e-12)
        knr_2x = non_radiative(200.0, 1e-12)
        assert_almost_equal(knr_2x, 4 * knr)

    def test_scales_linearly_with_fcwd(self):
        knr = non_radiative(100.0, 1e-12)
        knr_2x = non_radiative(100.0, 2e-12)
        assert_almost_equal(knr_2x, 2 * knr)


if __name__ == "__main__":
    unittest.main()
