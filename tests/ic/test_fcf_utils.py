#!/usr/bin/env python3

import unittest

import numpy as np
import scipy.special as scsp

from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

from nonradiative_rates import fcf_utils as FcfUtils
from nonradiative_rates.constants import au2rcm, amu, a02AA

class TestFcfUtils(unittest.TestCase):

    def test_fcf_morse(self):

        omega = 3000
        xi = 0.02
        D = omega/(4*xi)
        mred = 1
        mredau = mred*amu

        Dau = D/au2rcm
        omau = omega/au2rcm

        aau = np.sqrt(2.*omau*xi*mredau)

        dlt = -0.01
        dltau = dlt/a02AA

        m_system = FcfUtils.FcfMorse0(Dau,aau,mredau,dltau)

        max_n = m_system.max_n()
        assert_equal(max_n,24)

        int1 = []
        int2 = []
        for ii in range(4):
            int1.append(m_system.get_I0n_MI(ii))
            int2.append(m_system.get_I0n(ii))

        assert_array_almost_equal(int2,int1,6,"Comparing Morse algorithms")


    def test_morse_coordinate_integrals(self):
        # <0|n> and <0|x|n> of two displaced Morse oscillators against a numerical reference
        # (sinc-DVR eigenfunctions on a grid), for a C-H-like mode with mred = 1 amu

        omega = 3000.
        xi = 0.02
        S = 0.3
        mredau = 1.*amu
        omau = omega/au2rcm
        Dau = omau/(4*xi)
        aau = np.sqrt(2.*omau*xi*mredau)
        dltau = np.sqrt(2.*S)/np.sqrt(mredau*omau)

        m_system = FcfUtils.FcfMorse0(Dau,aau,mredau,dltau)

        x0 = 1./np.sqrt(mredau*omau)
        x = np.linspace(-5.*x0, 15.*x0, 1500)
        dx = x[1] - x[0]
        ij = np.subtract.outer(np.arange(len(x)), np.arange(len(x)))
        T = np.where(ij == 0, np.pi**2/3., 2.*(-1.)**ij/np.where(ij == 0, 1, ij)**2)/(2.*mredau*dx**2)
        _, c0 = np.linalg.eigh(T + np.diag(Dau*(1. - np.exp(-aau*x))**2))
        _, cn = np.linalg.eigh(T + np.diag(Dau*(1. - np.exp(-aau*(x - dltau)))**2))
        c0 = c0[:, 0]

        for nn in range(6):
            ovl = m_system.get_I0n(nn)
            cv = cn[:, nn]*np.sign(cn[:, nn] @ c0)*np.sign(ovl)
            assert_almost_equal(ovl, cv @ c0, 10, f"Morse <0|n>, n = {nn}")
            # in the dimensionless coordinate
            assert_almost_equal(np.sqrt(mredau*omau)*m_system.get_b0n(nn),
                                np.sqrt(mredau*omau)*(cv @ (x*c0)), 10, f"Morse <0|x|n>, n = {nn}")


    def test_morse_small_anharmonicity(self):
        # get_I0n/get_b0n do not overflow for small anharmonicities (get_I0n_ori does), agree with
        # get_I0n_ori where it is finite and approach the harmonic integrals for xi -> 0

        omega, S = 1409.2, 0.5432
        mredau = 6.*amu
        omau = omega/au2rcm
        d = np.sqrt(2.*S)
        q = np.sqrt(mredau*omau)
        nv = np.arange(10)
        s2_h = np.exp(-S)*S**nv/scsp.gamma(nv+1)
        st_h = s2_h*(S - nv)/d

        def system(xi):
            return FcfUtils.FcfMorse0(omau/(4.*xi), np.sqrt(2.*omau*xi*mredau), mredau, d/q)

        m_system = system(0.0117)
        assert_array_almost_equal([m_system.get_I0n(n) for n in nv], [m_system.get_I0n_ori(n) for n in nv], 10,
                                  "Morse <0|n> vs. direct evaluation of Eq. 16")

        for xi in (1e-3, 1e-5, 1e-7):
            m_system = system(xi)
            s = np.array([m_system.get_I0n(n) for n in nv])
            st = s*q*np.array([m_system.get_b0n(n) for n in nv])
            self.assertTrue(np.all(np.isfinite(st)))
            assert_almost_equal(np.sum(s**2), 1., 6)
            # deviations from the harmonic integrals ~ sqrt(xi)
            self.assertLess(np.max(np.abs(s**2 - s2_h)), 2.*np.sqrt(xi))
            self.assertLess(np.max(np.abs(st - st_h)), 2.*np.sqrt(xi))
            self.assertGreater(np.max(np.abs(st - st_h)), 0.2*np.sqrt(xi))
            # vibrationally averaged position of the ground state, <0|q|0> = 3/4 sqrt(2 xi) to first order
            assert_almost_equal(np.sum(st)/(0.75*np.sqrt(2.*xi)), 1., 2)


    def test_fcf_dho(self):

        S = 0.35

        dho = FcfUtils.FcfDHO(S)

        fcfs = [dho.get_I0n(ii) for ii in range(20)]

        fcfs_ref = [np.exp(-S)*S**ii/scsp.gamma(ii+1) for ii in range(20)]
        assert_array_almost_equal(fcfs,fcfs_ref,12,"Comparing DHO FCFs to the direct Poisson formula")

        assert_almost_equal(fcfs[0],np.exp(-S))
        assert_almost_equal(np.sum(fcfs),1.0,decimal=6)


    def test_FCWD_internal(self):

        modes = [[250.0,0.3],[400.0,0.1],[1100,0.25]]

        max_e = 30000.0
        e_bin = 0.25

        mode_fcf_list = []
        for mode in modes:
            omg = mode[0]
            S = mode[1]

            fcf_list = []
            for ii in range(10):
                fcf = np.exp(-S)*S**ii/scsp.gamma(ii+1)
                en = omg*(ii)
                fcf_list.append([fcf,en])

            mode_fcf_list.append(fcf_list)

        Ereo_class = 50.0
        fcwd_gen_0 = FcfUtils.FCWD(mode_fcf_list,Ereo_class,debug=0,use_cython=False)
        fcwd_gen_c = FcfUtils.FCWD(mode_fcf_list,Ereo_class,debug=0,use_cython=True)

        FCWD_0,offset_0 = fcwd_gen_0.get_FCWD(max_e,e_bin)
        FCWD_c,offset_c = fcwd_gen_c.get_FCWD(max_e,e_bin)

        assert_almost_equal(offset_c,offset_0)
        assert_array_almost_equal(FCWD_c,FCWD_0)


if __name__ == "__main__":
    unittest.main()
