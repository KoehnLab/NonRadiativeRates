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
