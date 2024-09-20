#!/usr/bin/env python3

import numpy as np
import scipy.special as scsp
from print_utilities import printMat

import FcfUtils
import read_turbomole as rtm

au2rcm = 219474.63068  # cm-1 / E_h
amu = 1./5.485799090441e-4 # me


def run_simulation(
        hessian_calculation="testmolecule",
        egrad_calculation="testmolecule",
        fcwd_file="fcwd.dat",       # output file for FCWD
        D = 30000,                  # diss. energy (for estimate of anharm.)
        e_trans = 15000,            # energy where FCWD is measured
        sigma = 100.,               # width of Gaussian energy window centered at e_trans
        thrmod = 1e-24,             # cutoff threshold for modes
        maxquanta = 200,            # max. quanta (cutoff should lead to smaller value)
        max_e = 18000.,             # maximum energy for comp. FCWD (e_trans + several sigma)
        e_bin = 1.                  # binning for FCWD
        ):


    mol_data = rtm.turbomole_results(hessian_calculation)

    coord,elems,natoms = mol_data.get_coords()
    masses = mol_data.get_masses()
    freqs,Lmat,redmass = mol_data.get_hessian()

    if hessian_calculation == egrad_calculation:
        mol_data2 = mol_data
    else:
        mol_data2 = rtm.turbomole_results(egrad_calculation)

    grad,coord_grad = mol_data2.get_gradient()

    test = (np.array(coord)-np.array(coord_grad))**2
    rms = np.sum(test)/len(test)
    print(f"RMSD of coordinates read from coord and gradient: {rms}")

    # mass weigh the gradient
    masses = np.array(masses)
    grad = np.array(grad)
    freqs_sh = np.array(freqs)/au2rcm


    for ii in range(natoms):
        grad[ii,:] /= np.sqrt(masses[ii]*amu)


    grad = np.reshape(grad,(3*natoms))

    for ii in range(natoms):
        if freqs[ii] < 1e-3:
            freqs_sh[ii] = 1e10



    # dsp = - grad/frq^2   (times √frq to get dimensionless coordinates)
    # for emission from state for which we have the gradient information,
    #  we actually have to reverse the sign, so + grad/freq^2 (no issue in harm. approx)
    dsp = np.sqrt(freqs_sh)*( grad.T @ Lmat ) / freqs_sh**2

    osci_list = []
    
    for ii in range(3*natoms):
        if freqs[ii] < 1e-3:
            continue
        osci_list.append([freqs[ii],0.5*dsp[ii]*dsp[ii],dsp[ii],redmass[ii]])

    mode_fcf_list = []
    mode_fcf_a_list = []

    Dau = D / au2rcm


    for osci in osci_list:

        omg = osci[0]
        S = osci[1]
        dsp = osci[2]

        Ereo = S*omg

        xi = omg/(4.*D)

        use_anh = "A" if xi>=7e-3 else "H"
        print(f"omg = {omg:8.2f} cm^-1    S = {S:10.4e}  d = {dsp:8.4f} AA   E_reo = {Ereo:8.2f} cm^-1    xi = {xi:10.4e}  {use_anh}")

        fcf_list = []
        for ii in range(maxquanta):
            fcf = np.exp(-S)*S**ii/scsp.gamma(ii+1)
            en = omg*(ii)

            fcf_list.append([fcf,en])

            if en > max_e:
                break

            if ii > 3:
                if fcf < fcf_list[-2][0] and fcf < fcf_list[-3][0] and fcf < thrmod:
                    break

        mode_fcf_list.append(fcf_list)

        mred = osci[3]
        aau = np.sqrt(2.*omg/au2rcm*xi*mred*amu)
        dltau = osci[2]/np.sqrt(mred*amu*omg/au2rcm)

        #print(f"args for morse: {Dau} {aau} {mred*amu} {dltau}")
        morse = FcfUtils.FcfMorse0(Dau,aau,mred*amu,dltau)

        fcf_a_list = []
        for ii in range(maxquanta):
            # for too small anharmonicity, the Morse integral code fails; use harminonic instead
            if xi < 7e-3: 
                fcf = np.exp(-S)*S**ii/scsp.gamma(ii+1)
                en = omg*(ii)
            else:
                fcf = morse.get_I0n(ii)
                if fcf is None:
                    break
                fcf = fcf**2
                en = omg*(ii) - omg*xi*(ii+0.5)**2 + omg*xi*0.25
            fcf_a_list.append([fcf,en])

            if en > max_e:
                break

            if ii > 3:
                if fcf < fcf_a_list[-2][0] and fcf < fcf_a_list[-3][0] and fcf < thrmod:
                    break

        mode_fcf_a_list.append(fcf_a_list)

        n_mode = len(fcf_list)
        n_mode_a = len(fcf_a_list)
        for idx in range(max(n_mode,n_mode_a)):
            print(f"{idx:4} ",end="")
            if idx < n_mode:
                print(f"{fcf_list[idx][1]:10.2f} {fcf_list[idx][0]:12.4e} ",end="")
            else:
                print(f"{' ':24}",end="")
            if idx < n_mode_a:
                print(f"{fcf_a_list[idx][1]:10.2f} {fcf_a_list[idx][0]:12.4e} ",end="")
            print("")


    fcwd_gen = FcfUtils.FCWD(mode_fcf_list)

    fcwd = fcwd_gen.get_FCWD(max_e,e_bin)

    fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list)

    fcwd_a = fcwd_gen_a.get_FCWD(max_e,e_bin)

    nbins = int(np.ceil(max_e/e_bin))

    fval = np.linspace(0.,max_e-e_bin,nbins)
    fwin = 1./(sigma*np.sqrt(2.*np.pi))*np.exp(-0.5*(((fval-e_trans)/sigma)**2))*e_bin

    print("fval: ",fval)
    print("sum(fwin):",np.sum(fwin))
    print("fwin(last)=",fwin[-1])

    with open(fcwd_file,"w") as outstr:
        ii = -1
        for val,vala in zip(fcwd,fcwd_a):
            ii = ii+1
            en = ii*1.  
            print(f" {en:10.2f} {val:20.6e} {vala:20.6e}",file=outstr)

    val_avg = np.sum(fcwd*fwin)
    vala_avg = np.sum(fcwd_a*fwin)

    print(f"Averages: {val_avg} {vala_avg}",flush=True)

    nmodes = len(mode_fcf_list)
    for mode_idx in range(nmodes-1,-1,-1):
        mode_fcf_list_sel = []
        mode_fcf_a_list_sel = []
        for idx in range(nmodes):
            if idx == mode_idx:
                continue
            mode_fcf_a_list_sel.append(mode_fcf_a_list[idx])
            mode_fcf_list_sel.append(mode_fcf_list[idx])

        print(f"Omitting idx = {mode_idx}  {osci_list[mode_idx][0]}")
        fcwd_gen = FcfUtils.FCWD(mode_fcf_list_sel)
        fcwd = fcwd_gen.get_FCWD(max_e,e_bin)

        fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list_sel)
        fcwd_a = fcwd_gen_a.get_FCWD(max_e,e_bin)

        val_avg_s = np.sum(fcwd*fwin)
        vala_avg_s = np.sum(fcwd_a*fwin)

        print(f"Averages: {val_avg_s:16.5e} {val_avg/val_avg_s:10.6f}    {vala_avg_s:16.5e} {vala_avg/vala_avg_s:10.6f} ",flush=True)


def main():
    run_simulation()

if __name__ == "__main__":
    main()
