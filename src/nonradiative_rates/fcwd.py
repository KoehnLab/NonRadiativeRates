import numpy as np
import scipy.special as scsp

from . import fcf_utils as FcfUtils
from . import read_turbomole as rtm
from .mode_direction import get_Lmat

au2rcm = 219474.63068  # cm-1 / E_h
amu = 1./5.485799090441e-4 # me


def run_simulation(
        hessian_calculation="testmolecule",
        egrad_calculation="testmolecule",
        fcwd_file="fcwd.dat",       # output file for FCWD
        D = 30000,                  # diss. energy (for estimate of anharm.)
        anh_thr = 400.,             # use anh. approx for modes larger than this
        e_trans = 15000,            # energy where FCWD is measured
        sigma = 100.,               # width of Gaussian energy window centered at e_trans
        damp = 10.,                 # damping for low-frequency modes
        damp_thr = 100.,            # threshold for low-frequency modes
        low_freq_approx = 0.,       # classical approx. for modes < this val.
        thrmod = 1e-24,             # cutoff threshold for modes
        maxquanta = 200,            # max. quanta (cutoff should lead to smaller value)
        max_e = 18000.,             # maximum energy for comp. FCWD (e_trans + several sigma)
        e_bin = 1.,                 # binning for FCWD
        run_mode_tests = False,     # see end of this routine
        mode_contrib_file = "FCWD_contr.dat",  # output file for run_mode_tests
        ):


    print("Entered FCWD simulation\n")
    print("Settings:")
    print(f"read Hessian from:  {hessian_calculation}")
    print(f"read gradient from: {egrad_calculation}")
    print(f"FCWD written to:    {fcwd_file}")
    print(f"D = {D}     e_trans = {e_trans}  sigma = {sigma}")
    print(f"damp = {damp}  damp_thr = {damp_thr}  low_freq_approx = {low_freq_approx}")
    print(f"thrmod = {thrmod}  maxquanta = {maxquanta}  max_e = {max_e}   e_bin={e_bin}")

    mol_data = rtm.turbomole_results(hessian_calculation)

    coord,elems,natoms = mol_data.get_coords()
    masses = mol_data.get_masses()
    freqs,Lmat,redmass = mol_data.get_hessian()
    Lmat = get_Lmat(mol_data)

    if hessian_calculation == egrad_calculation:
        mol_data2 = mol_data
    else:
        mol_data2 = rtm.turbomole_results(egrad_calculation)

    grad,coord_grad = mol_data2.get_gradient()

    test = (np.array(coord)-np.array(coord_grad))**2
    rms = np.sum(test)/len(test)
    print(f"RMSD of coordinates read from coord and gradient: {rms}")

    if rms > 0.01:
        print("WARNING: RMSD appears to be large! Check inputs!")

    # mass weigh the gradient
    masses = np.array(masses)
    grad = np.array(grad)
    freqs_sh = np.array(freqs)/au2rcm
    freqs0_sh = np.array(freqs)/au2rcm


    for ii in range(natoms):
        grad[ii,:] /= np.sqrt(masses[ii]*amu)

    if low_freq_approx > 0:
        print(f"treating modes below {low_freq_approx} cm-1 classically")

    print(f"Use damping of {damp} cm-1 for modes below {damp_thr} cm-1")
    damp_au = damp/au2rcm

    grad = np.reshape(grad,(3*natoms))

    for ii in range(natoms):
        if freqs[ii] < 1e-3:
            freqs_sh[ii] = 1e10
        if freqs[ii] < damp_thr:
            freqs_sh[ii] += damp_au


    # dsp = - grad/frq^2   (times √frq to get dimensionless coordinates)
    # for emission from state for which we have the gradient information,
    #  we actually have to reverse the sign, so + grad/freq^2 (no issue in harm. approx)
    dsp = np.sqrt(freqs0_sh)*( grad.T @ Lmat ) / (freqs_sh)**2

    osci_list = []

    for ii in range(3*natoms):
        if freqs[ii] < 1e-3:
            continue
        osci_list.append([freqs[ii],0.5*dsp[ii]*dsp[ii],dsp[ii],redmass[ii]])

    mode_fcf_list = []
    mode_fcf_a_list = []

    Dau = D / au2rcm

    Ereo_class = 0.
    Ereo_quant = 0.

    for osci in osci_list:

        omg = osci[0]
        S = osci[1]
        dsp = osci[2]

        Ereo = S*omg

        xi = omg/(4.*D)

        if omg <= low_freq_approx:
            treat = "C"
        elif omg <= anh_thr:
            treat = "H"
        else:
            treat = "A" if xi>=7e-3 else "H"
        print(f"omg = {omg:8.2f} cm^-1    S = {S:10.4e}  d = {dsp:8.4f} AA   E_reo = {Ereo:8.2f} cm^-1    xi = {xi:10.4e}  {treat}")

        if omg <= low_freq_approx:
            Ereo_class += Ereo
            continue
        else:
            Ereo_quant += Ereo

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
            if omg <= anh_thr or xi < 7e-3:
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

    print(f"reorganization energy in classical modes: {Ereo_class}")
    print(f"reorganization energy in quantum modes:   {Ereo_quant}")
    print(f"total reorganization energy:              {Ereo_class+Ereo_quant}")

    fcwd_gen = FcfUtils.FCWD(mode_fcf_list,Ereo_class,debug=2)

    fcwd,offset = fcwd_gen.get_FCWD(max_e,e_bin)

    fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list,Ereo_class,debug=2)

    fcwd_a,offset = fcwd_gen_a.get_FCWD(max_e,e_bin)

    nbins = len(fcwd_a)

    fval = np.linspace(offset,max_e-e_bin,nbins)
    fwin = 1./(sigma*np.sqrt(2.*np.pi))*np.exp(-0.5*(((fval-e_trans)/sigma)**2))*e_bin

    print("fval: ",fval)
    print("sum(fwin):",np.sum(fwin))
    print("fwin(last)=",fwin[-1])

    with open(fcwd_file,"w") as outstr:
        ii = -1
        print(f"{'FREQ':>10} {'FCWDh':>20} {'FCWDa':>20}",file=outstr)
        for val,vala in zip(fcwd,fcwd_a):
            ii = ii+1
            en = ii*1. + offset
            print(f" {en:10.2f} {val:20.6e} {vala:20.6e}",file=outstr)

    val_avg = np.sum(fcwd*fwin)
    vala_avg = np.sum(fcwd_a*fwin)

    print(f"Averages: {val_avg} (H)  {vala_avg} (A)",flush=True)

    if run_mode_tests:

        # exclude individual modes one at a time, to gauge their contribution to the FCWD;
        # osci_list also holds the classically-treated modes, which are skipped when
        # building mode_fcf_list, so mode_idx has to be offset by nclass to index back into it
        nmodes = len(mode_fcf_list)
        nclass = len(osci_list) - len(mode_fcf_list)
        with open(mode_contrib_file, "w") as outf:
            print(f"{'FREQ':>16} {'FCWDh':>16} {'FCWDh rel':>16}  {'FCWDa':>16} {'FCWDa rel':>16}",file=outf)
            for mode_idx in range(nmodes-1,-1,-1):
                mode_fcf_list_sel = []
                mode_fcf_a_list_sel = []
                for idx in range(nmodes):
                    if idx == mode_idx:
                        continue
                    mode_fcf_a_list_sel.append(mode_fcf_a_list[idx])
                    mode_fcf_list_sel.append(mode_fcf_list[idx])

                print(f"Omitting idx = {mode_idx}  {osci_list[mode_idx+nclass][0]}")
                fcwd_gen = FcfUtils.FCWD(mode_fcf_list_sel,Ereo_class,debug=2)
                fcwd,offset = fcwd_gen.get_FCWD(max_e,e_bin)

                fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list_sel,Ereo_class,debug=2)
                fcwd_a,offset = fcwd_gen_a.get_FCWD(max_e,e_bin)

                val_avg_s = np.sum(fcwd*fwin)
                vala_avg_s = np.sum(fcwd_a*fwin)

                print(f"{osci_list[mode_idx+nclass][0]:>16.2f} {val_avg_s:>16.5e} {val_avg/val_avg_s:>16.6f}  {vala_avg_s:>16.5e} {vala_avg/vala_avg_s:>16.6f} ",flush=True)
                print(f"{osci_list[mode_idx+nclass][0]:>16.2f} {val_avg_s:>16.5e} {val_avg/val_avg_s:>16.6f}  {vala_avg_s:>16.5e} {vala_avg/vala_avg_s:>16.6f} ",file=outf)

    return val_avg, vala_avg, osci_list, mode_fcf_list, mode_fcf_a_list
