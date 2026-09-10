import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.special as scsp

from . import fcf_utils as FcfUtils
from . import read_turbomole as rtm
from .constants import au2rcm, amu
from .mode_direction import get_Lmat


@dataclass
class FCWDResult:
    """Result of a run_simulation() call."""
    val_avg: float                  # harmonic FCWD, averaged over the e_trans window
    vala_avg: float                 # anharmonic (Morse) FCWD, averaged over the e_trans window
    offset: float                   # energy (cm-1) of the first bin in fcwd/fcwd_a
    fcwd: np.ndarray                # harmonic FCWD, on a grid starting at offset with spacing e_bin
    fcwd_a: np.ndarray              # anharmonic (Morse) FCWD, same grid as fcwd
    osci_list: list = field(repr=False)          # [freq, S, dsp, redmass] per mode, all modes
    mode_fcf_list: list = field(repr=False)      # per-mode harmonic FCF lists, quantum modes only
    mode_fcf_a_list: list = field(repr=False)    # per-mode anharmonic FCF lists, quantum modes only


def _read_oscillators(hessian_calculation, egrad_calculation, damp, damp_thr, verbose=True):
    """
    Reads the Hessian (with phase-corrected normal modes) and gradient from the given
    Turbomole calculations, and returns the list of oscillators [freq, S, dsp, redmass]
    describing the displacement along each non-trivial mode.
    """
    log = print if verbose else (lambda *a, **k: None)

    mol_data = rtm.turbomole_results(hessian_calculation)

    coord,elems,natoms = mol_data.get_coords()
    masses = mol_data.get_masses()
    freqs,_,redmass = mol_data.get_hessian()
    Lmat = get_Lmat(mol_data, verbose=verbose)

    if hessian_calculation == egrad_calculation:
        mol_data2 = mol_data
    else:
        mol_data2 = rtm.turbomole_results(egrad_calculation)

    grad,coord_grad = mol_data2.get_gradient()

    test = (np.array(coord)-np.array(coord_grad))**2
    rms = np.sum(test)/len(test)
    log(f"RMSD of coordinates read from coord and gradient: {rms}")

    if rms > 0.01:
        warnings.warn("RMSD between coord and gradient geometries appears large; check inputs!")

    # mass weigh the gradient
    masses = np.array(masses)
    grad = np.array(grad)
    freqs_sh = np.array(freqs)/au2rcm
    freqs0_sh = np.array(freqs)/au2rcm

    for ii in range(natoms):
        grad[ii,:] /= np.sqrt(masses[ii]*amu)

    log(f"Use damping of {damp} cm-1 for modes below {damp_thr} cm-1")
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

    return osci_list


def _compute_mode_fcfs(osci_list, D, anh_thr, low_freq_approx, thrmod, maxquanta, max_e, verbose=True):
    """
    Computes per-mode harmonic and anharmonic (Morse) FCF lists for all quantum-treated
    modes (those with freq > low_freq_approx), along with the classical/quantum
    reorganization energies.
    """
    log = print if verbose else (lambda *a, **k: None)

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
        log(f"omg = {omg:8.2f} cm^-1    S = {S:10.4e}  d = {dsp:8.4f} AA   E_reo = {Ereo:8.2f} cm^-1    xi = {xi:10.4e}  {treat}")

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
            log(f"{idx:4} ",end="")
            if idx < n_mode:
                log(f"{fcf_list[idx][1]:10.2f} {fcf_list[idx][0]:12.4e} ",end="")
            else:
                log(f"{' ':24}",end="")
            if idx < n_mode_a:
                log(f"{fcf_a_list[idx][1]:10.2f} {fcf_a_list[idx][0]:12.4e} ",end="")
            log("")

    log(f"reorganization energy in classical modes: {Ereo_class}")
    log(f"reorganization energy in quantum modes:   {Ereo_quant}")
    log(f"total reorganization energy:              {Ereo_class+Ereo_quant}")

    return mode_fcf_list, mode_fcf_a_list, Ereo_class, Ereo_quant


def _bin_fcwd(mode_fcf_list, mode_fcf_a_list, Ereo_class, max_e, e_bin, e_trans, sigma, verbose=True):
    """Bins the per-mode FCFs into harmonic/anharmonic FCWDs, and averages them over
    a Gaussian energy window centered at e_trans."""
    log = print if verbose else (lambda *a, **k: None)

    fcwd_gen = FcfUtils.FCWD(mode_fcf_list,Ereo_class,debug=2)
    fcwd,offset = fcwd_gen.get_FCWD(max_e,e_bin)

    fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list,Ereo_class,debug=2)
    fcwd_a,offset = fcwd_gen_a.get_FCWD(max_e,e_bin)

    nbins = len(fcwd_a)

    fval = np.linspace(offset,max_e-e_bin,nbins)
    fwin = 1./(sigma*np.sqrt(2.*np.pi))*np.exp(-0.5*(((fval-e_trans)/sigma)**2))*e_bin

    log("fval: ",fval)
    log("sum(fwin):",np.sum(fwin))
    log("fwin(last)=",fwin[-1])

    val_avg = np.sum(fcwd*fwin)
    vala_avg = np.sum(fcwd_a*fwin)

    log(f"Averages: {val_avg} (H)  {vala_avg} (A)",flush=True)

    return fcwd, fcwd_a, offset, fwin, val_avg, vala_avg


def _write_fcwd_file(fcwd_file, fcwd, fcwd_a, offset):
    with open(fcwd_file,"w") as outstr:
        ii = -1
        print(f"{'FREQ':>10} {'FCWDh':>20} {'FCWDa':>20}",file=outstr)
        for val,vala in zip(fcwd,fcwd_a):
            ii = ii+1
            en = ii*1. + offset
            print(f" {en:10.2f} {val:20.6e} {vala:20.6e}",file=outstr)


def _mode_contribution_tests(osci_list, mode_fcf_list, mode_fcf_a_list, Ereo_class,
                              max_e, e_bin, fwin, val_avg, vala_avg, mode_contrib_file, verbose=True):
    """
    Excludes each quantum-treated mode in turn, to gauge its contribution to the FCWD.
    osci_list also holds the classically-treated modes, which are skipped when
    building mode_fcf_list, so mode_idx has to be offset by nclass to index back into it.
    """
    log = print if verbose else (lambda *a, **k: None)

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

            log(f"Omitting idx = {mode_idx}  {osci_list[mode_idx+nclass][0]}")
            fcwd_gen = FcfUtils.FCWD(mode_fcf_list_sel,Ereo_class,debug=2)
            fcwd,_ = fcwd_gen.get_FCWD(max_e,e_bin)

            fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list_sel,Ereo_class,debug=2)
            fcwd_a,_ = fcwd_gen_a.get_FCWD(max_e,e_bin)

            val_avg_s = np.sum(fcwd*fwin)
            vala_avg_s = np.sum(fcwd_a*fwin)

            log(f"{osci_list[mode_idx+nclass][0]:>16.2f} {val_avg_s:>16.5e} {val_avg/val_avg_s:>16.6f}  {vala_avg_s:>16.5e} {vala_avg/vala_avg_s:>16.6f} ",flush=True)
            print(f"{osci_list[mode_idx+nclass][0]:>16.2f} {val_avg_s:>16.5e} {val_avg/val_avg_s:>16.6f}  {vala_avg_s:>16.5e} {vala_avg/vala_avg_s:>16.6f} ",file=outf)


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
        run_mode_tests = False,     # see _mode_contribution_tests
        mode_contrib_file = "FCWD_contr.dat",  # output file for run_mode_tests
        verbose = True,             # print progress/diagnostics to stdout
        ):
    """Runs a full FCWD simulation: reads Hessian/gradient, builds per-mode harmonic and
    anharmonic (Morse) Franck-Condon factors, bins them into an FCWD, and writes it to
    fcwd_file. Returns an FCWDResult with the computed averages and intermediate data."""

    log = print if verbose else (lambda *a, **k: None)

    log("Entered FCWD simulation\n")
    log("Settings:")
    log(f"read Hessian from:  {hessian_calculation}")
    log(f"read gradient from: {egrad_calculation}")
    log(f"FCWD written to:    {fcwd_file}")
    log(f"D = {D}     e_trans = {e_trans}  sigma = {sigma}")
    log(f"damp = {damp}  damp_thr = {damp_thr}  low_freq_approx = {low_freq_approx}")
    log(f"thrmod = {thrmod}  maxquanta = {maxquanta}  max_e = {max_e}   e_bin={e_bin}")
    if low_freq_approx > 0:
        log(f"treating modes below {low_freq_approx} cm-1 classically")

    osci_list = _read_oscillators(hessian_calculation, egrad_calculation, damp, damp_thr, verbose=verbose)

    mode_fcf_list, mode_fcf_a_list, Ereo_class, Ereo_quant = _compute_mode_fcfs(
        osci_list, D, anh_thr, low_freq_approx, thrmod, maxquanta, max_e, verbose=verbose)

    fcwd, fcwd_a, offset, fwin, val_avg, vala_avg = _bin_fcwd(
        mode_fcf_list, mode_fcf_a_list, Ereo_class, max_e, e_bin, e_trans, sigma, verbose=verbose)

    _write_fcwd_file(fcwd_file, fcwd, fcwd_a, offset)

    if run_mode_tests:
        _mode_contribution_tests(osci_list, mode_fcf_list, mode_fcf_a_list, Ereo_class,
                                  max_e, e_bin, fwin, val_avg, vala_avg, mode_contrib_file, verbose=verbose)

    return FCWDResult(
        val_avg=val_avg,
        vala_avg=vala_avg,
        offset=offset,
        fcwd=fcwd,
        fcwd_a=fcwd_a,
        osci_list=osci_list,
        mode_fcf_list=mode_fcf_list,
        mode_fcf_a_list=mode_fcf_a_list,
    )
