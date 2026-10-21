import warnings
from dataclasses import dataclass, field

import numpy as np

from . import fcf_utils as FcfUtils
from . import _fastfold as fastfold
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
    window: np.ndarray = field(default=None, repr=False)  # Gaussian window on the grid of fcwd (incl. e_bin)
    grid: np.ndarray = field(default=None, repr=False)    # energies (cm-1) of the window grid
    max_e: float = None             # maximum energy of the FCWD grid (cm-1)
    e_bin: float = 1.               # bin width (cm-1)
    Ereo_class: float = 0.          # reorganization energy of the classically treated modes (cm-1)
    e_trans: float = None           # center of the energy window (cm-1)
    sigma: float = None             # width of the energy window (cm-1)
    D: float = None                 # Morse dissociation energy (cm-1) of the anharmonic FCFs
    anh_thr: float = None           # modes above this frequency (cm-1) have Morse FCFs
    analysis: "FCWDAnalysis" = field(default=None, repr=False)  # FCWD analysis, if requested
    pair_analysis: "FCWDPairAnalysis" = field(default=None, repr=False)  # pair analysis, if requested


@dataclass
class FCWDAnalysis:
    """Sensitivity analysis of the FCWD (omission of single modes), see analyze_fcwd()."""
    freqs: np.ndarray               # frequencies (cm-1) of the quantum-treated modes
    fcwd_h: float                   # harmonic FCWD window average, all modes
    fcwd_a: float                   # anharmonic FCWD window average, all modes
    fcwd_h_omit: np.ndarray         # harmonic FCWD window averages with each mode omitted
    fcwd_a_omit: np.ndarray         # anharmonic FCWD window averages with each mode omitted
    retained_h: np.ndarray          # remaining harmonic FCWD with each mode omitted, fraction of fcwd_h
    retained_a: np.ndarray          # remaining anharmonic FCWD with each mode omitted, fraction of fcwd_a
    enhancement_h: np.ndarray       # enhancement factors fcwd_h / fcwd_h_omit (= 1/retained_h)
    enhancement_a: np.ndarray       # enhancement factors fcwd_a / fcwd_a_omit (= 1/retained_a)


@dataclass
class FCWDPairAnalysis:
    """Pair analysis of the FCWD (omission of pairs of modes), see analyze_fcwd_pairs()."""
    freqs: np.ndarray               # frequencies (cm-1) of the quantum-treated modes
    fcwd_h: float                   # harmonic FCWD window average, all modes
    fcwd_a: float                   # anharmonic FCWD window average, all modes
    pair_h: np.ndarray              # harmonic FCWD window averages, pair (i, j) omitted (diagonal: mode i omitted)
    pair_a: np.ndarray              # anharmonic, as pair_h
    synergy_h: np.ndarray           # synergy factors Q_ij = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j)) (diagonal: 1)
    synergy_a: np.ndarray           # anharmonic, as synergy_h
    reliable_h: np.ndarray          # True where all four window averages entering Q_ij exceed their round-off estimates
    reliable_a: np.ndarray          # anharmonic, as reliable_h
    reliability: float              # required ratio of window average to round-off estimate


@dataclass
class ModeSet:
    """Result of minimal_mode_set()."""
    fraction: float                 # requested fraction of the FCWD window average
    fcwd_full: float                # FCWD window average with all modes
    fcwd_set: float                 # FCWD window average with the modes of the set only
    retained: float                 # remaining FCWD with the modes of the set only, fraction of fcwd_full
    freqs: np.ndarray               # frequencies (cm-1) of the modes of the set
    S: np.ndarray                   # Huang-Rhys factors of the modes of the set
    omitted_freqs: np.ndarray       # frequencies of the omitted modes, in the order of omission
    omitted_values: np.ndarray      # FCWD window average after each omission
    omitted_retained: np.ndarray    # remaining FCWD after each omission, fraction of fcwd_full


@dataclass
class VibronicState:
    """Final vibrational state, see dominant_states()."""
    contribution: float             # contribution to the FCWD window average
    fraction: float                 # contribution, fraction of the FCWD window average
    energy: float                   # vibrational energy of the quantum modes (cm-1)
    quanta: dict                    # {frequency (cm-1): quantum number} of all excited quantum modes


def _read_oscillators(hessian_calculation, egrad_calculation, i_state_gradient, damp, damp_thr, verbose=True):
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

    # direction depends on whether the gradient refers to the initial or final state
    if i_state_gradient:
        dsp_sign = 1.0
    else:
        dsp_sign = -1.0


    # dsp = - grad/frq^2   (times √frq to get dimensionless coordinates)
    # for emission from state for which we have the gradient information,
    #  we actually have to reverse the sign, so + grad/freq^2 (no issue in harm. approx)
    dsp = dsp_sign * np.sqrt(freqs0_sh)*( grad.T @ Lmat ) / (freqs_sh)**2

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
            treat = "A"
        log(f"omg = {omg:8.2f} cm^-1    S = {S:10.4e}  d = {dsp:8.4f} AA   E_reo = {Ereo:8.2f} cm^-1    xi = {xi:10.4e}  {treat}")

        if omg <= low_freq_approx:
            Ereo_class += Ereo
            continue
        else:
            Ereo_quant += Ereo

        dho = FcfUtils.FcfDHO(S)

        fcf_list = []
        for ii in range(maxquanta):
            fcf = dho.get_I0n(ii)
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
            if omg <= anh_thr:
                fcf = dho.get_I0n(ii)
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


def _window(offset, max_e, e_bin, nbins, e_trans, sigma):
    """Grid energies and normalized Gaussian window (incl. bin width) centered at e_trans."""
    fval = np.linspace(offset,max_e-e_bin,nbins)
    fwin = 1./(sigma*np.sqrt(2.*np.pi))*np.exp(-0.5*(((fval-e_trans)/sigma)**2))*e_bin
    return fval, fwin


def _bin_fcwd(mode_fcf_list, mode_fcf_a_list, Ereo_class, max_e, e_bin, e_trans, sigma, verbose=True,
              fcwd_method="direct"):
    """Bins the per-mode FCFs into harmonic/anharmonic FCWDs, and averages them over
    a Gaussian energy window centered at e_trans."""
    log = print if verbose else (lambda *a, **k: None)

    fcwd_gen = FcfUtils.FCWD(mode_fcf_list,Ereo_class,debug=0,method=fcwd_method)
    fcwd,offset = fcwd_gen.get_FCWD(max_e,e_bin)

    fcwd_gen_a = FcfUtils.FCWD(mode_fcf_a_list,Ereo_class,debug=0,method=fcwd_method)
    fcwd_a,offset = fcwd_gen_a.get_FCWD(max_e,e_bin)

    nbins = len(fcwd_a)

    fval, fwin = _window(offset, max_e, e_bin, nbins, e_trans, sigma)

    log("fval: ",fval)
    log("sum(fwin):",np.sum(fwin))
    log("fwin(last)=",fwin[-1])

    val_avg = np.sum(fcwd*fwin)
    vala_avg = np.sum(fcwd_a*fwin)

    log(f"Averages: {val_avg} (H)  {vala_avg} (A)",flush=True)

    return fcwd, fcwd_a, offset, fval, fwin, val_avg, vala_avg


def _write_fcwd_file(fcwd_file, fcwd, fcwd_a, offset):
    with open(fcwd_file,"w") as outstr:
        ii = -1
        print(f"{'FREQ':>10} {'FCWDh':>20} {'FCWDa':>20}",file=outstr)
        for val,vala in zip(fcwd,fcwd_a):
            ii = ii+1
            en = ii*1. + offset
            print(f" {en:10.2f} {val:20.6e} {vala:20.6e}",file=outstr)


def analyze_fcwd(result, method="fft"):
    """
    Sensitivity analysis of the FCWD of a run_simulation() result: each quantum-treated mode
    is omitted in turn (i.e. treated as undisplaced), for the harmonic and the anharmonic FCFs.
    For each mode, the remaining FCWD without it is given as fraction of the full FCWD
    (retained_*); the enhancement factors (full FCWD / FCWD without the mode) are its inverse.
    method="fft": everything in a single call in Fourier space (_fastfold.fcwd_analysis_fft);
    method="direct": the FCWD is folded anew in energy space for each omission (slow, but
    accurate relative to each value, which matters only for FCWDs far in the tail).
    Returns an FCWDAnalysis.
    """
    mode_lists = (result.mode_fcf_list, result.mode_fcf_a_list)
    nclass = len(result.osci_list) - len(result.mode_fcf_list)
    freqs = np.array([osci[0] for osci in result.osci_list[nclass:]])
    if method == "fft":
        fcwd0, _ = FcfUtils.FCWD(result.mode_fcf_list, result.Ereo_class)._initial_fcwd(result.max_e, result.e_bin)
        full, omit, enh = fastfold.fcwd_analysis_fft(fcwd0, mode_lists, result.window, result.e_bin)
    elif method == "direct":
        full, omit = [], []
        for mode_list in mode_lists:
            fcwd, _ = FcfUtils.FCWD(mode_list, result.Ereo_class).get_FCWD(result.max_e, result.e_bin)
            full.append(np.sum(fcwd*result.window))
            values = []
            for idx in range(len(mode_list)):
                sel = mode_list[:idx] + mode_list[idx+1:]
                fcwd, _ = FcfUtils.FCWD(sel, result.Ereo_class).get_FCWD(result.max_e, result.e_bin)
                values.append(np.sum(fcwd*result.window))
            omit.append(np.array(values))
        with np.errstate(divide="ignore"):
            enh = [full[iset]/omit[iset] for iset in range(2)]
    else:
        raise ValueError(f"unknown method: {method}")
    return FCWDAnalysis(freqs=freqs, fcwd_h=full[0], fcwd_a=full[1], fcwd_h_omit=omit[0],
                        fcwd_a_omit=omit[1], retained_h=omit[0]/full[0], retained_a=omit[1]/full[1],
                        enhancement_h=enh[0], enhancement_a=enh[1])


def write_fcwd_analysis(fname, analysis, log=None):
    """Writes an FCWDAnalysis to a text file (modes in descending order of frequency);
    log (e.g. print) additionally receives each line."""
    with open(fname, "w") as outf:
        print(f"# FCWD window averages with each mode omitted, and the remaining FCWD in % of the full FCWD "
              f"(harmonic: {analysis.fcwd_h:.5e}, anharmonic: {analysis.fcwd_a:.5e})", file=outf)
        print(f"{'FREQ':>16} {'FCWDh(-k)':>16} {'FCWDh(-k) [%]':>16}  {'FCWDa(-k)':>16} {'FCWDa(-k) [%]':>16}", file=outf)
        for idx in range(len(analysis.freqs)-1, -1, -1):
            line = (f"{analysis.freqs[idx]:>16.2f} {analysis.fcwd_h_omit[idx]:>16.5e} {analysis.retained_h[idx]*100:>16.4f}  "
                    f"{analysis.fcwd_a_omit[idx]:>16.5e} {analysis.retained_a[idx]*100:>16.4f} ")
            if log is not None:
                log(line, flush=True)
            print(line, file=outf)


def analyze_fcwd_pairs(result, reliability=10.):
    """
    Pair analysis of the FCWD of a run_simulation() result, for the harmonic and the anharmonic
    FCFs: all pairs of quantum-treated modes are omitted (i.e. treated as undisplaced), in a
    single call in Fourier space (_fastfold.fcwd_pair_analysis_fft). The synergy factor
        Q_ij = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j))
    is > 1 if the modes i and j reinforce each other (together more important than separately),
    < 1 if they can partly replace each other, and 1 if they act independently.
    Q_ij is marked as reliable if each of the four window averages entering it is at least
    reliability times its (conservative) round-off estimate; Fourier-space round-off is
    absolute, so this matters for FCWDs far in the tail.
    Returns an FCWDPairAnalysis.
    """
    mode_lists = (result.mode_fcf_list, result.mode_fcf_a_list)
    nclass = len(result.osci_list) - len(result.mode_fcf_list)
    freqs = np.array([osci[0] for osci in result.osci_list[nclass:]])
    fcwd0, _ = FcfUtils.FCWD(result.mode_fcf_list, result.Ereo_class)._initial_fcwd(result.max_e, result.e_bin)
    full, full_err, pair, pair_err = fastfold.fcwd_pair_analysis_fft(fcwd0, mode_lists, result.window, result.e_bin)
    synergy, reliable = [], []
    for iset in range(2):
        single = np.diag(pair[iset])
        single_err = np.diag(pair_err[iset])
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = full[iset]*pair[iset]/np.outer(single, single)
        ok = (pair[iset] >= reliability*pair_err[iset])
        ok &= (single >= reliability*single_err)[:, None]
        ok &= (single >= reliability*single_err)[None, :]
        ok &= (full[iset] >= reliability*full_err[iset])
        np.fill_diagonal(Q, 1.)
        synergy.append(Q)
        reliable.append(ok)
    return FCWDPairAnalysis(freqs=freqs, fcwd_h=full[0], fcwd_a=full[1], pair_h=pair[0], pair_a=pair[1],
                            synergy_h=synergy[0], synergy_a=synergy[1], reliable_h=reliable[0],
                            reliable_a=reliable[1], reliability=reliability)


def strongest_pairs(pair_analysis, anharmonic=False, npairs=5, kind="replacing"):
    """
    Strongest reliable pairs of a FCWDPairAnalysis: kind="replacing" (smallest Q, modes that can
    partly replace each other) or "reinforcing" (largest Q). Returns a list of
    (frequency i, frequency j, Q_ij, remaining FCWD without both modes as fraction of the full FCWD).
    """
    Q = pair_analysis.synergy_a if anharmonic else pair_analysis.synergy_h
    ok = pair_analysis.reliable_a if anharmonic else pair_analysis.reliable_h
    pair = pair_analysis.pair_a if anharmonic else pair_analysis.pair_h
    full = pair_analysis.fcwd_a if anharmonic else pair_analysis.fcwd_h
    iu, ju = np.triu_indices(len(pair_analysis.freqs), 1)
    sel = ok[iu, ju]
    iu, ju = iu[sel], ju[sel]
    lnQ = np.log(Q[iu, ju])
    if kind == "replacing":
        order = np.argsort(lnQ)
    elif kind == "reinforcing":
        order = np.argsort(-lnQ)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return [(pair_analysis.freqs[iu[k]], pair_analysis.freqs[ju[k]], Q[iu[k], ju[k]], pair[iu[k], ju[k]]/full)
            for k in order[:npairs]]


def write_fcwd_pair_analysis(fname, pair_analysis, npairs=20):
    """
    Writes the strongest replacing and reinforcing pairs (ranked by |ln Q|) of a
    FCWDPairAnalysis to a text file, for harmonic (H) and anharmonic (A) FCFs.
    """
    with open(fname, "w") as outf:
        for anharmonic, label in ((False, "H"), (True, "A")):
            ok = pair_analysis.reliable_a if anharmonic else pair_analysis.reliable_h
            nunrel = int(np.sum(~ok[np.triu_indices(len(pair_analysis.freqs), 1)]))
            print(f"# {label}: synergy factors Q_ij = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j)) (> 1 reinforcing, "
                  f"< 1 replacing) and remaining FCWD without both modes in % of the full FCWD; "
                  f"{nunrel} pairs not reliable (round-off) and skipped", file=outf)
            for kind in ("replacing", "reinforcing"):
                print(f"# {label}, {kind} pairs", file=outf)
                print(f"{'FREQ_i':>12} {'FREQ_j':>12} {'Q':>12} {'FCWD(-i,-j) [%]':>16}", file=outf)
                for fi, fj, q, rel in strongest_pairs(pair_analysis, anharmonic, npairs, kind):
                    print(f"{fi:>12.2f} {fj:>12.2f} {q:>12.5f} {rel*100:>16.4f}", file=outf)


def imaginary_modes(hessian_calculation):
    """Returns the frequencies (rcm) of all imaginary modes (given as negative numbers) of the
    Turbomole calculation hessian_calculation. They are discarded by run_simulation()."""
    freqs, _, _ = rtm.turbomole_results(hessian_calculation).get_hessian()
    return freqs[freqs < -1e-3]


def relaxed_gradient_gap(e_vert, hessian_calculation, egrad_calculation, damp=10., damp_thr=100.):
    """
    Returns the energy (rcm) at which the FCWD has to be measured within the relaxed-gradient
    model, together with the reorganization energy lambda (rcm).
    e_vert is the electronic gap (rcm) between initial and final state at the initial-state
    geometry, at which also Hessian and gradient of the final state are computed. The FCWD is
    measured from the 0-0 line of the displaced oscillators, and the final-state minimum lies
    lambda = sum_k S_k omega_k below the final state at that geometry: e_trans = e_vert + lambda.
    damp and damp_thr have to agree with the values passed to run_simulation().
    """
    osci_list = _read_oscillators(hessian_calculation, egrad_calculation, False, damp, damp_thr, verbose=False)
    lam = sum(osci[0] * osci[1] for osci in osci_list)
    return e_vert + lam, lam


def run_simulation(
        hessian_calculation="testmolecule",
        egrad_calculation="testmolecule",
        i_state_gradient = True,    # whether gradient refers to initial state (True) or final state (False)
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
        fcwd_analysis = False,      # sensitivity analysis of the FCWD (omission of single modes), see analyze_fcwd()
        fcwd_analysis_file = "FCWD_analysis.dat",  # output file for fcwd_analysis
        fcwd_analysis_pairs = False,  # pair analysis of the FCWD (omission of pairs of modes), see analyze_fcwd_pairs()
        fcwd_analysis_pairs_file = "FCWD_pair_analysis.dat",  # output file for fcwd_analysis_pairs
        run_mode_tests = None,      # deprecated, use fcwd_analysis
        mode_contrib_file = None,   # deprecated, use fcwd_analysis_file
        verbose = True,             # print progress/diagnostics to stdout
        fcwd_method = "direct",     # "direct" (energy space) or "fft" (Fourier space) folding of the modes
        ):
    """Runs a full FCWD simulation: reads Hessian/gradient, builds per-mode harmonic and
    anharmonic (Morse) Franck-Condon factors, bins them into an FCWD, and writes it to
    fcwd_file. Returns an FCWDResult with the computed averages and intermediate data."""

    log = print if verbose else (lambda *a, **k: None)

    if run_mode_tests is not None:
        warnings.warn("run_mode_tests is deprecated, use fcwd_analysis", DeprecationWarning, stacklevel=2)
        fcwd_analysis = run_mode_tests
    if mode_contrib_file is not None:
        warnings.warn("mode_contrib_file is deprecated, use fcwd_analysis_file", DeprecationWarning, stacklevel=2)
        fcwd_analysis_file = mode_contrib_file

    log("Entered FCWD simulation\n")
    log("Settings:")
    log(f"read Hessian from:  {hessian_calculation}")
    log(f"read gradient from: {egrad_calculation}")
    if i_state_gradient:
       log("gradient refers to initial state")
    else:
       log("gradient refers to final state")
    log(f"FCWD written to:    {fcwd_file}")
    log(f"D = {D}     e_trans = {e_trans}  sigma = {sigma}")
    log(f"damp = {damp}  damp_thr = {damp_thr}  low_freq_approx = {low_freq_approx}")
    log(f"thrmod = {thrmod}  maxquanta = {maxquanta}  max_e = {max_e}   e_bin={e_bin}")
    log(f"FCWD folding: {fcwd_method}")
    if low_freq_approx > 0:
        log(f"treating modes below {low_freq_approx} cm-1 classically")

    osci_list = _read_oscillators(hessian_calculation, egrad_calculation, i_state_gradient, damp, damp_thr, verbose=verbose)

    mode_fcf_list, mode_fcf_a_list, Ereo_class, Ereo_quant = _compute_mode_fcfs(
        osci_list, D, anh_thr, low_freq_approx, thrmod, maxquanta, max_e, verbose=verbose)

    fcwd, fcwd_a, offset, fval, fwin, val_avg, vala_avg = _bin_fcwd(
        mode_fcf_list, mode_fcf_a_list, Ereo_class, max_e, e_bin, e_trans, sigma, verbose=verbose,
        fcwd_method=fcwd_method)

    _write_fcwd_file(fcwd_file, fcwd, fcwd_a, offset)

    result = FCWDResult(
        val_avg=val_avg,
        vala_avg=vala_avg,
        offset=offset,
        fcwd=fcwd,
        fcwd_a=fcwd_a,
        osci_list=osci_list,
        mode_fcf_list=mode_fcf_list,
        mode_fcf_a_list=mode_fcf_a_list,
        window=fwin,
        grid=fval,
        max_e=max_e,
        e_bin=e_bin,
        Ereo_class=Ereo_class,
        e_trans=e_trans,
        sigma=sigma,
        D=D,
        anh_thr=anh_thr,
    )

    if fcwd_analysis:
        log(f"\nFCWD analysis ({fcwd_method}): FCWD with each mode omitted, and remaining FCWD in % of the full FCWD")
        result.analysis = analyze_fcwd(result, method=fcwd_method)
        write_fcwd_analysis(fcwd_analysis_file, result.analysis, log=log)

    if fcwd_analysis_pairs:
        log(f"\nFCWD pair analysis (fft): synergy factors written to {fcwd_analysis_pairs_file}")
        result.pair_analysis = analyze_fcwd_pairs(result)
        write_fcwd_pair_analysis(fcwd_analysis_pairs_file, result.pair_analysis)

    return result


def _quantum_modes(result, anharmonic):
    """FCF lists and oscillators [freq, S, dsp, redmass] of the quantum-treated modes."""
    mode_list = result.mode_fcf_a_list if anharmonic else result.mode_fcf_list
    nclass = len(result.osci_list) - len(mode_list)
    return mode_list, result.osci_list[nclass:]


def minimal_mode_set(result, fraction=0.9, anharmonic=False):
    """
    Small set of modes that promotes the transition: starting from all quantum-treated modes
    of the run_simulation() result, the mode whose omission retains the largest FCWD window
    average is omitted, one at a time (greedy, combination effects included), as long as the
    window average stays >= fraction times that of the full FCWD.
    Omitting a mode means treating it as undisplaced. Uses the harmonic FCFs, or the
    anharmonic ones if anharmonic is True. Returns a ModeSet.
    """
    mode_list, oscis = _quantum_modes(result, anharmonic)
    gen = FcfUtils.FCWD(mode_list, result.Ereo_class)
    full, omitted, values, kept = gen.get_minimal_mode_set(result.max_e, result.e_bin, result.window, fraction)
    return ModeSet(
        fraction=fraction,
        fcwd_full=full,
        fcwd_set=values[-1] if len(values) > 0 else full,
        retained=(values[-1] if len(values) > 0 else full)/full,
        freqs=np.array([oscis[k][0] for k in kept]),
        S=np.array([oscis[k][1] for k in kept]),
        omitted_freqs=np.array([oscis[k][0] for k in omitted]),
        omitted_values=values,
        omitted_retained=values/full,
    )


def dominant_states(result, nstates=10, kbest=None, anharmonic=False):
    """
    Final vibrational states contributing most to the FCWD window average of the
    run_simulation() result (exact k-best search, kbest states kept per bin; kbest >= nstates,
    default max(nstates, 10)). Uses the harmonic FCFs, or the anharmonic ones if anharmonic
    is True. Classically treated modes enter only through their (Gaussian) energy distribution,
    which is folded into the window; the energies of the states then refer to the quantum
    modes only.
    Returns a list of VibronicState, ordered by contribution.
    """
    if kbest is None:
        kbest = max(nstates, 10)
    if kbest < nstates:
        raise ValueError("kbest has to be >= nstates")
    mode_list, oscis = _quantum_modes(result, anharmonic)
    gen = FcfUtils.FCWD(mode_list, result.Ereo_class)
    contrib, bins, quanta = gen.get_dominant_states(result.max_e, result.e_bin, result.window, kbest, nstates)
    total = result.vala_avg if anharmonic else result.val_avg
    states = []
    for cc, bb, qq in zip(contrib, bins, quanta):
        states.append(VibronicState(
            contribution=float(cc),
            fraction=float(cc/total),
            energy=float(bb*result.e_bin),
            quanta={oscis[k][0]: int(qq[k]) for k in np.nonzero(qq)[0]},
        ))
    return states


def write_mode_set(fname, mode_set):
    """Writes a ModeSet to a text file."""
    with open(fname, "w") as outf:
        print(f"# set of modes retaining {mode_set.retained*100:.1f} % of the full FCWD (requested: >= "
              f"{mode_set.fraction*100:.0f} %); FCWD window average with the set only: {mode_set.fcwd_set:.5e}, "
              f"full: {mode_set.fcwd_full:.5e}", file=outf)
        print(f"{'FREQ':>12} {'HR':>12}", file=outf)
        for frq, hr in zip(mode_set.freqs, mode_set.S):
            print(f"{frq:>12.2f} {hr:>12.4e}", file=outf)
        print(f"# omitted modes, in the order of omission, with the FCWD window average afterwards and the "
              f"remaining FCWD in % of the full FCWD", file=outf)
        for frq, val, ret in zip(mode_set.omitted_freqs, mode_set.omitted_values, mode_set.omitted_retained):
            print(f"# {frq:>10.2f} {val:>14.5e} {ret*100:>10.4f}", file=outf)


def write_dominant_states(fname, states):
    """Writes a list of VibronicState to a text file."""
    with open(fname, "w") as outf:
        print(f"# contributions of final vibrational states to the FCWD window average, and in % of the full FCWD", file=outf)
        print(f"{'CONTRIB':>12} {'[%]':>10} {'ENERGY':>10}   QUANTA (frequency(quantum number))", file=outf)
        for st in states:
            quanta = " ".join(f"{frq:.2f}({nq})" for frq, nq in st.quanta.items())
            print(f"{st.contribution:>12.4e} {st.fraction*100:>10.4f} {st.energy:>10.1f}   {quanta}", file=outf)
        print(f"# sum: {sum(st.fraction for st in states)*100:.2f} % of the full FCWD", file=outf)
