import numpy as np
cimport cython
from libc.math cimport floor

DYTPE=np.float64

def fold(fcwd,fcf_list,double dnu):

    # put integrals on sparse list according to graining
    values = []
    mult = []
    multscr = []

    cdef Py_ssize_t nbins = len(fcwd)

    for fcf in fcf_list:
        values.append(fcf[0])
        mult.append(int(np.floor(fcf[1]/dnu)))
        multscr.append(0)
        # may add screening here
        # should check that values on mult are not identical (too large grid)

    cdef Py_ssize_t nmult = len(mult)

    mult = np.array(mult)
    multscr = np.array(multscr,dtype=np.int64)

    cdef long[:] mult_view = mult
    cdef long[:] multscr_view = multscr

    scr   = np.zeros((nmult))
    fcwdn = np.zeros((nbins))
    values = np.array(values)

    cdef double[:] fcwd_view = fcwd
    cdef double[:] fcwdn_view = fcwdn
    cdef double[:] scr_view = scr
    cdef double[:] values_view = values

    cdef Py_ssize_t ii, jj, kk, jjmax

    jjmax=0
    # multiply the values on the sparse list onto the current FCWD
    for ii in range(nbins):

        kk = jjmax
        for jj in range(jjmax+1,nmult):
            if mult_view[jj] <= ii:
                kk+=1
            else:
                break
        jjmax = kk

        for jj in range(nmult):
            multscr_view[jj] = ii-mult_view[jj]

        for jj in range(jjmax+1):
            scr_view[jj] = fcwd_view[multscr_view[jj]]
        #scr = fcwd[multscr[0:jjmax+1]]

        #fcwdn_view[ii] = np.dot(scr[0:jjmax+1],values[0:jjmax+1])
        fcwdn_view[ii] = 0.
        for jj in range(jjmax+1):
            fcwdn_view[ii] += scr_view[jj]*values_view[jj]
        

    return fcwdn


def fold_fft(fcwd, mode_list, double dnu, Py_ssize_t pad=2):
    """
    Folds all FCF lists of mode_list onto fcwd in Fourier space: the FCWD is the
    convolution of the (binned) FCF progressions of all modes, i.e. the product of
    their Fourier transforms.

    As in fold(), the FCFs are put on the grid with spacing dnu (bin floor(E/dnu)).
    FCFs in bins >= nbins cannot contribute to bins < nbins (all shifts are positive)
    and are dropped, which is exact. The transforms are zero-padded to at least
    pad*nbins points (next power of two) to suppress the wrap-around of the circular
    convolution; the result agrees with fold() up to floating-point round-off.
    Note that this round-off is absolute (~1e-17 per bin for a normalized FCWD), whereas
    fold() is accurate relative to each value, which matters for FCWDs far in the tail.
    """
    cdef Py_ssize_t nbins = len(fcwd)
    cdef Py_ssize_t nfft = 1
    while nfft < pad*nbins:
        nfft *= 2

    stick = np.zeros((nfft))
    cdef double[:] stick_view = stick
    cdef Py_ssize_t mm
    cdef double en

    spec = np.fft.rfft(np.asarray(fcwd, dtype=np.float64), nfft)
    for fcf_list in mode_list:
        stick_view[:] = 0.
        for fcf in fcf_list:
            en = fcf[1]
            mm = <Py_ssize_t>floor(en/dnu)
            if 0 <= mm < nbins:
                stick_view[mm] += fcf[0]
        spec *= np.fft.rfft(stick)

    return np.fft.irfft(spec, nfft)[:nbins]


def mode_contributions_fft(fcwd, mode_list, window, double dnu, Py_ssize_t pad=2, Py_ssize_t block=1024):
    """
    Sensitivity analysis of the FCWD for one set of FCF progressions: returns the window
    average sum_E FCWD(E) window(E) of the full FCWD and, for each mode j, of the FCWD with
    mode j omitted (array of length len(mode_list)). See fcwd_analysis_fft().
    """
    full, omit, _ = fcwd_analysis_fft(fcwd, [mode_list], window, dnu, pad, block)
    return full[0], omit[0]


@cython.boundscheck(False)
@cython.wraparound(False)
def fcwd_analysis_fft(fcwd, mode_lists, window, double dnu, Py_ssize_t pad=2, Py_ssize_t block=1024):
    """
    Complete sensitivity analysis of the FCWD in a single call, for one or several sets of
    FCF progressions (e.g. harmonic and anharmonic) sharing the initial fcwd and the window.
    For each set, returns the window average sum_E FCWD(E) window(E) of the full FCWD, the
    window averages with each mode j omitted, and the enhancement factors
    FCWD / FCWD(without mode j).

    Everything is done in Fourier space: the FCWD is the product of the transforms H_k
    of the (binned) FCF progressions of all modes and of the initial FCWD; omitting
    mode j corresponds to the product of the transforms of all modes before j (prefix)
    and after j (suffix), so no division by H_j is needed. The window averages follow
    from Parseval's theorem, so no inverse transform is needed.
    The H_k are evaluated block-wise directly from the FCF sticks (with a table of the
    exact phase factors), so apart from that table (nfft complex numbers) the memory
    demand is ~ (total number of modes) x block. Binning, the dropping of FCFs in
    bins >= nbins and the zero padding are as in fold_fft(); the results agree with folds
    in energy space up to round-off (absolute, cf. fold_fft()).

    Returns full (array, one value per set), and lists (one entry per set) of arrays with
    the omitted-mode window averages and the enhancement factors.
    """
    cdef Py_ssize_t nbins = len(fcwd)
    if len(window) != nbins:
        raise ValueError("window and fcwd have to be defined on the same grid")
    cdef Py_ssize_t nfft = 1
    while nfft < pad*nbins:
        nfft *= 2
    cdef Py_ssize_t nfreq = nfft//2 + 1
    cdef Py_ssize_t nsets = len(mode_lists)

    # modes of all sets, set s owns modes mstart[s] ... mstart[s+1]-1
    all_modes = [mode for mode_list in mode_lists for mode in mode_list]
    mstart = np.zeros(nsets+1, dtype=np.intp)
    for iset, mode_list in enumerate(mode_lists):
        mstart[iset+1] = mstart[iset] + len(mode_list)
    cdef Py_ssize_t[:] mstart_v = mstart
    cdef Py_ssize_t ntot = len(all_modes)
    cdef Py_ssize_t nmax = max([len(mode_list) for mode_list in mode_lists] + [0])

    offs, bins, vals, _ = _sticks(all_modes, dnu, nbins)
    cdef Py_ssize_t[:] offs_v = offs
    cdef Py_ssize_t[:] bins_v = bins
    cdef double[:] vals_v = vals

    # base of all products: transform of the initial FCWD, times the conjugate transform
    # of the window and the weights of the one-sided (rfft) Parseval sum
    base = np.fft.rfft(np.asarray(fcwd, dtype=np.float64), nfft) \
         * np.conj(np.fft.rfft(np.asarray(window, dtype=np.float64), nfft))
    base[1:nfreq-1] *= 2.
    cdef double complex[:] base_v = base

    # exact phase factors exp(-2 pi i r / nfft), indexed by (bin * frequency) mod nfft
    twiddle = np.exp(-2j*np.pi*np.arange(nfft)/nfft)
    cdef double complex[:] tw_v = twiddle

    cdef double complex[:, :] H = np.empty((ntot, block), dtype=np.complex128)
    cdef double complex[:, :] pre = np.empty((nmax+1, block), dtype=np.complex128)
    cdef double complex[:] suf = np.empty(block, dtype=np.complex128)
    full = np.zeros(nsets)
    omit = np.zeros(ntot)
    cdef double[:] full_v = full
    cdef double[:] omit_v = omit
    cdef double acc, vv
    cdef Py_ssize_t t0, nt, tt, kk, ss, jj, idx, step, iset2, m0, nm

    for t0 in range(0, nfreq, block):
        nt = min(block, nfreq-t0)

        # transforms of the FCF progressions of all modes on this block of frequencies
        for kk in range(ntot):
            for tt in range(nt):
                H[kk, tt] = 0.
            for ss in range(offs_v[kk], offs_v[kk+1]):
                step = bins_v[ss]
                idx = (step*t0) % nfft
                vv = vals_v[ss]
                for tt in range(nt):
                    H[kk, tt] = H[kk, tt] + vv*tw_v[idx]
                    idx += step
                    if idx >= nfft:
                        idx -= nfft

        for iset2 in range(nsets):
            m0 = mstart_v[iset2]
            nm = mstart_v[iset2+1] - m0

            # prefix products
            for tt in range(nt):
                pre[0, tt] = base_v[t0+tt]
            for kk in range(nm):
                for tt in range(nt):
                    pre[kk+1, tt] = pre[kk, tt]*H[m0+kk, tt]
            for tt in range(nt):
                full_v[iset2] += pre[nm, tt].real

            # suffix products, accumulated backwards: omitting mode jj gives prefix(jj)*suffix(jj)
            for tt in range(nt):
                suf[tt] = 1.
            for jj in range(nm-1, -1, -1):
                acc = 0.
                for tt in range(nt):
                    acc += (pre[jj, tt]*suf[tt]).real
                    suf[tt] = suf[tt]*H[m0+jj, tt]
                omit_v[m0+jj] += acc

    full /= nfft
    omit /= nfft
    omits = [omit[mstart[iset]:mstart[iset+1]] for iset in range(nsets)]
    with np.errstate(divide="ignore"):
        enhancement = [full[iset]/omits[iset] for iset in range(nsets)]
    return full, omits, enhancement


@cython.boundscheck(False)
@cython.wraparound(False)
def fcwd_pair_analysis_fft(fcwd, mode_lists, window, double dnu, Py_ssize_t pad=2, Py_ssize_t block=512):
    """
    Pair analysis of the FCWD in a single call, for one or several sets of FCF progressions
    sharing the initial fcwd and the window: window averages sum_E FCWD(E) window(E) of the
    full FCWD, of the FCWD with each mode omitted, and of the FCWD with each pair of modes
    omitted, together with estimates of their absolute round-off errors.

    As in fcwd_analysis_fft(), everything is done in Fourier space: omitting modes i < j
    corresponds to prefix(i) * H_(i+1) ... H_(j-1) * suffix(j), built up for fixed i by
    stepping j upwards (no division); window averages follow from Parseval's theorem.
    Cost ~ (number of modes)^2/2 x nfft/2 complex multiply-adds per set; memory ~ nfft +
    3 x (number of modes) x block complex numbers.
    The error estimate of a window average is eps x (number of modes) x sum_t |term_t| / nfft
    (the Fourier-space round-off is absolute, cf. fold_fft()).

    Returns, per set (lists): full, its error estimate, and matrices (modes x modes, symmetric)
    with the window averages for omitted pairs (diagonal: single modes omitted) and their
    error estimates.
    """
    cdef Py_ssize_t nbins = len(fcwd)
    if len(window) != nbins:
        raise ValueError("window and fcwd have to be defined on the same grid")
    cdef Py_ssize_t nfft = 1
    while nfft < pad*nbins:
        nfft *= 2
    cdef Py_ssize_t nfreq = nfft//2 + 1
    cdef Py_ssize_t nsets = len(mode_lists)

    all_modes = [mode for mode_list in mode_lists for mode in mode_list]
    mstart = np.zeros(nsets+1, dtype=np.intp)
    for iset, mode_list in enumerate(mode_lists):
        mstart[iset+1] = mstart[iset] + len(mode_list)
    cdef Py_ssize_t[:] mstart_v = mstart
    cdef Py_ssize_t ntot = len(all_modes)
    cdef Py_ssize_t nmax = max([len(mode_list) for mode_list in mode_lists] + [0])

    offs, bins, vals, _ = _sticks(all_modes, dnu, nbins)
    cdef Py_ssize_t[:] offs_v = offs
    cdef Py_ssize_t[:] bins_v = bins
    cdef double[:] vals_v = vals

    base = np.fft.rfft(np.asarray(fcwd, dtype=np.float64), nfft) \
         * np.conj(np.fft.rfft(np.asarray(window, dtype=np.float64), nfft))
    base[1:nfreq-1] *= 2.
    cdef double complex[:] base_v = base

    twiddle = np.exp(-2j*np.pi*np.arange(nfft)/nfft)
    cdef double complex[:] tw_v = twiddle

    cdef double complex[:, :] H = np.empty((ntot, block), dtype=np.complex128)
    cdef double complex[:, :] pre = np.empty((nmax+1, block), dtype=np.complex128)
    cdef double complex[:, :] suf = np.empty((nmax+1, block), dtype=np.complex128)
    cdef double complex[:] run = np.empty(block, dtype=np.complex128)

    full = np.zeros(nsets)
    full_abs = np.zeros(nsets)
    pair = np.zeros((ntot, nmax))
    pair_abs = np.zeros((ntot, nmax))
    cdef double[:] full_v = full
    cdef double[:] full_abs_v = full_abs
    cdef double[:, :] pair_v = pair
    cdef double[:, :] pair_abs_v = pair_abs
    cdef double complex term
    cdef double acc, acc_abs, vv
    cdef Py_ssize_t t0, nt, tt, kk, ss, ii, jj, idx, step, iset2, m0, nm

    for t0 in range(0, nfreq, block):
        nt = min(block, nfreq-t0)

        for kk in range(ntot):
            for tt in range(nt):
                H[kk, tt] = 0.
            for ss in range(offs_v[kk], offs_v[kk+1]):
                step = bins_v[ss]
                idx = (step*t0) % nfft
                vv = vals_v[ss]
                for tt in range(nt):
                    H[kk, tt] = H[kk, tt] + vv*tw_v[idx]
                    idx += step
                    if idx >= nfft:
                        idx -= nfft

        for iset2 in range(nsets):
            m0 = mstart_v[iset2]
            nm = mstart_v[iset2+1] - m0

            # prefix products pre[k] = base H_0 ... H_(k-1), suffix products suf[k] = H_(k+1) ... H_(nm-1)
            for tt in range(nt):
                pre[0, tt] = base_v[t0+tt]
            for kk in range(nm):
                for tt in range(nt):
                    pre[kk+1, tt] = pre[kk, tt]*H[m0+kk, tt]
            if nm > 0:
                for tt in range(nt):
                    suf[nm-1, tt] = 1.
            for kk in range(nm-2, -1, -1):
                for tt in range(nt):
                    suf[kk, tt] = suf[kk+1, tt]*H[m0+kk+1, tt]

            for tt in range(nt):
                full_v[iset2] += pre[nm, tt].real
                full_abs_v[iset2] += abs(pre[nm, tt].real)

            for ii in range(nm):
                # single mode ii omitted
                acc = 0.
                acc_abs = 0.
                for tt in range(nt):
                    term = pre[ii, tt]*suf[ii, tt]
                    acc += term.real
                    acc_abs += abs(term.real)
                pair_v[m0+ii, ii] += acc
                pair_abs_v[m0+ii, ii] += acc_abs
                # pairs ii < jj omitted: run = pre[ii] H_(ii+1) ... H_(jj-1)
                for tt in range(nt):
                    run[tt] = pre[ii, tt]
                for jj in range(ii+1, nm):
                    acc = 0.
                    acc_abs = 0.
                    for tt in range(nt):
                        term = run[tt]*suf[jj, tt]
                        acc += term.real
                        acc_abs += abs(term.real)
                        run[tt] = run[tt]*H[m0+jj, tt]
                    pair_v[m0+ii, jj] += acc
                    pair_abs_v[m0+ii, jj] += acc_abs

    eps = np.finfo(np.float64).eps
    full_out, full_err, pair_out, pair_err = [], [], [], []
    for iset in range(nsets):
        m0 = mstart[iset]
        nm = mstart[iset+1] - m0
        pm = pair[m0:m0+nm, :nm]/nfft
        pe = pair_abs[m0:m0+nm, :nm]/nfft*eps*max(nm, 1)
        pm = np.triu(pm) + np.triu(pm, 1).T
        pe = np.triu(pe) + np.triu(pe, 1).T
        full_out.append(full[iset]/nfft)
        full_err.append(full_abs[iset]/nfft*eps*max(nm, 1))
        pair_out.append(pm)
        pair_err.append(pe)
    return full_out, full_err, pair_out, pair_err


def _sticks(mode_list, double dnu, Py_ssize_t nbins):
    """
    FCF sticks of all modes in compressed form: mode k owns sticks offs[k] ... offs[k+1]-1,
    with bin floor(E/dnu), value and quantum number (position in the FCF list);
    sticks in bins outside 0 ... nbins-1 are dropped (cf. fold_fft()).
    """
    cdef double en
    cdef Py_ssize_t mm
    offs = np.zeros(len(mode_list)+1, dtype=np.intp)
    bins = []
    vals = []
    quanta = []
    for kk, fcf_list in enumerate(mode_list):
        for vv, fcf in enumerate(fcf_list):
            en = fcf[1]
            mm = <Py_ssize_t>floor(en/dnu)
            if 0 <= mm < nbins:
                bins.append(mm)
                vals.append(fcf[0])
                quanta.append(vv)
        offs[kk+1] = len(bins)
    return (offs, np.array(bins, dtype=np.intp).reshape(-1),
            np.array(vals, dtype=np.float64).reshape(-1), np.array(quanta, dtype=np.intp).reshape(-1))


@cython.boundscheck(False)
@cython.wraparound(False)
def minimal_mode_set_fft(fcwd, mode_list, window, double dnu, double fraction=0.9, Py_ssize_t pad=2):
    """
    Greedy search for a small set of modes that promotes the transition: starting from all
    modes, the mode whose omission retains the largest window average sum_E FCWD(E) window(E)
    is omitted, one at a time, as long as the window average stays >= fraction times that
    of the full FCWD. Combination effects are included, since each step evaluates the FCWD
    of all remaining modes together.

    Done in Fourier space as in mode_contributions_fft(); the transforms of all modes are
    computed once and kept (memory ~ 2 x (number of modes) x nfft/2 complex numbers).
    Returns the window average of the full FCWD, the omitted modes (indices into mode_list,
    in the order of omission), the window averages after each omission, and the remaining
    modes (the set).
    """
    cdef Py_ssize_t nbins = len(fcwd)
    if len(window) != nbins:
        raise ValueError("window and fcwd have to be defined on the same grid")
    cdef Py_ssize_t nfft = 1
    while nfft < pad*nbins:
        nfft *= 2
    cdef Py_ssize_t nfreq = nfft//2 + 1
    cdef Py_ssize_t nmodes = len(mode_list)

    offs, bins, vals, _ = _sticks(mode_list, dnu, nbins)
    cdef Py_ssize_t[:] offs_v = offs
    cdef Py_ssize_t[:] bins_v = bins
    cdef double[:] vals_v = vals

    base = np.fft.rfft(np.asarray(fcwd, dtype=np.float64), nfft) \
         * np.conj(np.fft.rfft(np.asarray(window, dtype=np.float64), nfft))
    base[1:nfreq-1] *= 2.
    cdef double complex[:] base_v = base

    twiddle = np.exp(-2j*np.pi*np.arange(nfft)/nfft)
    cdef double complex[:] tw_v = twiddle

    # transforms of the FCF progressions of all modes
    cdef double complex[:, :] H = np.zeros((nmodes, nfreq), dtype=np.complex128)
    cdef Py_ssize_t kk, ss, tt, idx, step, nn, jj
    cdef double vv
    for kk in range(nmodes):
        for ss in range(offs_v[kk], offs_v[kk+1]):
            step = bins_v[ss]
            vv = vals_v[ss]
            idx = 0
            for tt in range(nfreq):
                H[kk, tt] = H[kk, tt] + vv*tw_v[idx]
                idx += step
                if idx >= nfft:
                    idx -= nfft

    cdef double complex[:, :] pre = np.empty((nmodes+1, nfreq), dtype=np.complex128)
    cdef double complex[:] suf = np.empty(nfreq, dtype=np.complex128)
    loo = np.zeros(nmodes)
    cdef double[:] loo_v = loo
    cdef double acc, full

    # window average of the full FCWD
    full = 0.
    for tt in range(nfreq):
        pre[0, tt] = base_v[tt]
    for kk in range(nmodes):
        for tt in range(nfreq):
            pre[kk+1, tt] = pre[kk, tt]*H[kk, tt]
    for tt in range(nfreq):
        full += pre[nmodes, tt].real
    full /= nfft

    active = list(range(nmodes))
    omitted = []
    values = []
    cdef Py_ssize_t[:] act_v
    while len(active) > 0:
        nn = len(active)
        act_v = np.array(active, dtype=np.intp)
        for tt in range(nfreq):
            pre[0, tt] = base_v[tt]
        for kk in range(nn):
            for tt in range(nfreq):
                pre[kk+1, tt] = pre[kk, tt]*H[act_v[kk], tt]
        for tt in range(nfreq):
            suf[tt] = 1.
        for jj in range(nn-1, -1, -1):
            acc = 0.
            for tt in range(nfreq):
                acc += (pre[jj, tt]*suf[tt]).real
                suf[tt] = suf[tt]*H[act_v[jj], tt]
            loo_v[jj] = acc/nfft
        jj = int(np.argmax(loo[:nn]))
        if loo_v[jj] < fraction*full:
            break
        omitted.append(active.pop(jj))
        values.append(loo_v[jj])

    return full, np.array(omitted, dtype=np.intp), np.array(values), np.array(active, dtype=np.intp)


@cython.boundscheck(False)
@cython.wraparound(False)
def dominant_states(fcwd, mode_list, window, double dnu, Py_ssize_t kbest=10, Py_ssize_t nstates=10):
    """
    Dominant final vibrational states within the window: the modes are folded in energy
    space as in fold(), but instead of summing, the kbest largest products of FCFs are kept
    in each bin, together with back pointers (exact k-best search: every state among the
    kbest of a bin continues a state among the kbest of the bin it came from).
    The initial fcwd (delta peak at bin 0, or Gaussian for classically treated modes) is
    included exactly by folding it into the window, w_eff(E) = sum_b0 fcwd(b0) window(E+b0),
    so that each state of the quantum modes appears once, with its full contribution.

    Returns up to nstates states, ordered by their contribution weight*w_eff(E) to the
    window average sum_E FCWD(E) window(E): an array of contributions, an array of bins
    (energy of the quantum modes only), and an integer array (nstates, number of modes)
    with the quantum numbers of all modes.
    Memory ~ 4 x (number of modes) x nbins x kbest bytes.
    """
    cdef Py_ssize_t nbins = len(fcwd)
    if len(window) != nbins:
        raise ValueError("window and fcwd have to be defined on the same grid")
    if kbest > 32767:
        raise ValueError("kbest too large")
    cdef Py_ssize_t nmodes = len(mode_list)

    offs, bins, vals, quanta = _sticks(mode_list, dnu, nbins)
    if len(quanta) > 0 and np.max(quanta) > 32767:
        raise ValueError("too many quanta for the back pointers")
    cdef Py_ssize_t[:] offs_v = offs
    cdef Py_ssize_t[:] bins_v = bins
    cdef double[:] vals_v = vals

    cdef double[:, :] W = np.zeros((nbins, kbest))
    cdef double[:, :] Wn = np.zeros((nbins, kbest))
    cdef Py_ssize_t[:] cnt = np.zeros(nbins, dtype=np.intp)
    cdef Py_ssize_t[:] cntn = np.zeros(nbins, dtype=np.intp)
    # back pointers per mode, bin and rank: stick index within the mode, and rank in the parent bin
    stick_ptr = np.zeros((nmodes, nbins, kbest), dtype=np.int16)
    rank_ptr = np.zeros((nmodes, nbins, kbest), dtype=np.int16)
    cdef short[:, :, :] sp_v = stick_ptr
    cdef short[:, :, :] rp_v = rank_ptr

    cdef double[:] f0 = np.asarray(fcwd, dtype=np.float64)
    cdef double[:] win = np.asarray(window, dtype=np.float64)
    cdef Py_ssize_t bb, kk, ss, pb, rr, pos, qq, s0, s1
    cdef double ww, val

    # effective window: initial distribution folded into the window
    weff = np.zeros(nbins)
    for bb in np.nonzero(np.asarray(f0))[0]:
        weff[:nbins-bb] += f0[bb]*np.asarray(win)[bb:]

    W[0, 0] = 1.
    cnt[0] = 1

    for kk in range(nmodes):
        s0 = offs_v[kk]
        s1 = offs_v[kk+1]
        for bb in range(nbins):
            cntn[bb] = 0
            for ss in range(s0, s1):
                pb = bb - bins_v[ss]
                if pb < 0:
                    continue
                val = vals_v[ss]
                for rr in range(cnt[pb]):
                    ww = val*W[pb, rr]
                    # parents are sorted (descending): nothing better follows
                    if cntn[bb] == kbest and ww <= Wn[bb, kbest-1]:
                        break
                    # insert, keeping Wn[bb, :] sorted (descending)
                    pos = cntn[bb] if cntn[bb] < kbest else kbest-1
                    while pos > 0 and Wn[bb, pos-1] < ww:
                        if pos < kbest:
                            Wn[bb, pos] = Wn[bb, pos-1]
                            sp_v[kk, bb, pos] = sp_v[kk, bb, pos-1]
                            rp_v[kk, bb, pos] = rp_v[kk, bb, pos-1]
                        pos -= 1
                    Wn[bb, pos] = ww
                    sp_v[kk, bb, pos] = <short>(ss - s0)
                    rp_v[kk, bb, pos] = <short>rr
                    if cntn[bb] < kbest:
                        cntn[bb] += 1
        W, Wn = Wn, W
        cnt, cntn = cntn, cnt

    # contributions to the window average, best nstates overall
    contrib = np.asarray(W)*weff[:, None]
    counts = np.asarray(cnt)
    for bb in range(nbins):
        contrib[bb, counts[bb]:] = 0.
    flat = np.argsort(-contrib, axis=None)[:nstates]
    flat = flat[contrib.ravel()[flat] > 0.]
    nfound = len(flat)
    out_c = contrib.ravel()[flat]
    out_b = flat // kbest
    out_q = np.zeros((nfound, nmodes), dtype=np.intp)
    for ii in range(nfound):
        bb = out_b[ii]
        rr = flat[ii] % kbest
        for kk in range(nmodes-1, -1, -1):
            ss = offs_v[kk] + sp_v[kk, bb, rr]
            out_q[ii, kk] = quanta[ss]
            pb = bb - bins_v[ss]
            rr = rp_v[kk, bb, rr]
            bb = pb
    return out_c, out_b, out_q
