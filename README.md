# NonRadiativeRates
Collection of python code to compute the bits and pieces needed for non-radiative rates.

Used, e.g., in R. Toews, A. Köhn, "Dissecting non-radiative decay in donor-functionalized radicals with a mode-resolved model", Phys. Chem. Chem. Phys. 2026, 28, 776-789. https://doi.org/10.1039/d5cp03613e

# Layout

The package consists of a common core (Franck-Condon weighted densities, FCWD, and their
analysis) and the rate expression for internal conversion (IC):

```
src/nonradiative_rates/
    fcwd.py, fcf_utils.py, ...   common core: FCWD simulation and analysis, readers, constants
    _fastfold.pyx                Cython extension: folding in energy and Fourier space, analyses
    ic/                          IC: rate from the norm of the NAC vector, NAC projection
examples/
    ic/                          IC example (TTM-1Cz; required data included, full data on DaRUS:
                                 doi:10.18419/DARUS-6477)
tests/
    ic/                          tests of the core and of the IC part
```

`nonradiative_rates.tools.non_radiative` is kept as an alias of `nonradiative_rates.ic.non_radiative`.
The equations (FCWD, computation in Fourier space, analyses) and references are collected in
`docs/FCWD_equations.txt`.

## FCWD folding

The FCWD is the convolution of the Franck-Condon progressions of all
modes. `run_simulation(..., fcwd_method="direct")` (default) folds the modes one by one in energy
space; `fcwd_method="fft"` folds all modes at once in Fourier space (`_fastfold.fold_fft`). Both
give the same binned FCWD up to round-off; note that the Fourier-space round-off is absolute
(~1e-17 per bin for the normalized FCWD), i.e. it matters for FCWD values far in the tail.

## FCWD analysis

`run_simulation(..., fcwd_analysis=True)` omits each quantum-treated mode in turn (i.e. treats it as
undisplaced), for the harmonic and the anharmonic FCFs, and returns the result as
`FCWDResult.analysis` (an `FCWDAnalysis`: FCWD window averages with each mode omitted, the
remaining FCWD as fraction of the full FCWD, `retained_*`, and its inverse, the enhancement
factors `enhancement_*`); it is also written to `fcwd_analysis_file`. The
analysis can also be run on a result with `fcwd.analyze_fcwd(result, method)`. With
`fcwd_method="fft"`, the complete analysis (both FCF sets, all omissions) is done in a single
call in Fourier space (`_fastfold.fcwd_analysis_fft`): the FCWD without mode j is the product of
the transforms of all other modes (prefix and suffix products, no division), and its window
average follows from Parseval's theorem. This is ~30-40x faster than folding anew for each
omission (`fcwd_method="direct"`), with the same precision caveat as above.
`run_mode_tests` and `mode_contrib_file` are deprecated aliases of `fcwd_analysis` and
`fcwd_analysis_file`.

## Presentation of the analyses

All analyses compare FCWD window averages with the full FCWD, and all output gives the
**remaining FCWD in % of the full FCWD**: without a single mode (`fcwd_analysis`), without a
pair of modes (`fcwd_analysis_pairs`), with the modes of the set only (`minimal_mode_set`);
the dominant final states are given as their contribution in % of the full FCWD. Note that only
the latter add up: the FCWD is a sum over final states, but a convolution (product in Fourier
space) over modes, so contributions of modes are not additive. The synergy factor Q of a pair
is a ratio of four FCWDs and is given as such.

## Pair analysis (combination effects)

`run_simulation(..., fcwd_analysis_pairs=True)` (or `fcwd.analyze_fcwd_pairs(result)`) omits all
pairs of quantum-treated modes, for the harmonic and the anharmonic FCFs, in a single call in
Fourier space (`_fastfold.fcwd_pair_analysis_fft`; cost ~ M^2/2 x nfft/2 per FCF set, i.e. a few
seconds for ~100 modes; memory independent of the grid apart from an nfft table). The result,
`FCWDResult.pair_analysis` (an `FCWDPairAnalysis`), contains the FCWD window averages with each
pair omitted and the synergy factors

    Q_ij = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j))

which are > 1 if modes i and j reinforce each other, < 1 if they can partly replace each other
(e.g. two stretching modes offering alternative combinations of quanta), and 1 if they act
independently. Since Fourier-space round-off is absolute, each window average comes with a
conservative round-off estimate; Q_ij is marked as reliable only if all four values entering it
are at least `reliability` (default 10) times their estimates. `strongest_pairs()` lists the most
replacing or reinforcing reliable pairs, `write_fcwd_pair_analysis()` writes them to
`fcwd_analysis_pairs_file`.

## Promoting modes and dominant final states

Omitting single modes does not cover combination effects. Two analyses of a `run_simulation()`
result (harmonic FCFs, or anharmonic ones with `anharmonic=True`) include them:

- `fcwd.minimal_mode_set(result, fraction=0.9)`: greedy search for a small set of modes that
  promotes the transition. Starting from all modes, the mode whose omission (i.e. treating it as
  undisplaced) retains the largest FCWD window average is omitted, one at a time, as long as at
  least `fraction` of the full value is retained; each step evaluates all remaining modes
  together. Done in Fourier space in a single call (`_fastfold.minimal_mode_set_fft`); memory
  ~ 2 x (number of modes) x nfft/2 complex numbers. Returns a `ModeSet`; `write_mode_set()`
  writes it to a file.
- `fcwd.dominant_states(result, nstates=10, kbest=None)`: the final vibrational states (quantum
  numbers of all modes) contributing most to the FCWD window average, from an exact k-best fold
  in energy space (`_fastfold.dominant_states`, `kbest` states kept per bin, back pointers;
  memory ~ 4 x (number of modes) x nbins x kbest bytes). Classically treated modes enter through
  their energy distribution, which is folded into the window. Returns a list of `VibronicState`;
  `write_dominant_states()` writes it to a file.

Both are used in `examples/ic/IC_rate_constant.py`.

## Relaxed-gradient model

In the relaxed-gradient model, Hessian and gradient of the final state are computed at the
initial-state geometry. The FCWD is then measured at `e_vert + lambda`, where `e_vert` is the
electronic gap at that geometry and `lambda` the reorganization energy of the model, see
`nonradiative_rates.fcwd.relaxed_gradient_gap`. Imaginary modes of the final-state Hessian
are discarded (`nonradiative_rates.fcwd.imaginary_modes` lists them).

## Anharmonic FCFs

Modes above `anh_thr` are treated as Morse oscillators: the FCFs <v|0> of two displaced Morse
oscillators (Valiev et al., Phys. Chem. Chem. Phys. 2023, 25, 6406, Eq. 13; `FcfMorse0.get_I0n`)
are evaluated in logarithms, with the terminating hypergeometric function expanded in
w = (Delta - 1)/(Delta + 1) (`FcfMorse0._integrals`), so that they neither overflow nor cancel for
small anharmonicities xi = omega/(4 D) (the direct evaluation of Eq. 16, `get_I0n_ori`, overflows
below xi ~ 6e-3).

The example `examples/ic/IC_rate_constant.py` contains all input it needs (TTM-1Cz, relaxed-gradient
model); the FCWD analysis tools are switched on with `FCWD_analysis = True`; its output is stored
in `examples/ic/reference_output/`, and `examples/ic/comparison_previous_version.log` compares it
with the previous version of the code (results and timings).


# Setup

This requires a C compiler (to build the `_fastfold` Cython extension). Install the package,
in editable mode, from the repository root:
```
pip install -e .
```

For development (running the test suite):
```
pip install -e ".[dev]"
```

# Tests

Run the test suite with:
```
pytest
```

# Cython

The `_fastfold` extension is built automatically by `pip install -e .`. If you change
`src/nonradiative_rates/_fastfold.pyx` and want to rebuild it in place without reinstalling:
```
python setup.py build_ext --inplace
```
