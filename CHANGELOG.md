# Changelog

## Unreleased (branch `ic-dev`, based on `main` at ecf3788)

Development branch for internal conversion (IC): FCWD in Fourier space, FCWD analyses including
combination effects, and the IC rate.

### Fixed

- `_fastfold.fold`: the grid spacing `dnu` was declared as single-precision `float`. For grid
  spacings that are not exactly representable in single precision (e.g. `e_bin = 0.1`), FCF lines
  could be put one bin too low (e.g. a line at 1436.9 rcm into bin 14368 instead of 14369). `dnu`
  is now `double`. **This changes results compared with earlier versions for such grids**
  (observed: relative changes of the window-averaged FCWD of 1e-8 to 1e-4, individual bins up to
  1e-2). Grid spacings exact in single precision (1, 0.5, 0.25, ...) are not affected.
- `FcfMorse0.get_b0n` (coordinate integrals <0|x|n> of displaced Morse oscillators, Eq. 15 of
  Valiev et al., PCCP 2023, 25, 6406): beta was computed as sqrt(2 D)/alpha without the reduced
  mass, inconsistent with the Morse parameters a_n of the same class (sqrt(2 mred D)/alpha); the
  results were wrong for mred != 1. Not used by the FCWD. Now tested against a numerical (DVR)
  reference, together with the overlaps <0|n> (`tests/ic/test_fcf_utils.py`).
- `examples/ic/IC_rate_constant.py`: the projected NACs in `rates_new.dat` were assigned to the
  modes with a fixed offset (`nacv_pf[7:]`), shifted by one mode for TTM-1Cz (6 discarded modes,
  no imaginary mode); they are now selected by frequency (>= 1e-3 cm-1, as in `run_simulation`).

### Changed

- `FcfMorse0.get_I0n` and `get_b0n` (new common `FcfMorse0._integrals`): evaluated in logarithms,
  with the terminating hypergeometric function expanded in w = (Delta - 1)/(Delta + 1) instead of
  1/A (connection formula, exact for equivalent oscillators), and dI_n/dB analytic instead of
  finite differences. They no longer overflow for small anharmonicities (previously NaN for
  xi = omega/(4 D) below ~6e-3; tested down to xi = 1e-7, approaching the harmonic integrals as
  sqrt(xi)) and agree with a numerical reference to ~1e-12. `get_b0n` no longer takes the step
  size `h`. `get_I0n_ori` (direct evaluation of Eq. 16) is unchanged.
- `run_simulation()`: all modes above `anh_thr` get Morse FCFs; the fallback to harmonic FCFs for
  xi < 7e-3 (overflow of the previous Morse code) is removed. **This changes anharmonic results**
  for modes with omega < 0.028 D (400-840 cm-1 for the default D = 30000 cm-1): IC example,
  anharmonic rate 1.042e7 -> 1.038e7 1/s (-0.4 %); harmonic results unchanged.
- Consistent presentation of all analyses: the output gives the remaining FCWD in % of the full
  FCWD (single modes and pairs omitted, mode set only) and the contributions of the dominant
  states in % of the full FCWD. `FCWDAnalysis` additionally holds `retained_h`/`retained_a`
  (fraction; the enhancement factors are kept as their inverse), `ModeSet` holds `retained` and
  `omitted_retained`.
- `run_simulation`: `run_mode_tests` and `mode_contrib_file` are renamed to `fcwd_analysis` and
  `fcwd_analysis_file` (the old names still work, with a `DeprecationWarning`); the internal
  `_mode_contribution_tests` is replaced by `analyze_fcwd`.
- `nonradiative_rates.tools` only re-exports `non_radiative` from `nonradiative_rates.ic`
  (backward compatible).
- Examples and tests moved into subdirectories: `examples/ic/` (`fcwd_simulation.py`,
  `IC_rate_constant.py`), `tests/ic/` (previous tests and test data).
- `examples/ic/IC_rate_constant.py` is self-contained and runs: the files of the TTM-1Cz data
  needed by the readers (D0 gradient and Hessian at the D1 geometry, derivative couplings;
  CAM-B3LYP/def2-SVP, from the DaRUS data set doi:10.18419/DARUS-6477) are included in
  `examples/ic/relaxed-gradient_model/` (0.6 MB), paths are relative to the script, it uses
  `ic.effective_nac` and `fcwd_method="fft"`, the analysis tools are switched with
  `FCWD_analysis`, and the reference output is in `examples/ic/reference_output/`. The energy gap
  of the original example (15924 cm-1) is kept; a comment notes the value e_vert + lambda =
  20561 cm-1 of the relaxed-gradient model. Single-mode results are presented as enhancement
  factors FCWD/FCWD(-k), as in the previous version (console output and `FCWD_contr.dat` in the
  previous format, in addition to `FCWD_analysis.dat`).
  `examples/ic/comparison_previous_version.log`: comparison with the previous version (main
  branch): identical rate constants; enhancement factors agree to <= 2e-5 (harmonic, Fourier-space
  round-off for FCWD(-k) ~ 2e-15) and 1e-8 (anharmonic); sensitivity analysis 41.3 s -> 0.61 s
  (x64), example with all analyses 42.4 s -> 12.5 s (state of 2026-10-05; see the note at its end
  for the later Morse change).
- `fcwd._bin_fcwd` additionally returns the grid energies; the window is set up in `fcwd._window`.
- `fcf_utils.FCWD`: the set-up of the initial FCWD (Gaussian for classically treated modes, or a
  delta peak) is factored out into `FCWD._initial_fcwd`; `get_FCWD` is otherwise unchanged.

### Added

- References: the relaxed-gradient model is attributed to the preprint R. Toews, A. Koehn,
  "Internal conversion in functionalized poly-chlorinated trityl radicals can be tuned by a
  characteristic high-frequency mode", ChemRxiv, doi:10.26434/chemrxiv.15003675/v2 (new reference
  [2] in `docs/FCWD_equations.txt`, the following references renumbered; README), not to the PCCP
  paper (Toews & Koehn, PCCP 2026, 28, 776), which remains the reference for the FCWD by folding
  and the mode-resolved analysis.
- `nonradiative_rates.ic`: `non_radiative(nac, fcwd)` (moved from `tools`) and
  `effective_nac(nac_calculation, hessian_calculation)`, the projection of the derivative-coupling
  vector onto the normal modes (mass weighting, projection, frequency weighting), previously part
  of `examples/IC_rate_constant.py`.
- `fcwd.relaxed_gradient_gap(e_vert, hessian_calculation, egrad_calculation, damp, damp_thr)`:
  energy at which the FCWD is measured in the relaxed-gradient model, e_vert + lambda, with the
  electronic gap e_vert at the initial-state geometry and the reorganization energy lambda.
- `fcwd.imaginary_modes(hessian_calculation)`: lists the imaginary modes, which `run_simulation()`
  discards.
- FCWD folding in Fourier space:
  - `_fastfold.fold_fft(fcwd, mode_list, dnu)`: folds all modes at once (product of the
    transforms of the binned FCF progressions, zero padding to >= 2 nbins). Same binning as
    `fold()`; agrees with it up to round-off, which is absolute (~1e-17 per bin of the normalized
    FCWD) and therefore matters only far in the tail.
  - `FCWD(..., method="direct" | "fft")` and `run_simulation(..., fcwd_method="direct" | "fft")`.
    Default `"direct"`, i.e. unchanged behaviour.
  - `fcf_utils.fold_fft_py`: NumPy version, used with `use_cython=False`.
- FCWD analysis (omission of single modes, enhancement factors) in a single call:
  - `_fastfold.fcwd_analysis_fft(fcwd, mode_lists, window, dnu)`: for one or several sets of FCF
    progressions at once (e.g. harmonic and anharmonic), window averages of the full FCWD and of
    the FCWD with each mode omitted in turn. Done in Fourier space (prefix and suffix products of
    the mode transforms, no division; window averages via Parseval's theorem, no inverse
    transform). Mode transforms are evaluated block-wise from the FCF sticks with a table of exact
    phase factors. `_fastfold.mode_contributions_fft` (one set) and `FCWD.get_mode_contributions`
    use it.
  - `fcwd.analyze_fcwd(result, method="fft" | "direct")` returns an `FCWDAnalysis`;
    `fcwd.write_fcwd_analysis` writes it.
  - `run_simulation(..., fcwd_analysis=True, fcwd_analysis_file="FCWD_analysis.dat")` runs the
    analysis with the method given by `fcwd_method` and stores it in `FCWDResult.analysis`
    (~30-60x faster with `"fft"` than with `"direct"`).
- Pair analysis of the FCWD (combination effects):
  - `_fastfold.fcwd_pair_analysis_fft(fcwd, mode_lists, window, dnu)`: window averages of the
    FCWD with each pair of modes omitted, in a single call in Fourier space, with conservative
    estimates of the absolute round-off of each value.
  - `fcwd.analyze_fcwd_pairs(result, reliability=10.)` returns an `FCWDPairAnalysis` with the
    synergy factors Q_ij = FCWD FCWD(-i,-j) / (FCWD(-i) FCWD(-j)) (> 1 reinforcing, < 1
    replacing) and reliability flags; `fcwd.strongest_pairs` and `fcwd.write_fcwd_pair_analysis`.
  - `run_simulation(..., fcwd_analysis_pairs=True, fcwd_analysis_pairs_file=...)`; the result is
    stored in `FCWDResult.pair_analysis`.
- Promoting modes and dominant final states:
  - `fcwd.minimal_mode_set(result, fraction=0.9, anharmonic=False)`: greedy search (backward
    elimination) for a small set of modes retaining at least `fraction` of the FCWD window
    average; single call in Fourier space (`_fastfold.minimal_mode_set_fft`). Returns a `ModeSet`.
  - `fcwd.dominant_states(result, nstates=10, kbest=None, anharmonic=False)`: dominant final
    vibrational states within the window, from an exact k-best fold in energy space with back
    pointers (`_fastfold.dominant_states`). Returns a list of `VibronicState`.
  - `fcwd.write_mode_set`, `fcwd.write_dominant_states`: text output.
  - `FCWDResult` additionally stores the window (`window`, `grid`), `max_e`, `e_bin` and the
    classical reorganization energy `Ereo_class`, so that the analyses work on a result directly.
- `docs/FCWD_equations.txt`: equations (FCWD, Morse FCFs, Fourier space, analyses, IC rate) and
  references.
- Tests (`tests/ic/`): Fourier-space vs. direct folding, single-call analysis vs. separate folds
  (1 and 0.25 rcm grids), pair analysis vs. brute-force enumeration and direct folds, dominant
  states vs. brute-force enumeration and products of FCFs, minimal mode set (incl. a fold of the
  set in energy space), Morse integrals vs. a numerical reference, deprecated options. Realistic
  cases: the IC test molecule (with and without classical treatment of low-frequency modes) and
  the TTM-1Cz data of the IC example, evaluated at 7000 cm-1 (FCWD ~1e-6, without classically
  treated modes).
- README: layout, FCWD folding, analyses, relaxed-gradient model, Morse FCFs.
