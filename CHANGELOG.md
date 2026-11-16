# Changelog

## 3.5.0 — 15 November 2026

Faster fits with long-tailed kernels. Results within the dense limits are unchanged apart from rounding (about 1e-15).

- **Tabulated kernel evaluation** (`include/deeprbf/kernel_table.hpp`). In the O(n²) loops (dense assembly, exact
  matrix-free product, prediction) the kernels `gen_cauchy`, `exponential`, `gaussian`, `multiquadric`,
  `inverse_multiquadric` and `thin_plate` are evaluated from geometric bands of degree-6 Chebyshev interpolants
  instead of `pow`/`exp` and the three-argument `hypot`. The table is checked against the closed form when built
  (relative error ≤ 1e-13, about 1e-15 in practice; closed form used if any check fails). These loops are 3–4 times
  faster: an iterative `gen_cauchy` fit of 7,980 drillhole samples takes 23 s instead of 81 s on two cores.
- **H-matrix product** (`include/deeprbf/hmatrix.hpp`, `src/hmatrix.cpp`). k-d cluster tree, admissible blocks
  compressed by adaptive cross approximation (partial pivoting, tolerance 1e-9) with QR/SVD recompression, exact near
  field (stored within 2 GB, else recomputed). ACA checks the rows and columns farthest from its pivots, and random
  ones, before accepting convergence, so separate drillhole pieces in one cluster are not missed. Product error about
  1e-9 on drillhole data for every kernel; cost about O(n log n).
- **Iterative solver.** With the H-matrix the two-level Schwarz preconditioner is applied in hybrid (multiplicative)
  form, the coarse level holds up to 4,000 centres (partial-pivot LU instead of full-pivot LU, also without the
  H-matrix) and the GMRES restart grows to at most 500 vectors within 512 MB. The local blocks of the preconditioner
  now run in parallel. `set_hmatrix("auto")` (default) uses the H-matrix from 4,000 centres for `gen_cauchy`,
  `exponential`, `linear`, `cubic` and `thin_plate`; the Gaussian, multiquadric and inverse multiquadric kernels keep the
  exact product (their ill-conditioned systems amplify the product error). The 7,980-sample fit above takes 4.7 s
  (17 times faster than 3.4.2); 19,980 samples take 22 s and 39,960 samples 117 s on two cores. On a 24-thread
  workstation: 8,000 samples 2.0 s (3.4.2: 23.0 s), 20,000 in 10 s, 40,000 in 42 s, 80,000 in 182 s.
- **Preconditioner fix (affects 3.4.2 too).** The coarse level and each local block of the Schwarz preconditioner
  re-checked their solve on every application and dropped out when the residual exceeded 1e-8. On ill-conditioned
  blocks (closely spaced composites) that made the preconditioner change from one GMRES iteration to the next, and
  GMRES stalled. Validity is now decided once, by a test solve when the preconditioner is built (relative residual
  1e-6), and blocks up to 25% over the size limit are kept. The Kilamongo indicator fit with `gen_cauchy` (17,933
  points), which never converged before, now converges in 45 iterations (28 s with the H-matrix, 136 s with the exact
  product, on two cores); the whole shell, including a 4.5-million-node grid, takes 255 s on the workstation.
- **Exact residual check.** After every iterative fit the residual is computed exactly on 2,000 evenly spaced rows. If
  an H-matrix solution fails the check, the solve is repeated with the exact product (backend name contains
  `after_hmatrix_check`), so a result is never less accurate than with the exact product.
- New Python API: `NativeExactRBFImplicitModel.set_hmatrix`, `hmatrix_mode`, `iterative_info`, `weights`; class
  `HMatrix`; function `kernel_table_check`.
- Tests: `tests/test_kernel_table.py`, `tests/test_hmatrix.py` (277 Python tests pass, 3 skipped).
- Benchmark B7 (`benchmarks/bench_deeprbf.py`): dense, exact-product and H-matrix fits of `gen_cauchy` and
  `exponential` on drillhole-like data, 5,000 to 80,000 centres.
- Known limitation, unchanged from 3.4.2: GMRES may not converge in the iterative path for the `cubic` kernel and for
  wide Gaussian shapes.
- Documentation: Technical Reference (Solvers, native interface) and User Guide updated for 3.5.0.

## 3.4.2 — 15 November 2026

- **Regularisation sign (engine fix).** The linear and multiquadric kernels are conditionally positive definite of
  order 1 only after a change of sign: on the drift-orthogonal subspace their matrices are negative definite. Earlier
  releases added +ν to the diagonal for every kernel, which for these two moved the spectrum towards zero instead of
  smoothing: with ν = 0.1 and noise of 0.3 the fit error rose to 18 (linear) and 8 (multiquadric). The engine now adds
  −ν for these two kernels (`regularization_sign` in `kernels.hpp`; dense, iterative and preconditioner paths) and +ν
  for all others, so ν smooths in every case. Results with ν = 0, and all results with the spherical, exponential,
  Gaussian, generalised Cauchy, cubic or thin-plate kernels, are unchanged; no Kilamongo shell changes.
  `tests/test_regularization_sign.py` checks dense and iterative fits; `tests/test_regression.py` uses the same sign.
- `benchmarks/bench_deeprbf.py`: B3 adds a signed-distance coding and a far field across the lens; B5 adds a
  threshold calibrated to the ore proportion. Results with 3.4.2 in `benchmarks/results/bench_results_3_4_2_B3_B5.jsonl`.
- `docs/`: User Guide for resource geologists and Technical Reference Manual (Markdown and PDF), and
  `docs/examples/krige_inside_shell.py`, all checked against 3.4.2.
- `CITATION.cff`: ORCID added.
- This changelog: the 3.4.1 entry wrongly said the 3.4.1 wheel carries the 3.4.0 native module; the wheel was rebuilt
  and its native module reports 3.4.1.

## 3.4.1 — 15 November 2026

Corrections to the Kilamongo example and the test suite. The native engine code is unchanged from 3.4.0; the 3.4.1
wheel is rebuilt from it and its native module reports version 3.4.1, as the version check requires.

- `examples/kilamongo/kilamongo_signed_distance.json`: drift changed from `linear` to `constant`. With a linear drift
  the signed-distance field crossed to the ore side beyond the drilling and the shell gained about 1.0 Mm³ against the
  southern model boundary with no composites in it (1.79 Mm³ in total; precision 0.93, recall 0.77). With a constant
  drift the shell is 0.78 Mm³ with the same honouring of the data (precision 0.93, recall 0.76). A signed-distance field
  needs a constant drift: with none it reverts to zero, which is the boundary itself (25.5 Mm³).
- `tests/test_runtime_bootstrap.py` runs only in the Windows development tree it was written for; elsewhere it is
  skipped. On Linux (GCC 13.3, Eigen 3.4, Python 3.11) all 19 native tests and all other Python tests pass.

## 3.4.0 — 15 November 2026

First public source release. Geostatistics with GSLIB conventions, and a faster dense solver.
(Released internally as 3.3.4 on 14 November 2026; identical code.)

- `kriging_gslib`: ordinary or simple point kriging with a nugget plus nested spherical, exponential and gaussian
  structures (`GSLIBVariogram`, `GSLIBStructure`), each with its own ranges and GSLIB angles (azimuth clockwise from
  north, dip positive down, rake). The variogram range is independent of the anisotropic search (`SearchEllipsoid`).
  Agrees with GSLIB kb2d (GeostatsPy port) to 1e-14 isotropic and 7e-10 anisotropic, about 100x faster.
- `sgs_gslib`: sequential Gaussian simulation with normal-score transform and linear tails, simple kriging by
  default (ordinary optional), separate limits on data and previously simulated nodes, multiple-grid path and
  realisations in parallel. Reproduces the variogram and histogram as GSLIB sgsim does on the same data.
- `sis_gslib`: sequential indicator simulation with simple IK on global proportions, one variogram per category,
  order-relation correction, spatial index (the 3.3.3 `run_sis` scanned every node) and multiple-grid path.
- `normal_score`, `gslib_rotation`, `GSLIBVariogram.covariance` exposed for checking.
- Dense RBF solves use partial-pivot LU, falling back to full-pivot LU only if the residual check fails. The 3.3.3
  full-pivot LU took 127 s at 5,000 points; partial-pivot LU takes 2.8-3.0 s (Windows, 24 threads).
- The 3.3.3 functions (`ordinary_kriging`, `run_sgs`, `run_sis`, ...) are unchanged for reproducibility. Note their
  conventions: range equals the search radius, azimuth is counter-clockwise from east, SGS uses ordinary kriging on
  standardised values without a normal-score transform.
- Tests: `tests/test_gslib_simulation.py` (12).

## 3.3.3 — 14 November 2026

Covariance kernels, drift choice, a sparse solver for compact kernels, and a reusable grade-shell workflow.

- Kernels: `spherical` and `exponential` (unit sill, epsilon = range) and `gen_cauchy[:alpha:power]`, a long-tailed
  covariance at 96 % of the sill at the range (the spheroidal convention of common commercial software; default alpha 9, power 1.667,
  fitted to a spheroidal alpha 9 curve). `deeprbf_recovered.kernel_value(kernel, r, epsilon)` evaluates any kernel.
- Drift: `NativeExactRBFImplicitModel.set_drift("none" | "constant" | "linear")`, default `linear` (unchanged).
  `none` is accepted only for positive-definite kernels, `constant` not for cubic or thin-plate.
- Sparse solver for compact kernels (spherical): neighbour assembly on a grid hash, sparse Cholesky (LDLT, AMD) with a
  Schur complement for the drift, selected automatically above the dense limit (`set_sparse_solver`,
  `sparse_solve_info`); compact prediction sums only centres within the range. 17,934 drillhole composites fitted in
  54 s where the dense and iterative paths did not finish; agreement with dense below 1e-8.
- `deeprbf.workflow.shell`: drillhole CSVs + one JSON settings file -> composites (minimum curvature) -> indicator,
  signed-distance (BOUNDSIM) or contacts-with-normals shell -> topography clipping (DXF, Surpac, CSV) -> small bodies
  removed on the grid before contouring -> Surpac .str/.dtm and DXF -> JSON report (honouring of the data, volumes,
  reference comparison, withheld-hole folds, conservative/expected/aggressive shells). Run
  `python -m deeprbf.workflow.shell settings.json`; examples in `examples/workflow`. Needs scikit-image.
- Kilamongo South 0.3 g/t (public data): with the settings of the commercial reference shell the shell honours the 1 m composites with F1 0.84
  (reference 0.82) and shares 90 % of the reference shell volume.
- Tests: 62 new (kernels and drift, sparse solver, sparse with drift, workflow).

## 3.3.2 — 13 November 2026

Large RBF models.

- Iterative fits now converge on drillhole data. The two-level domain-decomposition preconditioner uses
  fixed-size nearest-neighbour overlaps and restricted additive Schwarz; the coarse level is capped at 2,000
  points. In 3.3.1 most overlap blocks exceeded the local size limit and were skipped.
- GMRES: restart 150 and up to 3,000 iterations by default, configurable with
  `NativeExactRBFImplicitModel.set_iterative_solver(restart, max_iter, tol)`. The true residual is computed only
  when the Arnoldi estimate passes.
- Kernels are resolved once; non-Gaussian kernels use the exact OpenMP product directly.
- Prediction backend is explicit: `set_predict_backend("auto" | "direct" | "scalfmm")`, with the backend used
  recorded by `last_predict_backend()`. `scalfmm` never falls back silently; `auto` uses ScalFMM only for
  Gaussian evaluations of at least 2e9 source-target pairs.
- Dense-solver memory budget configurable with `set_dense_memory_limit_mb` (default 1 GB, was 256 MB).
- New tests: `tests/test_large_rbf_solver.py`.

Benchmark (Windows, 8 threads, oneMKL + ScalFMM build): a drillhole contact field (+1 inside the logged unit,
-1 outside, 0 at down-hole contacts) with the linear kernel fitted 6,000 points in 8 s, 12,000 in 40 s and
19,358 in 264 s, honouring every data point. 3.3.1 did not converge at 6,000 points on the same data.
For 4,000 Gaussian centres and 1,000,000 targets, ScalFMM took 8.6 s against 11.3 s for exact evaluation,
with a relative L2 difference of 4e-9.

Guidance: for implicit contact modelling use the `linear` (or `cubic`) kernel. A Gaussian kernel at a range of
tens of metres cannot separate contacts logged one to two metres apart down a hole.
