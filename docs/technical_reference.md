---
title: "DeepRBF Technical Reference Manual"
subtitle: "Version 3.5.0"
author: "Ramadhani W. Ndonde, DeepFusion Geoscience Limited"
date: "November 2026"
---

# Scope

This manual describes the DeepRBF engine: the interpolation model, kernels, drift, solvers and prediction
backends, the grade-shell workflow (every setting and report field), the GSLIB-convention geostatistics, the
native and Python interfaces, building from source, testing and versioning. The [User Guide](user_guide.md) covers
everyday use.

Everything stated here was checked against release 3.4.2 (Python package `deeprbf` and native module
`deeprbf_recovered`, both 3.4.2). Cite the release as Ndonde (2026), DeepRBF version 3.4.2, archived on Zenodo
(all versions: https://doi.org/10.5281/zenodo.24232966; each release also has its own version DOI).

# Architecture

| Layer | Content |
|---|---|
| Native core (C++20, Eigen 3.4, OpenMP) | RBF model, kernels (tabulated evaluation, `kernel_table.hpp`), dense, iterative and sparse solvers, H-matrix/ACA product (`hmatrix.hpp`), compact-support neighbour grid, kriging, simulation, variograms |
| Bindings (pybind11) | module `deeprbf_recovered`, built from `src/bindings.cpp` |
| Python package `deeprbf` | grade-shell workflow (`deeprbf.workflow`), lithology, categorical and simulation helpers, I/O |
| Optional | Intel oneMKL (dense solves), ScalFMM 3 (fast multipole evaluation of Gaussian sums), PyTorch (`deeprbf.svgp`) |

The pip distribution is named `deepfusion-deeprbf`. Importing `deeprbf` imports `deeprbf_recovered` and refuses
to load if the two versions differ (see *Versioning*).

# The interpolation model

## Formulation

Given centres $x_i$ with values $f_i$ ($i = 1 \dots n$), DeepRBF fits

$$ s(x) = \sum_{i=1}^{n} \lambda_i \, \varphi(\lVert x - x_i \rVert) + \sum_{k=1}^{m} c_k \, p_k(x) $$

where $\varphi$ is the kernel and $p_k$ the drift basis: none ($m = 0$), constant $[1]$ ($m = 1$) or linear
$[1, x, y, z]$ ($m = 4$). The coefficients solve the augmented system

$$ \begin{bmatrix} \Phi + \rho I & P \\ P^{\mathsf T} & 0 \end{bmatrix}
   \begin{bmatrix} \lambda \\ c \end{bmatrix} = \begin{bmatrix} f \\ 0 \end{bmatrix},
   \qquad \Phi_{ij} = \varphi(\lVert x_i - x_j \rVert), \; P_{ik} = p_k(x_i) $$

with regularisation $\rho \ge 0$ added to the diagonal. $\rho = 0$ interpolates exactly; $\rho > 0$ smooths
(it plays the role of a nugget effect). For the `linear` and `multiquadric` kernels, which are conditionally
positive definite of order 1 only after a change of sign, the engine adds $-\rho I$ instead, so that $\rho$ smooths
for every kernel. Releases before 3.4.2 added $+\rho I$ for all kernels, which for these two could make the system
nearly singular. Duplicate centres must be removed or averaged before fitting; the
workflow averages composites that coincide in model space.

Anisotropy is applied by transforming coordinates before fitting, so the kernel always sees isotropic distances
(see *Anisotropy*).

## Kernels

$r$ is distance, $\varepsilon$ the shape parameter (`epsilon`) and $q = r / \varepsilon$.

| Kernel string | $\varphi(r)$ | Type | Minimum drift |
|---|---|---|---|
| `linear` | $r$ | conditionally positive definite (CPD) | constant |
| `cubic` (alias `biharmonic`) | $r^3$ | CPD order 2 | linear |
| `thin_plate` (alias `thin_plate_spline`) | $q^2 \ln q$ | CPD order 2 | linear |
| `multiquadric` | $\sqrt{r^2 + \varepsilon^2}$ | CPD | constant |
| `inverse_multiquadric` | $1 / \sqrt{r^2 + \varepsilon^2}$ | positive definite (PD) | none |
| `gaussian` | $e^{-q^2}$ | PD | none |
| `spherical` | $1 - 1.5q + 0.5q^3$ for $q < 1$, else 0 | PD, compact support $r < \varepsilon$ | none |
| `exponential` | $e^{-3q}$ | PD (practical range $\varepsilon$) | none |
| `gen_cauchy[:a:p]` | $(1 + (r/b)^p)^{-a/p}$ | PD for $a > 0$, $0 < p \le 2$ | none |

For the generalised Cauchy kernel, $b = \varepsilon / (0.04^{-p/a} - 1)^{1/p}$, so $\varepsilon$ is the distance
at which the covariance falls to 0.04 (the variogram reaches 96% of the sill, the convention of common commercial software for the
spheroidal range). The defaults $a = 9$, $p = 1.667$ (`gen_cauchy` or `gen_cauchy:9:1.667`) are a fit to the
spheroidal curve with alpha 9 used in commercial software. The shape can also be set with `set_kernel_shape(alpha, power)` before
fitting.

`set_drift` enforces the minimum drift: `none` with a CPD kernel raises
`set_drift("none"): kernel "linear" is not positive definite ...`, and `constant` with a CPD order-2 kernel raises
`... is conditionally positive definite of order 2 ...`. The default drift of `NativeExactRBFImplicitModel` is
`linear`; the grade-shell workflow defaults to `none`.

`kernel_value(kernel, r, epsilon)` evaluates any kernel. Unknown strings raise `unknown RBF kernel: <name>`.

## Choosing kernel and drift

- **Contacts and signed implicit surfaces** with data in a few planes or holes: `linear` or `cubic` with a linear
  drift. The Gaussian kernel at ranges of tens of metres cannot separate contacts logged 1–2 m apart in one hole.
- **Indicator and distance fields** from many composites: `spherical` with a range matched to the continuity.
  Compact support makes the matrix sparse (see *Solvers*).
- **Smooth trends** far from data: `gen_cauchy` or `exponential`. The matrix is dense; cost grows as $n^2$ in memory.
- **Drift and extrapolation.** A linear drift extrapolates trends beyond the data; on a bounded indicator or
  distance field it can create spurious surfaces near the edge of the model box. A constant drift pins the far
  field to the data mean; no drift pulls the field to zero away from data. The workflow defaults are: indicator
  `none`, signed distance `constant`, contacts `linear`.

# Solvers

`fit` chooses a solver from the problem size and the kernel. $n$ is the number of centres and $m$ the drift size.

| Solver | `solver_backend()` reports | Used when |
|--------|----------------------|-------------|
| Dense LU | `dense_augmented_eigen_partial_pivot_lu` | $n \le$ dense limit and memory within budget (default) |
| Sparse LDLᵀ + Schur complement | `sparse_cholesky_schur` | compact kernel (`spherical`) and the dense limits exceeded, or forced |
| GMRES, two-level Schwarz | `iterative_augmented_direct_openmp_gmres_ddm_levels=2_blocks=<k>` | non-compact kernel beyond the dense limits, or forced |
| GMRES with H-matrix product (3.5.0) | `iterative_augmented_hmatrix_aca_gmres_ddm_levels=2_blocks=<k>_hybrid` | as GMRES, $n \ge 4000$, and a finitely smooth kernel (`gen_cauchy`, `exponential`, `linear`, `cubic`, `thin_plate`) |

**Dense.** Partial-pivot LU of the augmented matrix (Eigen, or oneMKL when built with it), with a full-pivot LU
fallback if the partial-pivot solve is inaccurate. The dense limits are $n \le 8000$
(`set_dense_direct_limit`) and a working-memory budget of $3 (n+m)^2 \times 8$ bytes $\le$ 1024 MB
(`set_dense_memory_limit_mb`).

**Sparse.** For compact kernels the matrix $\Phi + \rho I$ is sparse and positive definite. It is assembled from a
neighbour grid of cell size equal to the support radius, factorised by sparse LDLᵀ, and the drift is handled
through the Schur complement $P^{\mathsf T} (\Phi + \rho I)^{-1} P$. `sparse_solve_info()` returns
`nnz_lower`, `assembly_seconds`, `factor_seconds`, `solve_seconds` and `relative_residual`.
`set_sparse_solver(True)` forces this path for compact kernels at any size.

**Iterative.** Restarted GMRES on the augmented system, preconditioned by a two-level additive Schwarz method
(overlapping local blocks plus a coarse level). Defaults: restart 150, at most 3000 iterations, relative tolerance
$10^{-6}$ (`set_iterative_solver(restart, max_iter, tol)`). The full backend name is
`iterative_augmented_direct_openmp_gmres_ddm_levels=2_blocks=<k>`, with k the number of local blocks. `force_iterative_solver(True)` forces it.

**Kernel evaluation (3.5.0).** In the $O(n^2)$ loops (dense assembly, the exact matrix-free product, prediction)
each kernel except `linear`, `cubic` and `spherical`, which are cheap in closed form, is evaluated from a table: the
distance, scaled by $\varepsilon$ (or by the generalised Cauchy scale $b$), is split into geometric bands
$[2^e, 2^{e+1})$ of 32 cells, each with a degree-6 polynomial interpolant at Chebyshev nodes. Geometric bands keep the
error uniform near $r = 0$, where $r^p$ ($p < 2$), $|r|$ and $r^2 \log r$ are not smooth. The table is checked against
the closed form when it is built (relative error $\le 10^{-13}$; in practice a few units in the last place) and is
discarded in favour of the closed form if any check fails. It makes these loops 3 to 4 times faster with no change
to the results. `kernel_table_check(kernel, epsilon, d_max)` reports the table's error on random distances.

**H-matrix product (3.5.0).** For large iterative fits the product $\Phi \lambda$ is computed from a hierarchical
approximation of $\Phi$. The centres are ordered by a k-d cluster tree (bisection of the longest box side at the
median; leaves of at most 64 points). A pair of clusters $(t, s)$ with
$\min(\operatorname{diam} t, \operatorname{diam} s) \le \eta \operatorname{dist}(t, s)$, $\eta = 1.5$, is
compressed to $U V^{\mathsf T}$ by adaptive cross approximation with partial pivoting to a tolerance of $10^{-9}$
relative to the block, then recompressed by QR and a truncated SVD. Before accepting convergence ACA checks the 8 rows
and 8 columns farthest from all pivots so far and 8 random ones of each; this catches the separate drillhole pieces
and isolated samples that partial pivoting alone can miss. Blocks that are not admissible down to the leaves are
evaluated exactly: stored if they fit in 2048 MB, recomputed on every product otherwise. The product then costs about
$O(n \log n)$; its relative error is about $10^{-9}$ for all kernels on drillhole data.

With the H-matrix the Schwarz preconditioner is applied in hybrid (multiplicative) form: coarse correction, residual
update with one product, local blocks on the remainder. The coarse level grows from 2,000 to 4,000 centres (blocked
partial-pivot LU) and the GMRES restart to $\min(500, 512\,\text{MB} / (16 (n+m)))$ vectors.

*Exact check and fallback.* After the solve, the residual $|(\Phi + \rho I)\lambda + P c - f|$ is computed exactly
(closed-form kernel sum) on 2,000 evenly spaced rows. If its maximum exceeds $\max(10^{-4},
10\,\mathrm{tol}\,\|f\|_2 / \|f\|_\infty) \cdot \|f\|_\infty$, or GMRES fails, the solve is repeated with the exact
product and the additive preconditioner, starting from the H-matrix solution; the backend name then contains
`after_hmatrix_check`. The Gaussian, multiquadric and inverse multiquadric kernels give very ill-conditioned systems at
practical shape parameters, where the product error is amplified by large weights; `auto` therefore keeps the exact
product for them.

`set_hmatrix(mode, tol=1e-9, eta=1.5, leaf_size=64, near_memory_mb=2048)` sets `auto` (default), `on` (always in the
iterative path) or `off`. `iterative_info()` returns a dictionary for the last iterative fit: `products`, the set-up,
GMRES and check times, `gmres_restart`, `preconditioner_hybrid`, `sampled_rows`, `sampled_rel_residual`
($\max |r_i| / \max |f_i|$), `hmatrix`, `hmatrix_fallback`, and with the H-matrix its build time, memory, ranks,
compression and `hmatrix_sampled_rel_residual`.

*Timings (24-thread Windows workstation, 63 GB, release wheel; generalised Cauchy, drillhole-like data, constant
drift).* 8,000 centres: dense 12.2 s; 3.4.2 exact product 23.0 s; 3.5.0 exact product 6.8 s; 3.5.0 H-matrix 2.0 s.
H-matrix at 20,000 centres 10.2 s, 40,000 centres 42 s, 80,000 centres 182 s (658 GMRES iterations). Predictions
differ from the dense solve by $3.7 \times 10^{-6}$ of their range. The Kilamongo indicator shell with `gen_cauchy`
(17,958 composites, model fitted and evaluated on a 2 m grid of 4.5 million nodes) takes 255 s end to end; it did not
converge before the preconditioner fix below. GMRES iterations still grow with $n$; a more scalable preconditioner is
future work. Benchmark B7 in `benchmarks/bench_deeprbf.py` measures dense, exact-product and H-matrix fits up to
80,000 centres.

The coarse level and the local blocks are checked once, by a test solve when the preconditioner is built (relative
residual $10^{-6}$); before 3.5.0 they were re-checked on every application, which made the preconditioner vary
between GMRES iterations on ill-conditioned blocks and stalled the solver.

*Limitation.* With the `cubic` kernel and with wide Gaussian shapes, GMRES may fail to converge in the iterative path
(also in 3.4.2); use the dense solver within its limits, a compactly supported kernel, or `fit_with_reduction`.

The solver controls are static: they apply to all models created afterwards in the process.

**Greedy reduction.** `fit_with_reduction(centers, values, fitting_accuracy=1e-4, initial_ratio=0.1,
append_ratio=...)` starts from a subset of the centres and adds the worst-fitted points until every value is
reproduced within `fitting_accuracy`. Useful for very dense, redundant data.

# Prediction backends

| `set_predict_backend` | `last_predict_backend()` reports | Behaviour |
|------|---------|-----------|
| `auto` (default) | `compact_grid_openmp` for compact kernels, `direct_openmp` otherwise, `scalfmm` when used | compact kernels sum only centres within the support radius (exact); Gaussian uses ScalFMM when available and targets × centres ≥ 2×10⁹ |
| `direct` | `direct_openmp` | full $O(n)$ sum per target, exact |
| `scalfmm` | `scalfmm` | fast multipole method (approximate); raises if the build has no ScalFMM |

`deeprbf_scalfmm_runtime_status()` and `deeprbf_scalfmm_diagnostics()` report whether ScalFMM is compiled in.
The release wheel does not include ScalFMM or oneMKL.

# Anisotropy

## Dip, dip azimuth and pitch (grade-shell workflow)

`deeprbf.workflow.anisotropy.Frame(dip, dip_azimuth, pitch, ranges, origin)` maps world coordinates to a model
frame in which each range becomes 1:

- `dip`, `dip_azimuth` define the plane of maximum continuity.
- `pitch` is measured in that plane from the strike direction (dip azimuth − 90°) towards down-dip; 90 puts the
  maximum axis down-dip.
- `ranges` = (maximum, intermediate, minimum); minimum is normal to the plane.

Coordinates are first shifted by `origin` (the workflow uses the composite mean) for numerical conditioning.
Because ranges become 1, the workflow fits with `epsilon = 1`, and `range_factor` scales all three ranges.

## GSLIB convention (kriging and simulation)

`GSLIBStructure`, `SearchEllipsoid` and `gslib_rotation(azimuth, dip, rake, range_major, range_minor,
range_vertical)` use GSLIB `setrot` angles: `azimuth` of the major axis clockwise from north, `dip` of the major
axis, `rake` about the major axis. In DeepRBF a **positive dip plunges the major axis downward**.
`gslib_rotation` returns the row-major 3×3 matrix mapping a lag to range-scaled (major, minor, vertical)
coordinates; with unit ranges its first row is the major-axis direction (east, north, up).

Checked equivalence: workflow dip 18, dip azimuth 068, pitch 90 equals GSLIB azimuth 68, dip 18, rake 0. The
GSLIB vertical axis is then normal to the plane and the minor axis lies along strike.

# Grade-shell workflow

Entry point: `python -m deeprbf.workflow.shell settings.json` (or `deeprbf.workflow.shell.run(cfg, base)`).
Requires scikit-image (marching cubes).

## Processing steps

1. Load collar, survey and interval CSVs (`drillholes.load_tables`); missing columns raise
   `<table> table lacks columns [...]; set them in "columns"`.
2. Composite each hole (`drillholes.composite`): fixed length from the first sampled depth, length-weighted mean
   of the sampled part, drop composites with sampled fraction below `min_fraction`, break at hole ends.
3. Desurvey composite mid-points by minimum curvature (`drillholes.desurvey`). Dips with $90 < |dip| \le 90.001$
   are clipped to 90. Holes without collar or survey are dropped and listed.
4. Code ore as value ≥ `cutoff`.
5. Build constraints by method (below), transform to the anisotropic frame, average duplicates, fit
   `NativeExactRBFImplicitModel(kernel, 1.0, nugget/(sill − nugget))` with the chosen drift.
6. Evaluate on a regular grid of `spacing` over `bounds` (or data extent ± `padding`), clip to topography, remove
   bodies below `min_volume`, contour with marching cubes, write Surpac and DXF.
7. Score honouring on all composites; optionally compare with a reference wireframe, run withheld-hole folds and
   build uncertainty shells.

## Methods

| `method` | Constraints | Boundary |
|---|---|---|
| `indicator` | composites coded 1 (ore) / 0 (waste) | field ≥ `iso` |
| `signed_distance` | each composite coded with its anisotropic distance to the nearest composite of the other class, ore negative; in-hole contact mid-points (`distance.signed_distance_codes`) | field ≤ 0 |
| `contacts` | ore/waste contacts along holes at 0; points at ±`delta` along normals estimated by PCA of neighbouring contacts; interval mid-points up to `max_half` (`contacts.extract_contacts`, `estimate_normals`, `constraint_points`, `interval_points`) | field ≤ 0 |

For the distance-type methods the prediction is negated internally so that "inside" is positive, and the iso is
fixed at 0. The contacts method approximates Hermite (gradient) constraints with offset points; true gradient
constraints in the native engine are not implemented. Virtual boreholes are deliberately not implemented.

Normal estimation (`estimate_normals`): `k` = 8 neighbours, at most one contact per hole, at least 3 neighbours
and planarity ≥ 0.5; otherwise the trend normal is used (counted as `normals_fallback`).

## Settings schema

Paths are relative to the settings file. Defaults apply when a key is absent.

| Key | Type | Default | Notes |
|---------------|-----|--------|---------|
| `name` | string | `shell` | output file stem |
| `output_dir` | string | `out` | created if absent |
| `drillholes.collar`, `.survey`, `.intervals` | path | required | CSV |
| `drillholes.value` | string | required | grade column in the interval file |
| `drillholes.columns` | object | `BHID, XCOLLAR, YCOLLAR, ZCOLLAR, AT, DIP, BRG, FROM, TO` | keys `hole, x, y, z, depth, dip, azimuth, from_, to` |
| `drillholes.dip_positive_down` | bool | `true` | `false` negates survey dips |
| `drillholes.composite_length` | m | 1.0 | |
| `drillholes.min_fraction` | 0–1 | 0.5 | |
| `cutoff` | grade | required | ore = value ≥ cutoff |
| `trend.dip`, `.dip_azimuth` | degrees | required | dip, dip azimuth and pitch convention |
| `trend.pitch` | degrees | 90 | |
| `trend.ranges` | [m, m, m] | required | maximum, intermediate, minimum |
| `interpolant.kernel` | string | `spherical` | any kernel string |
| `interpolant.range_factor` | number | 1.0 | 1.04 ≈ spheroidal alpha 9 with `spherical` |
| `interpolant.sill` | number | 1.0 | |
| `interpolant.nugget` | number | 0.0 | 0 ≤ nugget < sill; regularisation = nugget/(sill − nugget) |
| `interpolant.drift` | string | `none` | `none`, `constant`, `linear` |
| `method` | string | `indicator` | `indicator`, `signed_distance`, `contacts` |
| `contacts.max_gap` | m | 2 × composite length | |
| `contacts.k` | int | 8 | |
| `contacts.same_kind` | bool | `true` | |
| `contacts.delta` | m | 2.0 | |
| `contacts.max_half` | m | 10.0 | |
| `surface.iso` | number | 0.5 | indicator only; 0 for distance-type methods |
| `surface.spacing` | m | 2.0 | |
| `surface.min_volume` | m³ | 0 | |
| `surface.padding` | m | 20 | used when `bounds` is absent |
| `surface.bounds` | [[x,y,z],[x,y,z]] | data extent ± padding | |
| `surface.topography` | path or null | null | DXF, Surpac `.str`/`.dtm`, CSV; Datamine `.dm` via `surfaces.register_reader` |
| `uncertainty.quantiles` | list | [0.1, 0.5, 0.9] | signed distance only |
| `uncertainty.folds` | int | 5 | |
| `uncertainty.seed` | int | 0 | |
| `validation.reference` | path | none | `.dtm`, `.str` or `.dxf` |
| `validation.folds` | int | 0 | > 1 enables withheld-hole folds |
| `validation.seed` | int | 0 | |

Topography CSV columns are recognised as X/Y/Z, XP/YP/ZP, EAST/NORTH/RL, EASTING/NORTHING/ELEVATION or XC/YC/ZC.
DXF entities read: 3DFACE, POLYLINE vertices, LWPOLYLINE (with elevation), POINT, LINE.

## Report schema (`<name>_report.json`)

| Block | Fields |
|---|---|
| top level | `name`, `settings` (the settings as read), `seconds`, `engine.native_version` |
| `drillholes` | `holes`, `composites`, `holes_skipped_no_collar_or_survey`, `survey_dips_clipped_to_90`, `composite_length`, `metres`, `share_at_or_above_cutoff` |
| `model` | `method`, `constraints`, `kernel`, `regularization`, `drift`, `backend` (solver), `fit_seconds`; contacts adds `contacts`, `normals_fallback`, `median_planarity` |
| `wireframe` | `files`, `points`, `triangles`, `volume_m3`, `grid` (`lo`, `hi`, `spacing`), `iso`, `topography`, `bodies_found`, `bodies_kept`, `bodies_removed`, `body_volumes_m3` |
| `honouring` | `precision`, `recall`, `f1`, `gold_inside_share`, `mean_inside`, `mean_outside` |
| `reference` | `file`, `volume_m3`, `honouring` (precision, recall, f1, gold_inside_share), `shared_share_of_reference`, `shared_share_of_deeprbf`, `grid_volume_reference_m3`, `grid_volume_deeprbf_m3` |
| `withheld_holes` | `folds`, `precision`, `recall`, `f1`, `baseline_f1_all_ore`, `gold_inside_share`, `note` |
| `uncertainty` | `method`, `C`, `label`, `coverage`, `shells` (per label: `quantile`, `C`, `volume_m3`, `bodies`, `files`) |

Definitions, with "inside" meaning inside the shell at the composite location and "ore" value ≥ cut-off:
precision = ore inside / all inside; recall = ore inside / all ore; F1 = 2PR/(P + R);
`gold_inside_share` = Σ value inside / Σ value; `baseline_f1_all_ore` = 2p/(1 + p) with p the ore share.
Reference overlap volumes are counted on a grid of twice the contouring spacing.

Withheld-hole folds assign whole holes to folds and predict each fold from a model fitted without it. They score
the trend and settings on unseen holes, as support for geometry decisions; the wireframe always uses all holes.

Uncertainty shells: for each quantile q, the shift C is the q-quantile of the withheld-hole field values at true
contacts, so the boundary at C reaches or passes a fraction q of unseen contacts (`coverage`). Shells are written as
`<name>_<label>` with label `conservative` (q < 0.5), `expected` (q = 0.5) or `aggressive` (q > 0.5). This differs from the bootstrap
of the mean code in McLennan and Deutsch (2006).

## Mesh and surface utilities (`deeprbf.workflow.mesh`, `surfaces`)

| Function | Purpose |
|---|---|
| `grid_axes(lo, hi, spacing)`, `evaluate(predict, lo, hi, spacing, slab=1)` | grid and field evaluation |
| `clip_topography(f, ax, topo_xyz, iso)`, `filter_small(f, spacing, iso, min_volume)` | clipping, small-body removal |
| `contour(f, ax, iso)`, `volume(V, F)` | marching cubes, enclosed volume |
| `write_surpac(V, F, stem, name)`, `write_dxf(V, F, path, layer)` | output |
| `read_surpac(dtm, str)`, `read_dxf(path)`, `read_wireframe(path)` | input |
| `inside(T, P, cell=10.0)` | point-in-solid test by vertical-ray parity |

Surpac string lines are written as `string, Y, X, Z`; DTM triangles index string points (1-based).

# Geostatistics (GSLIB conventions)

## Variogram objects

`GSLIBStructure(type='spherical', cc=1.0, range_major=100, range_minor=100, range_vertical=100, azimuth=0,
dip=0, rake=0)`: one nested structure; `type` is `spherical`, `exponential` or `gaussian`, `cc` its contribution.

`GSLIBVariogram(nugget=0.0, structures=[])`: nugget plus structures. `sill()` returns nugget + Σ cc;
`covariance(a, b)` the covariance between two points.

`SearchEllipsoid(radius_major=200, radius_minor=200, radius_vertical=200, azimuth=0, dip=0, rake=0)`.

## Functions

| Function | Signature (defaults) | Returns |
|---|---|---|
| `kriging_gslib` | `(sample_xyz, sample_values, target_xyz, variogram, search, max_data=24, min_data=1, kriging_type='ordinary', mean=0.0)` | dict `estimate`, `variance`, `n_samples`; NaN where fewer than `min_data` |
| `sgs_gslib` | `(sample_xyz, sample_values, target_xyz, variogram, search, n_realizations=10, max_data=16, max_nodes=16, min_data=0, kriging_type='simple', normal_score=True, zmin=nan, zmax=nan, multigrid_levels=3, seed=69069)` | array (targets × realisations), original units |
| `sis_gslib` | `(sample_xyz, sample_categories, target_xyz, categories, variograms, search, n_realizations=10, max_data=16, max_nodes=16, multigrid_levels=3, seed=69069)` | int array (targets × realisations) |
| `normal_score` | `(values, zmin=nan, zmax=nan)` | normal scores; ties share a score |
| `gslib_rotation` | `(azimuth, dip, rake, range_major, range_minor, range_vertical)` | 9 values, row-major 3×3 |

`kriging_type` is `ordinary` or `simple` (simple uses `mean`). SGS follows `sgsim`: normal-score transform of the
data, simple kriging of normal scores on a random multiple-grid path (`multigrid_levels`), previously simulated
nodes used up to `max_nodes`, back-transform with tails bounded by `zmin`/`zmax`. SIS follows `sisim`: simple
indicator kriging on global proportions with one standardised variogram per category, order-relation correction,
multiple-grid path. Coordinates are lists of `[x, y, z]`; NumPy arrays can be passed with `.tolist()`.

## Other geostatistics in the native module

Available and tested, but not described in the software paper; use `help(deeprbf_recovered.<name>)` for full
signatures: `ordinary_kriging` (and `_anisotropic`, `_nested`, `_local`, `_with_range`), `block_kriging`,
`audited_ordinary_kriging`, `indicator_kriging` (and `_anisotropic`), `ordinary_cokriging`,
`sequential_indicator_cokriging`, `run_sgs`, `run_sgs_summary`, `run_sgs_with_range`, `run_sis`,
`run_sis_anisotropic` (the 3.3.x simulation interface), `simulate_gaussian_native`, `gibbs_truncated_native`,
`vecchia_truncated_native`, `snesim_native_static`, variogram fitting (`fit_spherical_variogram`,
`fit_exponential_variogram`, `fit_gaussian_variogram`, `fit_nested_variogram`, `auto_fit_variogram`,
`fit_local_variogram_spherical`, `fit_local_variograms_batch`), cross and indicator variograms
(`compute_direct_indicator_variograms_fast`, `compute_cross_indicator_variograms_fast`), the linear model of
coregionalisation (`build_lmc_from_variograms`, `fit_lmc_from_experimental_variograms`, `is_positive_definite`),
change of support (`affine_correction`, `lognormal_correction`, `compute_variance_reduction_factor`,
`apply_change_of_support_to_realizations`), cross-validation (`cross_validate_ordinary_kriging`,
`cross_validate_indicator_kriging`), `select_neighborhood`, `categorical_rbf_native`,
`indicator_lithology_native`, `solve_dense_linear_system`, `spherical_covariance`,
`anisotropic_scaled_distance`.

# Native interface: `NativeExactRBFImplicitModel`

```python
import numpy as np
import deeprbf_recovered as d

rng = np.random.default_rng(0)
P = rng.uniform(0, 10, (3000, 3))
v = (P[:, 2] > 5).astype(float)

m = d.NativeExactRBFImplicitModel("spherical", 2.0, 0.01)   # kernel, epsilon, regularisation
m.set_drift("constant")                                     # before fit
m.fit(P.tolist(), v.tolist())
f = np.asarray(m.predict(P[:10].tolist()))
print(m.solver_backend(), m.last_predict_backend(), m.n_centers())
```

| Member | Description |
|---|---|
| `NativeExactRBFImplicitModel()`, `(kernel, epsilon, regularization)` | construct |
| `set_drift(drift)` / `drift()` | `none`, `constant`, `linear` (default); call before `fit` |
| `set_kernel_shape(alpha, power)` | generalised Cauchy shape; call before `fit` |
| `fit(centers, values)` | solve; chooses the solver (see *Solvers*) |
| `fit_with_reduction(centers, values, fitting_accuracy=1e-4, initial_ratio=0.1, append_ratio)` | greedy centre reduction |
| `predict(points)` | list of values |
| `kernel()`, `epsilon()`, `regularization()`, `n_centers()`, `is_fitted()` | properties |
| `solver_backend()`, `sparse_solve_info()` | solver used and sparse statistics |
| `last_predict_backend()`, `scalfmm_status()` | backend of the last `predict` |
| static `set_predict_backend(b)`, `predict_backend()` | `auto`, `direct`, `scalfmm` |
| static `set_dense_direct_limit(n)`, `set_dense_memory_limit_mb(mb)` | dense thresholds (8000, 1024 MB) |
| static `set_sparse_solver(force)`, `sparse_solver_forced()` | force sparse path for compact kernels |
| static `force_iterative_solver(force)`, `set_iterative_solver(restart=150, max_iter=3000, tol=1e-6)` | iterative solver |
| static `set_hmatrix(mode, tol=1e-9, eta=1.5, leaf_size=64, near_memory_mb=2048)`, `hmatrix_mode()` | H-matrix product in the iterative solver (3.5.0) |
| `iterative_info()` | statistics and exact residual check of the last iterative fit (3.5.0) |
| `weights()` | fitted coefficients: $n$ kernel weights, then the drift coefficients (3.5.0) |

`HMatrix(points, kernel, epsilon, tol=1e-9, eta=1.5, leaf_size=64, near_memory_mb=2048)` exposes the H-matrix on its
own: `apply(x)` returns $\Phi x$ (no regularisation), `stats()` its block counts, ranks, memory and build time.

Module functions: `kernel_value`, `kernel_table_check`, `set_openmp_threads(n)` (returns `False` without OpenMP),
`get_openmp_threads()`, `deeprbf_scalfmm_runtime_status()`, `deeprbf_scalfmm_diagnostics()`, `__version__`.

Other model classes in the module (not covered in the paper): `NativeGeologicalImplicitModel`,
`NativeTrendImplicitModel`, `NativeCategoricalModel`.

# Python package `deeprbf`

| Module | Content | Status |
|---|---|---|
| `deeprbf.workflow` | grade-shell workflow: `shell`, `drillholes`, `anisotropy`, `distance`, `contacts`, `mesh`, `surfaces` | described in the paper |
| `deeprbf.normal_score` | `NormalScoreTransform`, `run_sgs_normal_score` | available |
| `deeprbf.simulation` | turning bands, direct block simulation (`simulate_tb`, `simulate_dbs`) | available |
| `deeprbf.lithology`, `sparse_pgs`, `ensemble` | lithology IK, SIS, (sparse) plurigaussian simulation, ensembles | available |
| `deeprbf.snesim`, `training_images`, `geological_priors` | multiple-point simulation | available |
| `deeprbf.categorical`, `domain_rbf`, `contact_rbf`, `paired_contact_surface`, `paired_lens` | categorical and contact models | available |
| `deeprbf.grade`, `resource`, `recoverable` | `GradeEstimator`, `LocalEstimator`, recoverable moments | available |
| `deeprbf.desurvey`, `io` | `minimum_curvature`, `export_leapfrog_csv` | available |
| `deeprbf.svgp` | sparse variational Gaussian processes (needs PyTorch) | optional |

"Available" modules are tested and kept for compatibility, but they are not part of the software paper and their
interfaces may change in minor releases. `deeprbf.runtime_status()` reports the package and backend status.

# Building from source

Requirements: CMake ≥ 3.20, a C++20 compiler (MSVC 2026 or GCC ≥ 11), Eigen 3.4, pybind11, NumPy, Python 3.10+.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEEPRBF_EIGEN_ROOT=<eigen3> -DDEEPRBF_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

| Option | Default | Effect |
|---|---|---|
| `DEEPRBF_BUILD_PYTHON` | ON | build the pybind11 module |
| `DEEPRBF_BUILD_TESTS` | OFF | build the native tests |
| `DEEPRBF_PORTABLE_TESTS` | ON | native core tests without Python or ScalFMM |
| `DEEPRBF_ENABLE_OPENMP` | ON | OpenMP |
| `DEEPRBF_ENABLE_MKL` / `DEEPRBF_REQUIRE_MKL` | OFF | Intel oneMKL; `DEEPRBF_MKL_ROOT` |
| `DEEPRBF_ENABLE_SCALFMM` / `DEEPRBF_REQUIRE_SCALFMM` | OFF | ScalFMM 3; `DEEPRBF_SCALFMM_ROOT` |
| `DEEPRBF_SCALFMM_FORCE_SEQUENTIAL` | OFF | sequential FMM traversal |
| `DEEPRBF_STRICT_SCALFMM_RUNTIME` | OFF | strict ScalFMM runtime checks |
| `DEEPRBF_FAST_MATH` | OFF | fast-math compiler flags |
| `DEEPRBF_NATIVE_ARCH` | OFF | `-march=native` (non-MSVC) |
| `DEEPRBF_USE_NANOFLANN` | OFF | bundled nanoflann KD-tree |
| `DEEPRBF_EIGEN_ROOT` | (search) | Eigen include path; also `EIGEN3_ROOT` |

`pip wheel . --no-build-isolation` builds the portable wheel (OpenMP only) through `setup.py`. Results are
reproducible across platforms: the Kilamongo shells give identical volumes and body counts on Windows (MSVC) and
Linux (GCC 13).

# Testing

`ctest` runs 19 native tests, including a numerical regression test. `python -m pytest tests` runs 280 Python
tests with the built module on `PYTHONPATH`; `test_svgp.py` needs PyTorch and Windows-only bootstrap tests skip
elsewhere. `benchmarks/` holds the scripts and raw results behind the paper's timing and accuracy figures
(`bench_deeprbf.py` B0–B7, `bench_geostats.py` G1–G3).

# Versioning

Releases follow semantic versioning. The version is set in `CMakeLists.txt` (`DEEPRBF_VERSION_*`),
`pyproject.toml`, `deeprbf/__init__.py` and `CITATION.cff`, and must agree. On import, `deeprbf` compares its
version with `deeprbf_recovered.__version__` and raises
`ImportError: DeepRBF Python X requires native X; loaded Y` if they differ. Each GitHub release is archived by
Zenodo with its own DOI.

| Version | Notes |
|---|---|
| 3.4.0 | first public source release (DOI 10.5281/zenodo.24232967) |
| 3.4.1 | signed-distance example uses a constant drift; Windows-only test skipped elsewhere (DOI 10.5281/zenodo.24241552) |
| 3.4.2 | regularisation sign fixed for the linear and multiquadric kernels; B3/B5 benchmark additions; these manuals (DOI 10.5281/zenodo.24246601) |
| 3.5.0 | tabulated kernel evaluation; H-matrix/ACA product with hybrid preconditioner and exact residual check; benchmark B7 |

# References

Deutsch, C.V., Journel, A.G., 1998. GSLIB: Geostatistical Software Library and User's Guide, 2nd ed. Oxford
University Press, New York.

McLennan, J.A., Deutsch, C.V., 2006. BOUNDSIM: implicit boundary modeling. Centre for Computational
Geostatistics Report 8, University of Alberta.

Wang, J., Zhao, H., Bi, L., Wang, L., 2018. Implicit 3D modeling of ore body from geological boundary data
using Hermite radial basis functions. Minerals 8(10), 443. https://doi.org/10.3390/min8100443
