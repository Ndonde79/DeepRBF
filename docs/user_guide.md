---
title: "DeepRBF User Guide"
subtitle: "Grade shells and geostatistics from drillhole data. Version 3.4.2"
author: "Ramadhani W. Ndonde, DeepFusion Geoscience Limited"
date: "November 2026"
---

# About this guide

This guide is for resource geologists who want to build grade shells and run kriging or simulation with DeepRBF.
It assumes you know your deposit, your drillhole database and the usual implicit-modelling settings (trend, ranges,
nugget, iso value). It assumes very little Python: the shell workflow runs from one command and one settings file,
and the geostatistics section gives complete scripts you can copy and edit.

The [Technical Reference](technical_reference.md) describes the engine itself: kernels, solvers, every setting and
report field, the full Python and native interface, and build options.

> **Models built with DeepRBF are drafts for Competent Person review.** DeepRBF builds the geometry the data and your
> settings imply. The geological interpretation, the choice of cut-off, trend and domains, and any statement of
> Mineral Resources remain the responsibility of the Competent Person.

## What DeepRBF does

- **Grade shells from drillholes.** Collar, survey and assay CSV files plus one JSON settings file give a closed
  wireframe at your cut-off: Surpac `.str`/`.dtm`, DXF, and a JSON report. Three methods are available: indicator,
  signed distance and contacts.
- **Geostatistics.** Ordinary and simple kriging, sequential Gaussian simulation (SGS) and sequential indicator
  simulation (SIS), with GSLIB-style variograms and search ellipsoids.
- **Speed.** Fitting 18,000 composites with a compact (spherical) kernel takes about 40 seconds on a two-core computer, because
  the engine uses a sparse solver for compact kernels.

# Installing

## Windows (release wheel)

You need 64-bit Python 3.11. Any Python 3.11 works (python.org, Anaconda or Miniconda). With conda, a separate
environment keeps DeepRBF apart from other tools:

```
conda create -n deeprbf python=3.11
conda activate deeprbf
```

Download `deepfusion_deeprbf-3.4.2-cp311-cp311-win_amd64.whl` and `SHA256SUMS` from the
[latest release](https://github.com/Ndonde79/DeepRBF/releases/latest). Check the file before installing: the hash
printed by `certutil` must match the line for the wheel in `SHA256SUMS`.

```
certutil -hashfile deepfusion_deeprbf-3.4.2-cp311-cp311-win_amd64.whl SHA256
pip install "deepfusion_deeprbf-3.4.2-cp311-cp311-win_amd64.whl[workflow]"
```

The `[workflow]` part also installs scikit-image, which the shell workflow uses to contour the wireframe.

Check the installation:

```
python -c "import deeprbf, deeprbf_recovered as n; print(deeprbf.__version__, n.__version__)"
```

Both numbers must read `3.4.2`. DeepRBF refuses to start if the Python package and the compiled engine come from
different versions; reinstall the wheel if they differ.

## Linux and macOS

There is no release wheel for these platforms yet. Build from source as described in the README and the Technical
Reference (section *Building from source*). The engine builds with GCC 11 or later and gives the same results as the
Windows wheel.

# Preparing drillhole data

The workflow reads three CSV files. Column names are yours: you map them in the settings file.

| File | Required columns | Kilamongo example |
|---|---|---|
| Collar | hole ID, X, Y, Z | `BHID, XCOLLAR, YCOLLAR, ZCOLLAR` |
| Survey | hole ID, depth, dip, azimuth | `BHID, AT, DIP, BRG` |
| Intervals (assays) | hole ID, from, to, grade | `BHID, FROM, TO, AU` |

Points to check before the first run:

- **Coordinates in metres** in one projected system (for example UTM). Ranges, spacing and volumes are all in metres.
- **Dip sign.** By default a positive dip points down the hole (a hole drilled at −60° is entered as 60). If your
  survey file stores downward dips as negative numbers, set `"dip_positive_down": false`.
- **Every hole needs a collar and at least one survey record.** Holes without either are skipped, and the report
  lists them under `holes_skipped_no_collar_or_survey`. Read that list after every run.
- **Unsampled intervals** are left blank (or omitted). Do not enter them as zero grade unless they really are waste
  samples: a zero is treated as an assay.
- **Survey dips just past 90°** (for example 90.000002, from rounding) are clipped to 90 and counted in
  `survey_dips_clipped_to_90`.

## Compositing and desurvey

DeepRBF composites each hole to fixed lengths (`composite_length`, default 1 m) starting at the first sampled depth.
Each composite takes the length-weighted mean of the sampled part. A composite with less than `min_fraction`
(default 0.5) of its length sampled is dropped. Composites do not cross the end of a hole. Composite mid-points are
located by minimum-curvature desurvey.

The composites are written to `<name>_composites.csv` with columns `hole, from, to, value, sampled, X, Y, Z,
indicator`. `indicator` is 1 where the grade is at or above the cut-off. Use this file to check the compositing and
as input for kriging (see *Kriging and simulation*).

# The settings file

One JSON file drives a run. File paths in it are relative to the settings file. This is the Kilamongo indicator
file shipped in `examples/kilamongo/kilamongo_indicator.json`:

```json
{
 "name": "Kilamongo_Au_0.3_indicator",
 "drillholes": {
  "collar": "collar.csv", "survey": "survey.csv", "intervals": "assay.csv", "value": "AU",
  "columns": {"hole": "BHID", "x": "XCOLLAR", "y": "YCOLLAR", "z": "ZCOLLAR",
              "depth": "AT", "dip": "DIP", "azimuth": "BRG", "from_": "FROM", "to": "TO"},
  "dip_positive_down": true, "composite_length": 1.0, "min_fraction": 0.5
 },
 "cutoff": 0.3,
 "trend": {"dip": 17.975221077, "dip_azimuth": 67.69, "pitch": 90,
           "ranges": [52.6323, 43.8603, 11.6961]},
 "interpolant": {"kernel": "spherical", "range_factor": 1.04, "sill": 12.0,
                 "nugget": 0.30901, "drift": "none"},
 "surface": {"iso": 0.25, "spacing": 2.0, "min_volume": 1000.0,
             "bounds": [[621590, 9801940, 1288], [622090, 9802390, 1446.6]],
             "topography": null},
 "validation": {"folds": 0, "reference": "Au_Ind_kila_0.3.dtm"},
 "output_dir": "out"
}
```

The sections below explain each block. The Technical Reference has the complete list of keys with their defaults.

## `name` and `output_dir`

`name` becomes the file name of every output (default `shell`). `output_dir` is created if it does not exist
(default `out`).

## `drillholes`

`collar`, `survey` and `intervals` are the three CSV files. `value` is the grade column in the interval file.
`columns` maps your column names; any key you leave out falls back to the Kilamongo names shown above. Note the key
`from_`, with an underscore. `dip_positive_down`, `composite_length` and `min_fraction` are described under
*Preparing drillhole data*.

## `cutoff`

The grade that separates ore from waste: a composite is ore when its grade is **at or above** the cut-off.

## `trend`: the anisotropy

The trend uses the dip, dip azimuth and pitch convention of common implicit-modelling software, so you can copy it
from an existing interpolant:

- `dip` and `dip_azimuth` define the plane of greatest continuity.
- `pitch` is the direction of the maximum axis within that plane, measured from the strike line (dip azimuth − 90°)
  towards down-dip. `pitch: 90` (the default) puts the maximum axis straight down-dip.
- `ranges` are `[maximum, intermediate, minimum]` in metres. The minimum range is across the plane.

For Kilamongo the zone dips 18° towards 068°, with the longest continuity down-dip.

## `interpolant`

| Key | Meaning | Default |
|---|---|---|
| `kernel` | `spherical`, `exponential` or `gen_cauchy` (see below) | `spherical` |
| `range_factor` | multiplies all three ranges before fitting | 1.0 |
| `sill` | total sill of the indicator model | 1.0 |
| `nugget` | nugget; must satisfy 0 ≤ nugget < sill | 0.0 |
| `drift` | `none`, `constant` or `linear` | `none` |

**Kernel.** Use `spherical` by default. It has a finite range, which lets DeepRBF use its sparse solver: this is
what makes large drillhole sets fast. `exponential` and `gen_cauchy` have long tails; they give smoother shells
far from data but every point influences every other, so large runs are much slower. `gen_cauchy:9:1.667` sets the
two shape parameters (alpha and power) explicitly.

**Nugget and sill.** DeepRBF does not fit a variogram here. It uses the ratio nugget / (sill − nugget) as the
smoothing (regularisation) of the interpolant. With the Kilamongo values, 0.309 / (12 − 0.309) = 0.026. A larger
nugget gives a smoother shell that honours individual composites less closely.

**range_factor.** With the spherical kernel, `range_factor: 1.04` reproduces the shape of the spheroidal
interpolant with alpha 9 used in commercial software (a close fit, not an exact match). Use 1.0 when you are not
trying to match such an interpolant.

**Drift.** Use `none` for the indicator method, `constant` for signed distance and `linear` for contacts. The
*Choosing a method* section explains why.

## `surface`: the wireframe

| Key | Meaning | Default |
|---|---|---|
| `iso` | contour value of the field (indicator method only) | 0.5 |
| `spacing` | grid spacing in metres for contouring (resolution) | 2.0 |
| `min_volume` | bodies smaller than this (m³) are removed | 0 |
| `bounds` | `[[xmin, ymin, zmin], [xmax, ymax, zmax]]` | data extent + `padding` |
| `padding` | margin around the data when `bounds` is not given (m) | 20 |
| `topography` | surface file to clip the shell against, or `null` | `null` |

**iso.** For the indicator method the field runs from about 0 (waste) to 1 (ore); 0.5 is the neutral boundary.
A lower iso gives a larger, more inclusive shell. Kilamongo uses 0.25.
An iso calibrated so that the shell volume matches the declustered share of composites at or above the cut-off
usually recovers the geometry best (benchmark B5). For signed distance and contacts the boundary is always at 0 and `iso` is ignored.

**spacing.** 2 m suits a deposit drilled at 20–40 m spacing. Halving the spacing multiplies the grid size by eight;
run at a coarser spacing while you experiment.

**Topography.** Give a DXF (3DFACE, POLYLINE, LWPOLYLINE, POINT or LINE entities), a Surpac `.str`/`.dtm` pair or a
CSV of points with X/Y/Z columns (also accepted: `XP/YP/ZP`, `EAST/NORTH/RL`, `EASTING/NORTHING/ELEVATION`,
`XC/YC/ZC`). Everything above the surface is removed from the shell.

## `validation`

| Key | Meaning |
|---|---|
| `reference` | an existing wireframe (Surpac `.dtm` or `.str`, or DXF) to compare against, for example a shell from other software |
| `folds` | number of withheld-hole folds; 0 or 1 switches the test off |
| `seed` | random seed for assigning holes to folds |

See *Reading the report* for what these numbers mean and how to use them.

## `method` and its options

`"method"` is `indicator` (default), `signed_distance` or `contacts`. The contacts method takes an optional
`"contacts"` block (`delta`, `max_half`, `max_gap`, `k`, `same_kind`) and the signed-distance method an optional
`"uncertainty"` block. They are described in the next section.

# Choosing a method

All three methods use the same trend, kernel and surface settings. They differ in what the interpolant is fitted to.

## Indicator (default)

Each composite is coded 1 (ore) or 0 (waste) and the field is contoured at `iso`. This is the familiar indicator
RBF. Use it first: it is robust, and the iso value gives you direct control over how inclusive the shell is.

Settings: `"drift": "none"`, `iso` between 0.25 and 0.5.

## Signed distance (BOUNDSIM)

Each composite is coded with its anisotropic distance to the nearest composite of the other class: ore negative,
waste positive, and contacts within a hole at the mid-point between the two samples. The shell is the zero
surface. The method follows the BOUNDSIM approach of McLennan and Deutsch (2006). It tends to give tighter shells
than the indicator method, closer to the drilled contacts.

Settings: **`"drift": "constant"`**. A linear drift lets the field rise or fall steadily away from the data and can
produce a spurious closed surface along the edge of the model box; with no drift at all the field relaxes towards
zero between holes, which can join everything into one large body. A constant drift avoids both.

Optional uncertainty shells:

```json
"uncertainty": {"quantiles": [0.1, 0.5, 0.9], "folds": 5, "seed": 0}
```

This writes three more wireframes, `<name>_conservative`, `<name>_expected` and `<name>_aggressive`, for the
quantiles 0.1, 0.5 and 0.9. Each boundary is shifted by an amount measured at true contacts in withheld holes: the
aggressive (0.9) shell reaches or passes 90% of the contacts that were not used to build it, the conservative (0.1)
shell 10%. Any quantile below 0.5 is labelled conservative and any above 0.5 aggressive, so use at most one of each. Treat the spread as a measure of how well the drilling
fixes the boundary, not as a probability of tonnes. At Kilamongo the three shells are 0.50, 3.27 and 30.5 Mm³ (against 0.78 Mm³ for the
unshifted shell): between holes the drilling fixes the boundary only loosely, and the aggressive shell grows into a
single body that fills much of the model box. A spread that wide says where infill drilling would pay. The
`expected` shell is not the same as the unshifted shell: it moves the boundary to the median position of unseen
contacts.

## Contacts

DeepRBF finds the ore/waste contacts along each hole, estimates the orientation of the contact surface at each one
from the contacts in neighbouring holes, and fits a surface through the contacts with points a short distance
either side (after Wang et al., 2018). Use it when the contacts are sharp and well defined and you want the
shell to pass through them.

Settings: `"drift": "linear"`. Optional block:

| Key | Meaning | Default |
|---|---|---|
| `delta` | offset of the inside/outside points from each contact (m) | 2.0 |
| `max_half` | furthest interval mid-point used, in metres from the contact | 10.0 |
| `max_gap` | longest gap between contacts treated as one surface (m) | 2 × composite length |
| `k` | neighbouring contacts used to estimate each orientation | 8 |
| `same_kind` | only use contacts of the same kind (top or bottom) for orientation | `true` |

The report gives the number of contacts and how many orientations fell back to the trend because the neighbouring
contacts were not planar enough (`normals_fallback`).

## Kilamongo results

The three example settings files give these results at 0.3 g/t (Linux and Windows builds give the same volumes):

| | Indicator | Signed distance | Contacts |
|---|---|---|---|
| Volume | 3.29 Mm³ | 0.78 Mm³ | 2.51 Mm³ |
| Bodies kept | 23 | 41 | 25 |
| Precision / recall | 0.74 / 0.98 | 0.93 / 0.76 | 0.61 / 0.81 |
| F1 (all composites) | 0.84 | 0.84 | 0.70 |
| Gold inside the shell | 94% | 80% | 83% |
| Share of the reference volume covered | 90% | 27% | 62% |
| Run time (two-core test machine) | 54 s | 48 s | 16 s |

The reference shell is 2.67 Mm³. The three methods answer different questions: the indicator shell is
inclusive, the signed-distance shell is tight around the ore composites, and the contacts shell follows the drilled
contacts. Compare them on section against the geology before choosing.

# Running

Open a command prompt (Anaconda Prompt if you use conda), activate the environment, go to the folder with the
settings file and run:

```
cd examples\kilamongo
python -m deeprbf.workflow.shell kilamongo_indicator.json
```

The console shows a summary:

```
Kilamongo_Au_0.3_indicator: 17958 composites, 23 bodies, 3.29 Mm3, F1 0.8412,
  gold inside 94.1%, 54.0 s
reference: F1 0.8194, shared 90% of reference volume
outputs: out/Kilamongo_Au_0.3_indicator.str, ...indicator.dtm, ...indicator.dxf
```

## Outputs

| File | Content |
|---|---|
| `<name>.str`, `<name>.dtm` | the shell as a Surpac string file and DTM |
| `<name>.dxf` | the same shell as a DXF mesh (most mining software reads it) |
| `<name>_composites.csv` | composites with coordinates and the ore indicator |
| `<name>_field.npy` | the interpolated field on the contouring grid (NumPy array) |
| `<name>_report.json` | settings, data summary, model, wireframe and honouring statistics |
| `<name>_conservative.*`, `_expected.*`, `_aggressive.*` | uncertainty shells (signed distance with `uncertainty` only) |

Import the `.dtm`/`.str` pair or the DXF into your modelling package and check it against the drillholes on
section before using it.

# Reading the report

The report is a JSON file you can open in any text editor. Its main blocks, with the Kilamongo indicator values:

**`drillholes`**: 402 holes, 17,958 composites, no holes skipped, 120 survey dips clipped to 90°, 19.5% of composites
at or above the cut-off. Check `holes_skipped_no_collar_or_survey` first: a skipped hole is a hole the shell knows
nothing about.

**`model`**: the method, kernel, regularisation (0.026), drift, the solver used (`sparse_cholesky_schur`) and the fit
time (40 s).

**`wireframe`**: volume (3,293,495 m³), triangle count, the grid, and the bodies: 58 found, 35 removed by
`min_volume`, 23 kept, with the volume of each. A long list of small bodies usually means the iso or nugget is too
low, or the trend does not match the geology.

**`honouring`**: how well the shell agrees with the composites it was built from.

| Field | Meaning | Kilamongo |
|---|---|---|
| `precision` | share of composites inside the shell that are ore | 0.74 |
| `recall` | share of ore composites that are inside the shell | 0.98 |
| `f1` | balance of the two (harmonic mean) | 0.84 |
| `gold_inside_share` | share of the total metal (grade × length) inside the shell | 0.94 |
| `mean_inside`, `mean_outside` | mean composite grade inside and outside (g/t) | 1.93, 0.04 |

High recall with lower precision means an inclusive shell that takes in some waste; the reverse means a tight
shell that leaves ore outside. `iso` (indicator) moves the balance.

**`reference`**: the same statistics for the reference wireframe, both volumes, and the overlap:
`shared_share_of_reference` is the share of the reference volume that the new shell also covers (0.90 here) and
`shared_share_of_deeprbf` the share of the new shell covered by the reference (0.73).

**`withheld_holes`** (when `folds` > 1): the holes are split into folds; each fold is predicted from a model built
without it, and the same precision, recall and F1 are reported for these unseen holes, beside
`baseline_f1_all_ore` (the F1 you would get by calling everything ore).

> **How to use the numbers.** The wireframe is always built from all holes: the geometry of the mineralisation should
> use all the data. Honouring statistics tell you whether the shell respects the composites. The withheld-hole scores
> are support for geometry decisions, such as choosing between two trends or two range sets, or judging where
> more drilling would change the shell. They are not a measure of the quality of a resource model, and neither is the
> honouring.

# Kriging and simulation

The native engine provides GSLIB-style point kriging, SGS and SIS. They take NumPy-style lists of coordinates and
values, so you can feed them the composites file written by the shell workflow. The script below is
`docs/examples/krige_inside_shell.py`; run it from `examples/kilamongo` after the indicator run.

```python
# Krige Au inside the 0.3 g/t shell, then run SGS and SIS (GSLIB conventions).
import numpy as np, pandas as pd
import deeprbf_recovered as d
from deeprbf.workflow import mesh

comp = pd.read_csv("out/Kilamongo_Au_0.3_indicator_composites.csv")
xyz = comp[["X", "Y", "Z"]].to_numpy()
au = comp["value"].to_numpy()

# 5 m block centroids inside the shell
shell = mesh.read_wireframe("out/Kilamongo_Au_0.3_indicator.dtm")
lo, hi = xyz.min(0), xyz.max(0)
gx, gy, gz = (np.arange(lo[i] + 2.5, hi[i], 5.0) for i in range(3))
grid = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(3, -1).T
blocks = grid[mesh.inside(shell, grid)]
inside = comp["indicator"].to_numpy() == 1          # composites at or above the cut-off

# Variogram: nugget + one spherical structure, GSLIB angles (azimuth, dip, rake)
vario = d.GSLIBVariogram(0.3, [d.GSLIBStructure("spherical", 0.7, 52.6, 43.9, 11.7,
                                                  azimuth=68.0, dip=18.0, rake=0.0)])
search = d.SearchEllipsoid(100.0, 80.0, 25.0, azimuth=68.0, dip=18.0, rake=0.0)

ok = d.kriging_gslib(xyz[inside].tolist(), au[inside].tolist(), blocks.tolist(), vario, search,
                     max_data=24, min_data=4, kriging_type="ordinary")
est = np.asarray(ok["estimate"])
print(f"{len(blocks)} blocks, mean OK estimate {np.nanmean(est):.2f} g/t, "
      f"unestimated {int(np.isnan(est).sum())}")

# Ten SGS realisations (normal-score transform and back-transform inside)
sim = d.sgs_gslib(xyz[inside].tolist(), au[inside].tolist(), blocks.tolist(), vario, search,
                  n_realizations=10, seed=69069)
print("SGS", sim.shape, f"mean {sim.mean():.2f} g/t")

# SIS of ore / waste over the whole grid, one standardised variogram per category
cat = comp["indicator"].to_numpy().astype(int)
v1 = d.GSLIBVariogram(0.1, [d.GSLIBStructure("spherical", 0.9, 52.6, 43.9, 11.7,
                                                68.0, 18.0, 0.0)])
real = d.sis_gslib(xyz.tolist(), cat.tolist(), blocks.tolist(), [0, 1], [v1, v1], search,
                   n_realizations=10, seed=69069)
print("SIS", real.shape, f"P(ore) mean {(real == 1).mean():.2f}")
```

The variogram here is for illustration only. Model your own variogram from the composites inside the domain
before estimating.

## Converting a workflow trend to GSLIB angles

The shell workflow uses dip, dip azimuth and pitch; the kriging functions use GSLIB angles: `azimuth` of the major axis
(clockwise from north), `dip` of the major axis (positive downward in DeepRBF) and `rake` (rotation about the major
axis). The ranges are `range_major`, `range_minor` (the horizontal axis at right angles to the major) and
`range_vertical`.

For the common case of `pitch: 90` (maximum axis down-dip), the conversion is direct:

| Workflow trend | GSLIB (DeepRBF) |
|---|---|
| dip 18, dip azimuth 068, pitch 90 | azimuth 68, dip 18, rake 0 |
| ranges max, int, min = 52.6, 43.9, 11.7 | major 52.6, minor 43.9, vertical 11.7 |

For other pitches, check the result with `d.gslib_rotation(azimuth, dip, rake, 1, 1, 1)`: its first row is the
direction of the major axis as an (east, north, up) vector.

## Results

`kriging_gslib` returns a dictionary with `estimate`, `variance` and `n_samples` (one value per target).
Targets with fewer than `min_data` samples in the search ellipsoid get `NaN`. `sgs_gslib` returns an array of
shape (targets, realisations) in original grade units; `sis_gslib` returns the same shape with category codes.

# Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `need 0 <= nugget < sill` | the nugget is negative or not smaller than the sill |
| `unknown method ...` | `method` must be `indicator`, `signed_distance` or `contacts` |
| `unknown RBF kernel: ...` | check the kernel spelling, e.g. `thin_plate`, not `thinplate` |
| `set_drift("none"): kernel "linear" is not positive definite` | `linear`, `cubic`, `thin_plate` and `multiquadric` need a drift; use `constant` or `linear` |
| `DeepRBF Python 3.4.2 requires native 3.4.2; loaded ...` | the Python package and the engine come from different releases; reinstall the wheel |
| `No module named skimage` | install with the `[workflow]` extra, or `pip install scikit-image` |
| `survey table lacks columns [...]; set them in "columns"` | a name in `drillholes.columns` does not match your CSV header (names are case-sensitive) |
| holes listed in `holes_skipped_...` | those holes have no collar or no survey record; fix the database |
| one huge body touching the box edge | signed distance with `linear` or `none` drift; use `constant` |
| many tiny bodies | raise `min_volume`, raise `iso`, or check the trend against the geology |
| run is very slow | a long-tailed kernel (`exponential`, `gen_cauchy`) on many composites; use `spherical`, or a coarser `spacing` while testing |

# References

McLennan, J.A., Deutsch, C.V., 2006. BOUNDSIM: implicit boundary modeling. Centre for Computational
Geostatistics Report 8, University of Alberta.

Wang, J., Zhao, H., Bi, L., Wang, L., 2018. Implicit 3D modeling of ore body from geological boundary data
using Hermite radial basis functions. Minerals 8(10), 443. https://doi.org/10.3390/min8100443

Deutsch, C.V., Journel, A.G., 1998. GSLIB: Geostatistical Software Library and User's Guide, 2nd ed. Oxford
University Press, New York.
