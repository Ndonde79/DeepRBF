# Indicator grade-shell workflow

    cd examples/kilamongo
    python -m deeprbf.workflow.shell kilamongo_indicator.json

Inputs: collar, survey and interval CSVs (column names mapped in `drillholes.columns`), one JSON settings file,
optional topography (DXF, Surpac .str/.dtm, CSV points; Datamine .dm through `surfaces.register_reader`) and an
optional reference wireframe (e.g. a Leapfrog export) for comparison. Outputs in `output_dir`: `<name>.str/.dtm`
(Surpac), `<name>.dxf`, `<name>_composites.csv`, `<name>_field.npy`, `<name>_report.json`.

Settings follow the Leapfrog indicator RBF dialog: trend dip / dip azimuth / pitch with maximum, intermediate and
minimum ranges; interpolant kernel, sill, nugget and drift; iso value, resolution (`spacing`), small-body filter
(`min_volume`). `range_factor` 1.04 with the spherical kernel follows a Leapfrog spheroidal alpha 9 curve.
Withheld-hole folds (`validation.folds`) score the trend and settings on unseen holes; the wireframe uses all holes.
Requires scikit-image for contouring (`pip install deepfusion-deeprbf[workflow]`).

## Methods (`"method"`)
- `indicator` (default): 0/1 indicator at the cut-off, contoured at `surface.iso`.
- `signed_distance` (BOUNDSIM, McLennan & Deutsch 2006): each composite coded with its anisotropic distance to the
  nearest composite of the other class (ore negative; in-hole contacts at mid-points), boundary at 0.
  `"uncertainty": {"quantiles": [0.1, 0.5, 0.9], "folds": 5}` adds conservative / expected / aggressive shells: the
  shift C is the quantile of the field at true contacts in withheld holes (coverage of contacts), not the paper's
  bootstrap of the mean code.
- `contacts` (after Wang et al. 2018, Hermite RBF): contacts along holes at 0, normals from neighbouring holes' contacts
  (PCA), off-surface points at +/- `contacts.delta` and interval mid-points (`contacts.max_half`). An approximation
  of Hermite gradient constraints; true gradient constraints in the native engine are future work.
Virtual boreholes (classifier-labelled points between holes) are deliberately not implemented: they would be honoured
as if drilled.
