# Kilamongo South 0.3 g/t Au grade shell

Public example dataset for the DeepRBF grade-shell workflow (`deeprbf.workflow.shell`).

| File | Content |
|---|---|
| `collar.csv`, `survey.csv`, `assay.csv` | 2020 Kilamongo South (Buhemba, Tanzania) drillholes: collars, downhole surveys, Au assays (g/t) |
| `kilamongo_indicator.json` | Indicator RBF at 0.3 g/t (spherical kernel, trend dip 18° towards 068°) |
| `kilamongo_signed_distance.json` | Signed-distance (BOUNDSIM) shell at the same cut-off |
| `kilamongo_contacts.json` | Contacts-with-normals shell at the same cut-off |
| `Au_Ind_kila_0.3.str`, `Au_Ind_kila_0.3.dtm` | Reference 0.3 g/t indicator shell (Surpac), built independently with Leapfrog, used for comparison |

Run from this folder (Python 3.11 with DeepRBF and scikit-image installed):

```
python -m deeprbf.workflow.shell kilamongo_indicator.json
python -m deeprbf.workflow.shell kilamongo_signed_distance.json
python -m deeprbf.workflow.shell kilamongo_contacts.json
```

Each run takes one to three minutes. Outputs go to `out/`: `<name>.str/.dtm` (Surpac), `<name>.dxf`,
`<name>_composites.csv`, `<name>_field.npy` and `<name>_report.json` (data honouring, volume, comparison with the
reference shell).

Coordinates are UTM (metres). The data are released for research and testing with this software; they are not a
public report of Mineral Resources or Exploration Results.
