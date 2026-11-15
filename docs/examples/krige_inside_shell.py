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
