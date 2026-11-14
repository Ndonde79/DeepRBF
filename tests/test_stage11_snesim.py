# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from deeprbf_snesim import (
    SNESIMConfig,
    SNESIMMineralSimulator,
    UNKNOWN,
    build_template_offsets,
)


def channel_ti(ny=72, nx=72):
    yy, xx = np.mgrid[0:ny, 0:nx]
    ti = np.zeros((ny, nx), dtype=np.int64)
    center1 = ny * 0.32 + 8.0 * np.sin(xx / 10.0)
    center2 = ny * 0.67 + 6.0 * np.sin(xx / 12.0 + 1.0)
    ti[np.abs(yy-center1) < 3.0] = 1
    ti[np.abs(yy-center2) < 4.0] = 1
    return ti


def layered_3d_ti(nz=18, ny=30, nx=34):
    z,y,x = np.mgrid[0:nz,0:ny,0:nx]
    ti = np.zeros((nz,ny,nx), dtype=np.int64)
    surf = nz*0.45 + 2.5*np.sin(x/6.0) + 1.5*np.cos(y/7.0)
    ti[z > surf] = 1
    lens = ((x-nx*0.55)/9.0)**2 + ((y-ny*0.5)/7.0)**2 + ((z-nz*0.45)/3.0)**2
    ti[lens < 1.0] = 2
    return ti


def main():
    # 2D hard conditioning + deterministic run
    ti = channel_ti()
    cfg = SNESIMConfig(
        max_conditioning_nodes=24,
        min_replicates=8,
        n_multigrids=3,
        search_radii=(7,7,1),
        servo_lambda=0.05,
        seed=12345,
        n_realizations=2,
    )
    hard = np.full((40,44), UNKNOWN, dtype=np.int64)
    hard[5,5] = 1
    hard[10,10] = 0
    hard[22,18] = 1
    eng = SNESIMMineralSimulator(ti, cfg)
    sims, diag = eng.simulate(hard.shape, hard_data=hard, return_diagnostics=True)
    assert sims.shape == (2,40,44)
    assert np.all(sims[:,5,5] == 1)
    assert np.all(sims[:,10,10] == 0)
    assert np.all(sims[:,22,18] == 1)
    assert set(np.unique(sims)).issubset({0,1})

    sims2, _ = eng.simulate(hard.shape, hard_data=hard, return_diagnostics=True, seed=12345)
    assert np.array_equal(sims, sims2)

    # Soft conditioning must move a local area toward facies 1.
    soft = np.zeros((40,44,2), dtype=float)
    soft[...,0] = 0.90
    soft[...,1] = 0.10
    soft[:,22:,0] = 0.10
    soft[:,22:,1] = 0.90
    softcfg = SNESIMConfig(
        max_conditioning_nodes=20,
        min_replicates=6,
        n_multigrids=2,
        search_radii=(6,6,1),
        soft_tau=1.0,
        seed=99,
        n_realizations=1,
    )
    softeng = SNESIMMineralSimulator(ti, softcfg)
    ss, _ = softeng.simulate((40,44), soft_probabilities=soft)
    left = np.mean(ss[0,:,:22] == 1)
    right = np.mean(ss[0,:,22:] == 1)
    assert right > left

    # 3D mineral-deposit style facies.
    ti3 = layered_3d_ti()
    cfg3 = SNESIMConfig(
        max_conditioning_nodes=20,
        min_replicates=5,
        n_multigrids=2,
        search_radii=(5,5,3),
        affinity=(1.0,0.8,0.6),
        seed=777,
        n_realizations=1,
    )
    hard3 = np.full((10,18,20), UNKNOWN, dtype=np.int64)
    hard3[4,9,10] = 2
    e3 = SNESIMMineralSimulator(ti3, cfg3)
    s3, d3 = e3.simulate(hard3.shape, hard_data=hard3)
    assert s3.shape == (1,10,18,20)
    assert s3[0,4,9,10] == 2
    assert set(np.unique(s3)).issubset({0,1,2})

    print("="*72)
    print("DEEPRBF STAGE 11 SNESIM REFERENCE TESTS PASSED")
    print("="*72)
    print("2D shapes:", sims.shape)
    print("soft left facies1 :", left)
    print("soft right facies1:", right)
    print("3D shape:", s3.shape)
    print("diagnostics:")
    print(json.dumps({"2d":diag[0], "3d":d3[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
