# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
from __future__ import annotations

import numpy as np
from deeprbf_recovered import NativeExactRBFImplicitModel


def anisotropy_matrix(azimuth_deg: float = 0.0, dip_deg: float = 0.0, ranges=(1.0, 1.0, 1.0)) -> np.ndarray:
    ranges=np.asarray(ranges,float)
    if ranges.shape!=(3,) or not np.isfinite(ranges).all() or (ranges<=0).any() or not np.isfinite([azimuth_deg,dip_deg]).all():
        raise ValueError('Finite angles and three positive ranges required')
    # Simple transparent geological anisotropy matrix.
    az = np.deg2rad(azimuth_deg)
    dp = np.deg2rad(dip_deg)

    rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az),  np.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ])

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(dp), -np.sin(dp)],
        [0.0, np.sin(dp),  np.cos(dp)],
    ])

    scale = np.diag([
        1.0 / max(float(ranges[0]), 1e-12),
        1.0 / max(float(ranges[1]), 1e-12),
        1.0 / max(float(ranges[2]), 1e-12),
    ])
    return scale @ rx @ rz


def estimate_grade(sample_xyz, sample_grade, block_xyz, ranges=(100.0, 50.0, 25.0), azimuth_deg=0, dip_deg=0):
    a = anisotropy_matrix(azimuth_deg, dip_deg, ranges)
    origin = np.mean(np.asarray(sample_xyz, dtype=float), axis=0)
    model=NativeExactRBFImplicitModel('gaussian',1.,1e-8)
    model.fit((np.asarray(sample_xyz,float)-origin)@a.T,sample_grade)
    return np.asarray(model.predict((np.asarray(block_xyz,float)-origin)@a.T))
