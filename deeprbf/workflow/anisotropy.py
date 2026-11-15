# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Anisotropy in the convention of common implicit-modelling software: a plane given by dip and dip azimuth, a pitch measured in the plane from the
strike direction (dip azimuth - 90 degrees, right-hand rule) toward down-dip, and ranges along maximum, intermediate
and minimum axes (minimum = plane normal). Pitch 90 puts the maximum axis down-dip."""
import numpy as np

def axes(dip, dip_azimuth, pitch):
    d, a, p = np.radians([dip, dip_azimuth, pitch])
    strike = np.array([np.sin(a - np.pi / 2), np.cos(a - np.pi / 2), 0.0])
    down = np.array([np.sin(a) * np.cos(d), np.cos(a) * np.cos(d), -np.sin(d)])
    normal = np.cross(strike, down); normal /= np.linalg.norm(normal)
    mx = np.cos(p) * strike + np.sin(p) * down; it = np.cross(normal, mx)
    return np.vstack([mx, it / np.linalg.norm(it), normal])                       # rows: maximum, intermediate, minimum

class Frame:
    """Maps world XYZ to anisotropic coordinates in which the range is 1 along every axis."""
    def __init__(self, dip, dip_azimuth, pitch, ranges, origin):
        self.A = axes(dip, dip_azimuth, pitch); self.r = np.asarray(ranges, float); self.o = np.asarray(origin, float)
        if self.r.shape != (3,) or (self.r <= 0).any(): raise ValueError('ranges must be three positive lengths (max, int, min)')
    def __call__(self, xyz):
        return ((np.asarray(xyz, float) - self.o) @ self.A.T) / self.r
