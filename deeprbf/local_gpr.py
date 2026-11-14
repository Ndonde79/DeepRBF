# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Local anisotropic Gaussian-process estimates with explicit support.

Fixed-mean GPR and unknown-constant-mean (ordinary kriging) share the same
covariance and neighbourhood. Variances condition on fitted parameters.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve


class LocalGPR:
    def __init__(self, xyz, values, holes, *, axes, ranges, radii,
                 sill, nugget, mean=None, max_neighbors=14, max_per_hole=4,
                 min_neighbors=4, min_holes=2, candidates=144, jitter=1e-8):
        self.xyz = np.asarray(xyz, float)
        self.y = np.asarray(values, float)
        self.holes = np.asarray(holes)
        self.axes = np.asarray(axes, float)
        self.ranges = np.asarray(ranges, float)
        self.radii = np.asarray(radii, float)
        if self.xyz.shape != (len(self.y), 3) or len(self.holes) != len(self.y):
            raise ValueError('Inconsistent sample dimensions')
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.y).all():
            raise ValueError('Nonfinite samples')
        if sill <= 0 or not 0 <= nugget < 1 or jitter < 0:
            raise ValueError('Invalid covariance variance or nugget')
        if np.any(self.ranges <= 0) or np.any(self.radii <= 0):
            raise ValueError('Ranges and radii must be positive')
        if not np.allclose(self.axes @ self.axes.T, np.eye(3), atol=1e-8):
            raise ValueError('Axes must be orthonormal')
        self.sill, self.nugget, self.jitter = sill, nugget, jitter
        self.mean = float(np.mean(self.y) if mean is None else mean)
        self.maximum, self.per_hole = max_neighbors, max_per_hole
        self.minimum, self.min_holes, self.candidates = min_neighbors, min_holes, candidates
        self.tree = cKDTree(self.xyz @ self.axes.T / self.radii)

    def kernel(self, a, b):
        return self.sill * (1-self.nugget) * np.exp(
            -3 * cdist(np.asarray(a) @ self.axes.T / self.ranges,
                       np.asarray(b) @ self.axes.T / self.ranges))

    def predict(self, targets, block_size=None, discretization=2):
        targets = np.asarray(targets, float)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError('Targets must be finite Nx3 coordinates')
        offsets = np.zeros((1, 3))
        if block_size is not None:
            size = np.asarray(block_size, float)
            if size.shape != (3,) or np.any(size <= 0) or discretization < 1:
                raise ValueError('Invalid block support')
            a = (np.arange(discretization)+.5)/discretization-.5
            offsets = np.array(np.meshgrid(a, a, a)).reshape(3, -1).T * size
        out = {k: np.full(len(targets), np.nan) for k in
               ['gpr', 'ok', 'gpr_latent_variance', 'ok_latent_variance', 'observation_variance']}
        out['n_samples'] = np.zeros(len(targets), int)
        out['n_holes'] = np.zeros(len(targets), int)
        d, indices = self.tree.query(targets @ self.axes.T / self.radii,
                                    k=min(self.candidates, len(self.y)))
        d, indices = np.asarray(d).reshape(len(targets), -1), np.asarray(indices).reshape(len(targets), -1)
        for row, target in enumerate(targets):
            chosen, counts = [], {}
            for dist, idx in zip(d[row], indices[row]):
                if dist > 1: break
                hole = self.holes[idx]
                if counts.get(hole, 0) >= self.per_hole: continue
                chosen.append(idx); counts[hole] = counts.get(hole, 0)+1
                if len(chosen) == self.maximum: break
            out['n_samples'][row], out['n_holes'][row] = len(chosen), len(counts)
            if len(chosen) < self.minimum or len(counts) < self.min_holes: continue
            src, y = self.xyz[chosen], self.y[chosen]
            quadrature = target+offsets
            K = self.kernel(src, src)
            K.flat[::len(chosen)+1] += self.sill*(self.nugget+self.jitter)
            factor = cho_factor(K, lower=True)
            k = self.kernel(src, quadrature).mean(axis=1)
            prior = self.kernel(quadrature, quadrature).mean()
            weights = cho_solve(factor, k)
            one = cho_solve(factor, np.ones(len(chosen)))
            denom = one.sum()
            correction = (1-weights.sum())/denom
            out['gpr'][row] = self.mean + weights @ (y-self.mean)
            out['ok'][row] = (weights+correction*one) @ y
            v = max(0., prior-k@weights)
            out['gpr_latent_variance'][row] = v
            out['ok_latent_variance'][row] = v+(1-weights.sum())**2/denom
            # Observation noise is only defined here for point observations.
            if block_size is None:
                out['observation_variance'][row] = v+self.sill*self.nugget
        return out
