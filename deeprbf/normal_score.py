# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Weighted empirical normal scores and an explicit Gaussian-space SGS workflow.

Independent Python implementation; no GSLIB source is embedded.
Tied values share their group's weighted midpoint probability. Back-transforms
interpolate in score space and clamp tails to observed bounds.
"""
import numpy as np
from scipy.special import ndtri

class NormalScoreTransform:
    def __init__(self, values, weights=None):
        data = np.asarray(values, dtype=float)
        if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
            raise ValueError("values must be a nonempty finite 1-D array")
        weight = np.ones_like(data) if weights is None else np.asarray(weights, dtype=float)
        if weight.shape != data.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("weights must match values and be finite and positive")
        # Scaling prevents overflow when all individual weights are large.
        weight = weight / weight.max()
        unique, inverse = np.unique(data, return_inverse=True)
        mass = np.bincount(inverse, weights=weight, minlength=unique.size)
        cumulative = np.cumsum(mass / mass.sum())
        midpoint = cumulative - .5 * (mass / mass.sum())
        score = ndtri(np.clip(midpoint, np.finfo(float).eps, 1-np.finfo(float).eps))
        if np.any(np.diff(score) <= 0):
            raise ValueError("weight dynamic range is too large for distinct normal scores")
        self.values = unique
        self.scores = score
        self.values.setflags(write=False)
        self.scores.setflags(write=False)

    def transform(self, values):
        """Map original values to scores; values beyond the table are clamped."""
        data = np.asarray(values, dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("values must be finite")
        return np.interp(data, self.values, self.scores)

    def inverse_transform(self, scores):
        """Back-transform, clamping tails to the observed minimum and maximum."""
        data = np.asarray(scores, dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("scores must be finite")
        return np.interp(data, self.scores, self.values)


def run_sgs_normal_score(sample_xyz, sample_values, target_xyz, *, weights=None,
                         n_realizations=100, max_neighbors=24, min_neighbors=5,
                         search_radius=200., variogram_range=100., sill=1.,
                         nugget=.05, seed=42):
    """SGS on empirical normal scores, then back-transform to original units.

    The covariance sill/nugget are in Gaussian-score units. This explicit path
    does not change the legacy run_sgs standardization default. Weights affect
    the empirical distribution, not measurement-error weights in kriging.
    Returns an array with shape (number of targets, number of realizations).
    Tails are bounded by observed sample values; no tail model is inferred.
    """
    from deeprbf_recovered import run_sgs_with_range
    table = NormalScoreTransform(sample_values, weights)
    if table.values.size < 2:
        raise ValueError("normal-score simulation needs at least two distinct values")
    scores = table.transform(sample_values)
    simulated = run_sgs_with_range(sample_xyz, scores, target_xyz,
        n_realizations=n_realizations, max_neighbors=max_neighbors,
        min_neighbors=min_neighbors, search_radius=search_radius,
        variogram_range=variogram_range, sill=sill, nugget=nugget, seed=seed,
        standardize=False)
    return table.inverse_transform(simulated)
