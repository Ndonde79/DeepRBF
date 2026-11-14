# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Support-explicit reporting of joint grade distributions.

These functions summarize supplied distributions; they do not fit MIK, infer
spatial dependence, or convert point distributions into block distributions.
Grades must already represent the stated selection support. Units are retained;
multiply tonnes times Cu percent by .01, or Au g/t by 1/31.1034768 for ounces.
"""
from dataclasses import dataclass
import numpy as np


def _finite_nonnegative(value, name):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(name + ' must be finite and nonnegative')
    return a


@dataclass(frozen=True)
class RecoverableMoments:
    cutoffs: np.ndarray
    probability: np.ndarray
    truncated_grade: np.ndarray
    support: str

    def totals(self, tonnes):
        """Return expected selected tonnes and tonne-grade products per cutoff.

        Fixed tonnes are assumed. For uncertain correlated density, calculate
        totals per joint realization instead of using these marginal moments.
        """
        t = _finite_nonnegative(tonnes, 'tonnes')
        if t.shape != (self.probability.shape[0],):
            raise ValueError('tonnes must have one value per location')
        return t @ self.probability, np.einsum('n,ncm->cm', t, self.truncated_grade)


def recoverable_moments(grades, weights, cutoffs, selection_coefficients, *, support):
    """Summarize paired grades [location, outcome, metal], at explicit support.

    weights: [location, outcome], normalized independently for each location.
    Selection is inclusive: grades @ coefficients >= cutoff. Coefficients must
    use compatible units. Cu/Au example [1, .537] is caller configuration.
    Outcomes must retain joint metal dependence, but need not be spatially joint
    to calculate expected totals. Marginal outcomes cannot give total quantiles.
    """
    if not isinstance(support, str) or not support.strip():
        raise ValueError('an explicit nonempty support description is required')
    g = _finite_nonnegative(grades, 'grades')
    w = _finite_nonnegative(weights, 'weights')
    c = _finite_nonnegative(cutoffs, 'cutoffs')
    a = _finite_nonnegative(selection_coefficients, 'selection_coefficients')
    if g.ndim != 3 or any(d == 0 for d in g.shape):
        raise ValueError('grades must be nonempty [location, outcome, metal]')
    if w.shape != g.shape[:2] or (w.max(axis=1) <= 0).any():
        raise ValueError('weights must match locations/outcomes with positive row totals')
    if c.ndim != 1 or not len(c) or (np.diff(c) <= 0).any():
        raise ValueError('cutoffs must be a strictly increasing nonempty vector')
    if a.shape != (g.shape[2],) or not a.any():
        raise ValueError('coefficients must match metals with at least one positive value')
    # Scale before normalizing to avoid overflow for large valid weights.
    w = w / w.max(axis=1, keepdims=True)
    w /= w.sum(axis=1, keepdims=True)
    with np.errstate(over='raise', invalid='raise'):
        selection = g @ a
    p = np.empty((len(g),len(c)))
    m = np.empty((len(g),len(c),g.shape[2]))
    # Loop over cutoffs avoids a location*outcome*cutoff temporary mask.
    for j, cutoff in enumerate(c):
        selected_weights = w * (selection >= cutoff)
        p[:,j] = selected_weights.sum(axis=1)
        m[:,j] = np.einsum('nk,nkm->nm', selected_weights, g)
    return RecoverableMoments(c.copy(), p, m, support.strip())


def aggregate_joint_realizations(grades, cell_tonnes, group_ids):
    """Mass-weight spatially JOINT realizations before applying cutoffs.

    grades: [realization, cell, metal]; cell_tonnes: fixed positive cell masses;
    group_ids: integer destination support ID per cell. Returns sorted group IDs,
    tonnes per group and grades [realization, group, metal]. This operation
    requires aligned joint realizations, never independently sorted marginals.
    """
    g = _finite_nonnegative(grades, 'grades')
    t = _finite_nonnegative(cell_tonnes, 'cell_tonnes')
    ids = np.asarray(group_ids)
    if g.ndim != 3 or any(d == 0 for d in g.shape):
        raise ValueError('grades must be nonempty [realization, cell, metal]')
    if t.shape != (g.shape[1],) or (t <= 0).any():
        raise ValueError('one positive fixed mass is required per cell')
    if ids.shape != t.shape or ids.dtype.kind not in 'iu':
        raise ValueError('group_ids must be one integer per cell')
    labels, inverse = np.unique(ids, return_inverse=True)
    masses = np.bincount(inverse, weights=t)
    if not np.isfinite(masses).all():
        raise ValueError('group mass overflow')
    output = np.zeros((g.shape[0],len(labels),g.shape[2]))
    normalized = t / masses[inverse]
    for r in range(g.shape[0]):
        for metal in range(g.shape[2]):
            output[r,:,metal] = np.bincount(inverse, weights=normalized*g[r,:,metal], minlength=len(labels))
    return labels, masses, output
