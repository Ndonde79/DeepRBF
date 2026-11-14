# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Signed-distance (volume function) codes for ore/waste boundaries, after BOUNDSIM (McLennan & Deutsch, CCG 2006).

Each composite is coded with the distance to the nearest composite of the OTHER class: negative inside ore, positive in
waste (BOUNDSIM sign convention). The codes are interpolated and the boundary is the zero iso-surface: ore = field <= 0.

Units. Without a frame, distances are Euclidean in world metres. With an anisotropy Frame the distance is measured in
the frame's unit-range space and multiplied by the geometric mean of the three frame ranges, so codes stay in
"metres-equivalent": an isotropic frame of any range gives exactly the world-metre codes, and an anisotropic frame
preserves the volume scale (a unit ball in frame space has the volume of a sphere of radius gmean(ranges)).

Shift C. BOUNDSIM inflates/contracts the boundary by changing the mean of the volume function (lower mean -> larger
body). Here the shift is explicit: shifted = d - C, boundary at shifted == 0, so ore = field <= C and LARGER C = LARGER
ore body. With an exact interpolant whose drift contains a constant ('constant' or 'linear'), fitting d - C gives
exactly the fitted field minus C, so the shift can be applied to codes before fitting or to the field after it.

Calibration of C (calibrate_shift). Holes are withheld in folds; codes are recomputed from the training holes only,
interpolated, and the field is read at the TRUE contacts of the withheld holes (midpoint between consecutive samples
of opposite class in a hole), where the true volume function is zero. C_q is the q-quantile of those field values:
a fraction q of withheld true contacts lies inside or on the shifted boundary {field <= C_q}. So q is the probability
that the shifted boundary reaches or passes the true contact, i.e. that ore just inside a true contact at an unsampled
location lies inside the modelled boundary. p10 -> small, conservative boundary (10% of contacts enclosed); p50 ->
expected (unbiased contact position); p90 -> large, aggressive boundary. Note this is the reverse of the BOUNDSIM
labelling of quantiles of the MEAN (where a low quantile of the mean gives a large body); the meaning of q here is
stated in terms of the boundary, which is what is reported. Withheld-hole statistics are support for the geometry
decision (how far to inflate/contract); the final boundary uses all holes."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

__all__ = ['signed_distance_codes', 'shift_codes', 'contact_points', 'calibrate_shift', 'native_fit_predict']

def _space(xyz, frame):
    X = np.asarray(xyz, float)
    if X.ndim != 2 or X.shape[1] != 3: raise ValueError('xyz must be (n, 3)')
    if frame is None: return X, 1.0
    return np.asarray(frame(X), float), float(np.exp(np.log(np.asarray(frame.r, float)).mean()))

def contact_points(xyz, ore, holes):
    """Midpoints (world XYZ) between consecutive samples of opposite class within each hole. Samples are taken in the
    order given within each hole (pass composites sorted down-hole)."""
    X = np.asarray(xyz, float); o = np.asarray(ore, bool); h = np.asarray(holes)
    if not (len(X) == len(o) == len(h)): raise ValueError('xyz, ore and holes must have the same length')
    out = []
    for u in np.unique(h):
        i = np.flatnonzero(h == u); j = np.flatnonzero(o[i][1:] != o[i][:-1])
        if len(j): out.append(0.5 * (X[i[j]] + X[i[j + 1]]))
    return np.vstack(out) if out else np.empty((0, 3))

def signed_distance_codes(xyz, ore, frame=None, holes=None):
    """BOUNDSIM sample codes. Returns d (n,): distance to the nearest sample of the other class, negative for ore,
    positive for waste, in metres-equivalent units (see module notes). If holes is given, the distance is also measured
    to in-hole contact midpoints and the smaller is used (the contact is placed halfway between the samples, so samples
    next to a contact get half the composite spacing instead of a full spacing). Raises ValueError if all samples are
    one class."""
    o = np.asarray(ore, bool); P, s = _space(xyz, frame)
    if len(o) != len(P): raise ValueError('xyz and ore must have the same length')
    if o.all() or not o.any(): raise ValueError('signed distance codes need both ore and waste samples')
    d = np.empty(len(P))
    d[o] = cKDTree(P[~o]).query(P[o])[0]; d[~o] = cKDTree(P[o]).query(P[~o])[0]
    if holes is not None:
        C = contact_points(xyz, o, holes)
        if len(C): d = np.minimum(d, cKDTree(_space(C, frame)[0]).query(P)[0])
    return np.where(o, -d, d) * s

def shift_codes(d, C):
    """Shifted codes d - C. The boundary stays at 0, so larger C inflates the ore body (ore = d - C <= 0)."""
    return np.asarray(d, float) - float(C)

def calibrate_shift(xyz, ore, holes, frame, fit_predict, targets=(0.1, 0.5, 0.9), folds=5, seed=0, contact_codes=False):
    """Withheld-hole calibration of the shift C (see module notes for the meaning of the quantiles).
    fit_predict(train_xyz, train_codes, query_xyz) -> predicted codes. Codes of each fold are recomputed from its
    training holes (contact_codes=True passes holes to signed_distance_codes). Returns dict with
      C: {q: C_q}, label: {q: 'conservative'|'expected'|'aggressive'}, contact_values: pooled field at withheld contacts,
      coverage: {q: fraction of withheld contacts with field <= C_q} (= q up to ties/interpolation),
      ore_inside / waste_outside: {q: fraction of withheld ore samples with field <= C_q / waste samples with field > C_q},
      folds: per-fold dicts (holes, train/test samples, contacts, median and quartiles of contact values,
             ore_inside and waste_outside at C=0)."""
    X = np.asarray(xyz, float); o = np.asarray(ore, bool); h = np.asarray(holes)
    if not (len(X) == len(o) == len(h)): raise ValueError('xyz, ore and holes must have the same length')
    if o.all() or not o.any(): raise ValueError('calibration needs both ore and waste samples')
    q = np.asarray(targets, float)
    if q.ndim != 1 or (q <= 0).any() or (q >= 1).any(): raise ValueError('targets must be quantiles in (0, 1)')
    u = np.random.default_rng(seed).permutation(np.unique(h)); k = int(folds)
    if not 2 <= k <= len(u): raise ValueError('need 2 <= folds <= number of holes')
    vals, field, rep = [], np.full(len(X), np.nan), []
    for i in range(k):
        te = np.isin(h, u[i::k]); tr = ~te
        if o[tr].all() or not o[tr].any(): raise ValueError(f'fold {i}: training holes are all one class')
        d = signed_distance_codes(X[tr], o[tr], frame, h[tr] if contact_codes else None)
        Cp = contact_points(X[te], o[te], h[te]); Q = np.vstack([X[te], Cp])
        p = np.asarray(fit_predict(X[tr], d, Q), float).ravel(); field[te] = p[:te.sum()]; v = p[te.sum():]; vals.append(v)
        f = field[te]; ot = o[te]
        rep.append(dict(fold=i, holes=int(len(u[i::k])), train_samples=int(tr.sum()), test_samples=int(te.sum()), contacts=int(len(v)),
                        contact_value_p25_p50_p75=[float(x) for x in np.percentile(v, [25, 50, 75])] if len(v) else None,
                        ore_inside_at_0=float((f[ot] <= 0).mean()) if ot.any() else None,
                        waste_outside_at_0=float((f[~ot] > 0).mean()) if (~ot).any() else None))
    v = np.concatenate(vals)
    if len(v) == 0: raise ValueError('no ore/waste contacts in the withheld holes')
    Cq = np.quantile(v, q); key = [round(float(t), 6) for t in q]
    lab = lambda t: 'expected' if abs(t - 0.5) < 1e-9 else ('conservative' if t < 0.5 else 'aggressive')
    return dict(C={t: float(c) for t, c in zip(key, Cq)}, label={t: lab(t) for t in key}, contact_values=v,
                coverage={t: float((v <= c).mean()) for t, c in zip(key, Cq)},
                ore_inside={t: float((field[o] <= c).mean()) for t, c in zip(key, Cq)},
                waste_outside={t: float((field[~o] > c).mean()) for t, c in zip(key, Cq)},
                folds=rep, n_contacts=int(len(v)),
                note='q = share of withheld true contacts enclosed by field <= C_q; supports the boundary geometry decision')

def native_fit_predict(kernel='linear', drift='linear', regularization=0.0, frame=None, epsilon=1.0):
    """fit_predict callable for calibrate_shift using deeprbf_recovered.NativeExactRBFImplicitModel. Points are mapped
    to frame space if a frame is given (otherwise centred on the training mean, world units), duplicates in model space
    (rounded to 1e-7) are merged with their codes averaged. 'linear' kernel needs drift 'linear' or 'constant'."""
    import deeprbf_recovered as native
    if kernel == 'linear' and drift not in ('linear', 'constant'): raise ValueError("kernel 'linear' needs drift 'linear' or 'constant'")
    def fit_predict(train_xyz, train_codes, query_xyz):
        T = np.asarray(train_xyz, float); Qx = np.asarray(query_xyz, float); c = np.asarray(train_codes, float)
        if frame is not None: P, Q = frame(T), frame(Qx)
        else: m0 = T.mean(0); P, Q = T - m0, Qx - m0
        P, inv = np.unique(np.asarray(P).round(7), axis=0, return_inverse=True); inv = inv.ravel()
        v = np.bincount(inv, c) / np.bincount(inv)
        m = native.NativeExactRBFImplicitModel(kernel, float(epsilon), float(regularization)); m.set_drift(drift)
        m.fit(P.tolist(), v.tolist())
        return np.asarray(m.predict(np.asarray(Q).tolist()), float) if len(Q) else np.empty(0)
    return fit_predict
