# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Paired point-support grade validation with a common native neighbourhood."""
import numpy as np
from deeprbf_recovered import select_neighborhood


def compare_idw2(estimator, centers, domains):
    """Return native grade estimates and IDW² using the same samples and search metric."""
    centers = np.asarray(centers, float)
    domains = np.asarray(domains)
    result = estimator.predict(centers, target_domains=domains)
    x, y, holes, codes = estimator.xyz, estimator.grades, estimator.holes, estimator.domains
    search = estimator.search
    reference = np.full(len(centers), np.nan)
    for i, (q, domain) in enumerate(zip(centers, domains)):
        neighbors = select_neighborhood(x, holes, codes, q, int(domain), search)
        if not neighbors.supported:
            if result['supported'][i]:
                raise RuntimeError('Native estimator used an unsupported neighbourhood')
            continue
        indices = np.asarray(neighbors.indices, int)
        if result['supported'][i] and len(indices)!=result['n_samples'][i]:
            raise RuntimeError('Estimator and IDW sample counts differ')
        distance = np.linalg.norm(((x[indices]-q)@np.asarray(search.axes).T)/search.radii, axis=1)
        coincident = distance<=1e-12
        if coincident.any():
            reference[i] = y[indices[coincident]].mean()
        else:
            reference[i] = np.average(y[indices], weights=(distance.min()/distance)**2)
    return result, reference


def paired_grade_scores(observed, estimate, reference, weights, baseline, gates):
    """Score exact observations only; callers omit censored observations."""
    y, a, b, w = (np.asarray(v, float) for v in (observed, estimate, reference, weights))
    if (y.ndim!=1 or any(v.shape!=y.shape for v in (a,b,w)) or not np.isfinite(y).all()
            or (y<0).any() or not np.isfinite(w).all() or (w<=0).any() or not np.isfinite(baseline)):
        raise ValueError('Expected exact grades and positive aligned weights')
    w = w/w.sum() if len(w) else w
    a_ok, b_ok = np.isfinite(a), np.isfinite(b)

    def score(p, mask):
        coverage = float(w@mask)
        if not mask.any():
            return dict(scored=0, coverage=coverage, bias=None, rmse=None, rmse_ratio=None, pass_all=False)
        ww = w[mask]/coverage
        t, v = y[mask], p[mask]
        mean = float(ww@t)
        bias = float((ww@v-mean)/mean) if mean else None
        rmse = float(np.sqrt(ww@(v-t)**2))
        naive = float(np.sqrt(ww@(baseline-t)**2))
        ratio = rmse/naive if naive else None
        checks = dict(coverage=coverage>=gates['coverage'], scored=int(mask.sum())>=gates['scored'],
                      bias=bias is not None and abs(bias)<=gates['abs_relative_bias'],
                      rmse=ratio is not None and ratio<=gates['rmse_ratio'])
        return dict(scored=int(mask.sum()), coverage=coverage, observed_mean=mean, bias=bias,
                    rmse=rmse, rmse_ratio=ratio, baseline_rmse=naive, checks=checks, pass_all=all(checks.values()))

    return dict(deeprbf=score(a,a_ok), idw2=score(b,b_ok),
                paired_deeprbf=score(a,a_ok&b_ok), paired_idw2=score(b,a_ok&b_ok))
