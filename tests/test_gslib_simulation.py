# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""3.3.4 GSLIB-convention kriging and simulation: rotation convention, kriging against a direct solve,
normal-score round trip, SGS/SIS conditioning, repeatability, histogram and variogram reproduction."""
import os
import sys

sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
import numpy as np
import pytest
import deeprbf_recovered as n


def _sph(h):
    h = np.minimum(h, 1.0)
    return 1.0 - 1.5 * h + 0.5 * h ** 3


@pytest.mark.parametrize("az,dip,expected", [(0, 0, [0, 1, 0]), (90, 0, [1, 0, 0]), (45, 0, [0.70710678, 0.70710678, 0]),
                                             (0, 30, [0, 0.8660254, -0.5]), (90, 60, [0.5, 0, -0.8660254])])
def test_rotation_major_axis(az, dip, expected):
    R = np.asarray(n.gslib_rotation(az, dip, 0, 100, 10, 5)).reshape(3, 3)
    d = np.asarray(expected) * 50.0
    assert np.linalg.norm(R @ d) == pytest.approx(0.5, abs=1e-7)         # 50 m along the major axis = half its range
    assert np.linalg.norm(R @ np.cross(d, [0, 0, 1] if dip == 0 else [1, 0, 0])) > 0.5


def _ok_direct(X, v, q, V, k, ordinary=True, mean=0.0):
    out = []
    for t in q:
        d = np.linalg.norm(X - t, axis=1)
        nb = np.argsort(d)[:k]
        P = X[nb]
        C = np.array([[V.covariance(a.tolist(), b.tolist()) for b in P] for a in P])
        c = np.array([V.covariance(a.tolist(), t.tolist()) for a in P])
        if ordinary:
            A = np.ones((k + 1, k + 1)); A[:k, :k] = C; A[k, k] = 0
            w = np.linalg.solve(A, np.r_[c, 1.0])
            out.append((w[:k] @ v[nb], V.sill() - w[:k] @ c - w[k]))
        else:
            w = np.linalg.solve(C, c)
            out.append((mean + w @ (v[nb] - mean), V.sill() - w @ c))
    return np.array(out)


@pytest.mark.parametrize("ktype", ["ordinary", "simple"])
def test_kriging_matches_direct_solve(ktype):
    rg = np.random.default_rng(0)
    X = rg.uniform(0, 100, (300, 3)); v = rg.normal(size=300)
    q = rg.uniform(20, 80, (60, 3))
    V = n.GSLIBVariogram(0.1, [n.GSLIBStructure("spherical", 0.6, 60, 30, 10, 30, 20, 10),
                               n.GSLIBStructure("exponential", 0.3, 120, 120, 40, 30, 20, 10)])
    r = n.kriging_gslib(X.tolist(), v.tolist(), q.tolist(), V, n.SearchEllipsoid(500, 500, 500), max_data=20, min_data=1,
                        kriging_type=ktype, mean=0.2)
    ref = _ok_direct(X, v, q, V, 20, ktype == "ordinary", 0.2)
    assert np.allclose(r["estimate"], ref[:, 0], atol=1e-10)
    assert np.allclose(r["variance"], ref[:, 1], atol=1e-10)


def test_normal_score_monotone_and_standard():
    z = np.exp(np.random.default_rng(1).normal(size=500))
    y = np.asarray(n.normal_score(z.tolist()))
    assert np.all(np.diff(y[np.argsort(z)]) >= 0)
    assert abs(y.mean()) < 0.01 and abs(y.std() - 1.0) < 0.03


def _grid(nx, s):
    gx, gy = np.meshgrid(s / 2 + s * np.arange(nx), s / 2 + s * np.arange(nx))
    return np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]


def test_sgs_conditioning_repeatability_histogram():
    rg = np.random.default_rng(2)
    q = _grid(40, 5.0)
    idx = rg.choice(len(q), 50, replace=False)
    z = np.exp(0.8 * rg.normal(size=50))
    V = n.GSLIBVariogram(0.0, [n.GSLIBStructure("spherical", 1.0, 60, 60, 60)])
    S = n.SearchEllipsoid(120, 120, 120)
    a = n.sgs_gslib(q[idx].tolist(), z.tolist(), q.tolist(), V, S, n_realizations=6, seed=5)
    b = n.sgs_gslib(q[idx].tolist(), z.tolist(), q.tolist(), V, S, n_realizations=6, seed=5)
    assert np.array_equal(a, b)
    assert np.abs(a[idx] - z[:, None]).max() == 0.0
    assert a.min() >= z.min() - 1e-12 and a.max() <= z.max() + 1e-12      # tails default to the data limits
    assert abs(np.median(a) / np.median(z) - 1.0) < 0.25


def test_sgs_variogram_reproduction():
    rg = np.random.default_rng(3)
    q = _grid(50, 4.0)
    X = rg.uniform(0, 200, (40, 3)); X[:, 2] = 0
    V = n.GSLIBVariogram(0.0, [n.GSLIBStructure("spherical", 1.0, 50, 50, 50)])
    R = n.sgs_gslib(X.tolist(), rg.normal(size=40).tolist(), q.tolist(), V, n.SearchEllipsoid(100, 100, 100),
                    n_realizations=10, max_data=0, max_nodes=24, normal_score=False, seed=9)
    from scipy.spatial import cKDTree
    t = cKDTree(q[:, :2]); g = []
    for h in (8.0, 24.0, 60.0):
        pr = t.query_pairs(h + 1.0, output_type="ndarray")
        d = np.linalg.norm(q[pr[:, 0], :2] - q[pr[:, 1], :2], axis=1); pr = pr[d > h - 1.0]
        g.append(np.mean([0.5 * np.mean((R[pr[:, 0], r] - R[pr[:, 1], r]) ** 2) for r in range(R.shape[1])]))
    model = 1.0 - _sph(np.array([8.0, 24.0, 60.0]) / 50.0)
    assert np.all(np.abs(np.array(g) - model) < 0.12)


def test_sis_conditioning_and_proportion():
    rg = np.random.default_rng(4)
    q = _grid(40, 5.0)
    idx = rg.choice(len(q), 60, replace=False)
    c = (rg.random(60) < 0.3).astype(int)
    V = n.GSLIBVariogram(0.0, [n.GSLIBStructure("spherical", 1.0, 50, 50, 50)])
    R = n.sis_gslib(q[idx].tolist(), c.tolist(), q.tolist(), [0, 1], [V, V], n.SearchEllipsoid(100, 100, 100), n_realizations=8, seed=3)
    assert (R[idx] == c[:, None]).all()
    assert abs((R == 1).mean() - c.mean()) < 0.06


def test_invalid_inputs():
    V = n.GSLIBVariogram(0.0, [n.GSLIBStructure("cubicish", 1.0, 50, 50, 50)])
    with pytest.raises(ValueError):
        n.kriging_gslib([[0, 0, 0], [1, 1, 1]], [1.0, 2.0], [[0.5, 0.5, 0.5]], V, n.SearchEllipsoid())
    V = n.GSLIBVariogram(0.0, [n.GSLIBStructure("spherical", 1.0, 50, 50, 50)])
    with pytest.raises(ValueError):
        n.kriging_gslib([[0, 0, 0]], [1.0], [[0, 0, 0]], V, n.SearchEllipsoid(), kriging_type="universal")
