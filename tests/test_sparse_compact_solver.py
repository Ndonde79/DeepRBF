# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""3.3.3 sparse direct path for compactly supported kernels (spherical): sparse Cholesky + Schur complement on the
linear drift, and neighbour-limited prediction, checked against the dense augmented solver and full-sum predict."""
import os
import sys
import time

sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
import numpy as np
import pytest
import deeprbf_recovered as n

M = n.NativeExactRBFImplicitModel


@pytest.fixture(autouse=True)
def _reset_solver_state():
    yield
    M.set_sparse_solver(False)
    M.force_iterative_solver(False)
    M.set_dense_direct_limit(8000)
    M.set_predict_backend("auto")


def _random_case(npts=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 6.0, (npts, 3))
    y = np.sin(1.3 * x[:, 0]) + np.cos(0.7 * x[:, 1]) * x[:, 2] + 0.2 * x[:, 0]
    q = rng.uniform(-0.5, 6.5, (3000, 3))
    return x, y, q


def _fit(kernel, eps, reg, x, y, sparse):
    M.set_sparse_solver(sparse)
    m = M(kernel, eps, reg)
    m.fit(x.tolist(), y.tolist())
    M.set_sparse_solver(False)
    return m


def _predict(m, q, backend="auto"):
    M.set_predict_backend(backend)
    try:
        return np.asarray(m.predict(np.asarray(q).tolist()))
    finally:
        M.set_predict_backend("auto")


@pytest.mark.parametrize("reg", [0.0, 0.03])
def test_sparse_matches_dense(reg):
    x, y, q = _random_case()
    dense = _fit("spherical", 1.0, reg, x, y, sparse=False)
    sparse = _fit("spherical", 1.0, reg, x, y, sparse=True)
    assert dense.solver_backend().startswith("dense_augmented")
    assert sparse.solver_backend() == "sparse_cholesky_schur"
    assert set(sparse.sparse_solve_info()) >= {"nnz_lower", "factor_seconds", "relative_residual"}
    assert dense.sparse_solve_info() == {}
    ref = _predict(dense, q, "direct")                       # dense weights, full O(n) sum
    got = _predict(sparse, q)                                # sparse weights, neighbour-limited sum
    assert sparse.last_predict_backend() == "compact_grid_openmp"
    scale = max(1.0, np.abs(ref).max())
    assert np.abs(got - ref).max() < 1e-8 * scale
    # Same weights, two predict paths: neighbour-limited sum equals the full sum to rounding.
    full = _predict(sparse, q, "direct")
    assert np.abs(got - full).max() < 1e-10 * scale


def test_exact_interpolation_without_regularization():
    x, y, _ = _random_case(seed=1)
    m = _fit("spherical", 1.0, 0.0, x, y, sparse=True)
    assert np.abs(_predict(m, x) - y).max() < 1e-9 * max(1.0, np.abs(y).max())


def test_drift_is_reproduced_exactly():
    """A linear field is reproduced by the drift alone (weights ~ 0), checking the drift back-transformation."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 4.0, (800, 3)) + np.array([5.0e5, 9.6e6, 1.2e3])   # UTM-like offsets
    y = 3.0 - 0.5 * (x[:, 0] - 5e5) + 0.25 * (x[:, 1] - 9.6e6) + 2.0 * (x[:, 2] - 1.2e3)
    m = _fit("spherical", 1.0, 0.0, x, y, sparse=True)
    q = rng.uniform(0.0, 4.0, (500, 3)) + np.array([5.0e5, 9.6e6, 1.2e3])
    yq = 3.0 - 0.5 * (q[:, 0] - 5e5) + 0.25 * (q[:, 1] - 9.6e6) + 2.0 * (q[:, 2] - 1.2e3)
    assert np.abs(_predict(m, q) - yq).max() < 1e-6


def test_automatic_selection_above_dense_limit():
    x, y, q = _random_case(npts=1500, seed=2)
    M.set_dense_direct_limit(1000)
    m = M("spherical", 1.0, 0.03)
    m.fit(x.tolist(), y.tolist())
    assert m.solver_backend() == "sparse_cholesky_schur"
    g = M("gaussian", 1.0, 0.03)                             # global kernel: never the sparse path
    M.set_sparse_solver(True)
    g.fit(x[:300].tolist(), y[:300].tolist())
    assert not g.solver_backend().startswith("sparse")
    assert M.sparse_solver_forced()


def _drillholes(holes=400, per_hole=50, seed=11):
    """400 vertical holes on a jittered 20 x 20 collar grid (~10 m apart), 50 samples 1 m apart down each hole,
    indicator values from a dipping lens; coordinates scaled by 1/20 so the range epsilon = 1 (20 m) reaches about
    a dozen neighbouring holes."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(20) * 10.0, np.arange(20) * 10.0)
    collars = np.c_[gx.ravel(), gy.ravel()][:holes] + rng.uniform(-3.0, 3.0, (holes, 2))
    z = -np.arange(per_hole, dtype=float)
    pts = np.c_[np.repeat(collars, per_hole, axis=0), np.tile(z, holes)]
    s = pts[:, 2] + 0.15 * pts[:, 0] - 0.05 * pts[:, 1] + 30.0 + 4.0 * np.sin(pts[:, 0] / 25.0)
    ind = np.where(np.abs(s) < 6.0, 1.0, 0.0)
    return pts / 20.0, ind


def test_drillhole_20k_fit_is_fast_and_accurate():
    x, y = _drillholes()
    assert len(x) == 20000
    m = M("spherical", 1.0, 0.026)
    t0 = time.perf_counter()
    m.fit(x.tolist(), y.tolist())                            # 20,000 > dense limit: sparse path automatically
    t_fit = time.perf_counter() - t0
    assert m.solver_backend() == "sparse_cholesky_schur"
    assert t_fit < 45.0, f"sparse fit took {t_fit:.1f} s"
    info = m.sparse_solve_info()
    assert info["relative_residual"] < 1e-10
    t0 = time.perf_counter()
    fit = _predict(m, x)
    t_pred = time.perf_counter() - t0
    assert t_pred < 10.0
    print(f"\n20k drillhole case: fit {t_fit:.2f} s {info}, predict {len(x)} points {t_pred:.2f} s")
    # Residual check of the saddle system. The model output at the data is f = K w + F a and the system rows are
    # (K + reg I) w + F a = v, so w = (v - f) / reg. Check the constraint rows F^T w = 0 and, on 300 rows, that
    # f - K w (K w by brute force in numpy) is the linear drift F a.
    reg = 0.026
    assert np.isfinite(fit).all()
    w = (y - fit) / reg
    F = np.c_[np.ones(len(x)), x]
    assert np.abs(F.T @ w).max() < 1e-7 * np.abs(F).max() * np.abs(w).sum()
    idx = np.random.default_rng(1).choice(len(x), 300, replace=False)
    d = np.sqrt(((x[idx, None, :] - x[None, :, :]) ** 2).sum(-1))
    k = np.where(d < 1.0, 1.0 - 1.5 * d + 0.5 * d ** 3, 0.0)
    drift = fit[idx] - k @ w
    a, *_ = np.linalg.lstsq(F[idx], drift, rcond=None)
    assert np.abs(F[idx] @ a - drift).max() < 1e-8
    # The indicator is honoured at almost all samples.
    assert np.mean((fit > 0.5) == (y > 0.5)) > 0.97
    assert np.abs(fit - y).max() < 1.0
    # Spot check against the full-sum predict on a subset (same weights).
    full = _predict(m, x[idx], "direct")
    assert np.abs(full - fit[idx]).max() < 1e-10


def test_drillhole_subset_residual_against_dense():
    """Residual check on a dense-solvable subset of the drillhole layout: sparse and dense weights agree."""
    x, y = _drillholes(holes=60, per_hole=50)
    dense = _fit("spherical", 1.0, 0.026, x, y, sparse=False)
    sparse = _fit("spherical", 1.0, 0.026, x, y, sparse=True)
    q = x + 0.01
    ref = _predict(dense, q, "direct")
    assert np.abs(_predict(sparse, q) - ref).max() < 1e-8
