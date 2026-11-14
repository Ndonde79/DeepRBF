# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Merged 3.3.3: the sparse compact solver honours every drift option and matches the dense solver."""
import os, sys, numpy as np, pytest
sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
import deeprbf_recovered as n
M = n.NativeExactRBFImplicitModel

@pytest.mark.parametrize('drift', ['none', 'constant', 'linear'])
@pytest.mark.parametrize('reg', [0.0, 0.03])
def test_sparse_matches_dense_for_each_drift(drift, reg):
    r = np.random.default_rng(5); x = r.uniform(0, 10, (1500, 3)); y = (np.sin(x[:, 0]) + x[:, 1] / 5 > 1).astype(float)
    q = r.uniform(-2, 12, (400, 3)).tolist()
    d = M('spherical', 2.0, reg); d.set_drift(drift); d.fit(x.tolist(), y.tolist()); pd_ = np.asarray(d.predict(q))
    M.set_sparse_solver(True)
    try:
        s = M('spherical', 2.0, reg); s.set_drift(drift); s.fit(x.tolist(), y.tolist()); ps = np.asarray(s.predict(q))
        assert s.solver_backend() == 'sparse_cholesky_schur' and s.drift() == drift
    finally:
        M.set_sparse_solver(False)
    assert np.max(np.abs(pd_ - ps)) < 1e-8

def test_sparse_none_drift_is_zero_far_away():
    r = np.random.default_rng(6); x = r.uniform(0, 10, (1200, 3)); y = r.uniform(0, 1, 1200)
    M.set_sparse_solver(True)
    try:
        s = M('spherical', 2.0, 0.02); s.set_drift('none'); s.fit(x.tolist(), y.tolist())
        assert np.allclose(s.predict([[50.0, 50.0, 50.0], [-40.0, 0.0, 5.0]]), 0.0, atol=1e-14)
    finally:
        M.set_sparse_solver(False)
