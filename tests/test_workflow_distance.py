# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""BOUNDSIM signed-distance codes on the paper's synthetic example: a 20 m radius sphere in a 51 m cube sampled by 100
vertical holes at 1 m. Codes: signs and distances; native linear-kernel + linear-drift interpolation of the codes
recovers the sphere volume; withheld-hole calibration of the shift gives ordered C and ordered boundary volumes."""
import os, sys, numpy as np, pytest
sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
from deeprbf.workflow import distance as ds
from deeprbf.workflow.anisotropy import Frame

R, CEN = 20.0, np.array([25.0, 25.0, 25.0])

def sphere_holes(dz=1.0):
    xs = 2.5 + 5.0 * np.arange(10); z = np.arange(0.5, 51.0, dz)
    X = np.array([(x, y, zz) for x in xs for y in xs for zz in z]); holes = np.repeat(np.arange(100), len(z))
    return X, np.linalg.norm(X - CEN, axis=1) <= R, holes

def brute(P, o):
    D = np.linalg.norm(P[:, None] - P[None], axis=2); return np.where(o, -np.where(o[None], np.inf, D).min(1), np.where(~o[None], np.inf, D).min(1))

@pytest.fixture(scope='module')
def data(): return sphere_holes(1.0)

@pytest.fixture(scope='module')
def fit2m():
    """Every second 1 m composite (2 m along holes) keeps the dense exact solve fast on 2 cores (~2600 points)."""
    pytest.importorskip('deeprbf_recovered')
    X, o, h = sphere_holes(2.0); d = ds.signed_distance_codes(X, o)
    lo, step = 0.0, 2.0; ax = np.arange(lo + 1.0, 51.0, step)                  # cell centres of a 2 m grid over the cube
    G = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    f = ds.native_fit_predict('linear', 'linear')(X, d, G)
    return X, o, h, d, G, f, step ** 3

def test_codes_signs_and_distances(data):
    X, o, h = data; d = ds.signed_distance_codes(X, o)
    assert len(X) == 5100 and (d[o] < 0).all() and (d[~o] > 0).all()
    i = np.flatnonzero(h == 44)                                                # hole at (22.5, 22.5) through the sphere
    top = i[np.flatnonzero(o[i])[-1]]; assert d[top] == pytest.approx(-1.0) and d[top + 1] == pytest.approx(1.0)
    j = np.flatnonzero(h == 0)[25]                                             # (2.5, 2.5, 25.5): far waste
    ore_pts = X[o]; assert d[j] == pytest.approx(np.linalg.norm(ore_pts - X[j], axis=1).min())
    r = np.random.default_rng(1).choice(len(X), 600, replace=False)            # brute-force check on a subset
    assert np.allclose(ds.signed_distance_codes(X[r], o[r]), brute(X[r], o[r]))
    dc = ds.signed_distance_codes(X, o, holes=h)                               # in-hole contacts at the midpoint
    assert dc[top] == pytest.approx(-0.5) and dc[top + 1] == pytest.approx(0.5) and (np.abs(dc) <= np.abs(d) + 1e-12).all()
    assert np.allclose(ds.shift_codes(d, 2.0), d - 2.0)

def test_codes_in_frame_space(data):
    X, o, h = data; d = ds.signed_distance_codes(X, o)
    assert np.allclose(ds.signed_distance_codes(X, o, Frame(30, 45, 20, [10, 10, 10], CEN)), d)  # isotropic: metres kept
    r = np.random.default_rng(2).choice(len(X), 500, replace=False); F = Frame(20, 70, 90, [40.0, 20.0, 5.0], CEN)
    assert np.allclose(ds.signed_distance_codes(X[r], o[r], F), brute(F(X[r]), o[r]) * (40 * 20 * 5) ** (1 / 3))

def test_one_class_raises(data):
    X, o, h = data
    for bad in (np.ones(len(X), bool), np.zeros(len(X), bool)):
        with pytest.raises(ValueError): ds.signed_distance_codes(X, bad)
        with pytest.raises(ValueError): ds.calibrate_shift(X, bad, h, None, lambda a, b, c: np.zeros(len(c)))
    with pytest.raises(ValueError): ds.native_fit_predict('linear', 'none')

def test_native_zero_isosurface_volume(fit2m):
    X, o, h, d, G, f, cell = fit2m; true = 4 / 3 * np.pi * R ** 3
    assert abs((f <= 0).sum() * cell / true - 1) < 0.25
    inside = np.linalg.norm(G - CEN, axis=1) <= R; assert ((f <= 0) == inside).mean() > 0.95

def test_calibrate_shift_ordered(fit2m):
    X, o, h, d, G, f, cell = fit2m
    cal = ds.calibrate_shift(X, o, h, None, ds.native_fit_predict('linear', 'linear'), folds=4, seed=0)
    C = cal['C']; assert C[0.1] <= C[0.5] <= C[0.9] and C[0.1] < C[0.9]
    assert cal['label'] == {0.1: 'conservative', 0.5: 'expected', 0.9: 'aggressive'}
    for q, c in cal['coverage'].items(): assert abs(c - q) < 0.05
    assert cal['ore_inside'][0.1] <= cal['ore_inside'][0.9] and cal['waste_outside'][0.1] >= cal['waste_outside'][0.9]
    assert len(cal['folds']) == 4 and cal['n_contacts'] > 50
    V = {q: (ds.shift_codes(f, c) <= 0).sum() * cell for q, c in C.items()}   # shift applied to the all-holes field
    assert V[0.1] <= V[0.5] <= V[0.9] and V[0.1] < V[0.9]
    assert abs(V[0.5] / (4 / 3 * np.pi * R ** 3) - 1) < 0.25
