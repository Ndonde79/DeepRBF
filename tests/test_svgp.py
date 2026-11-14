# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""SVGP module: classifier finds a dipping lens, regressor beats the mean with calibrated intervals, block support works."""
import numpy as np
import pytest

svgp = pytest.importorskip("deeprbf.svgp")
if not svgp.svgp_available()[0]:
    pytest.skip("torch not available", allow_module_level=True)


def _data(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform([-200, -200, -150], [200, 200, 0], (n, 3))
    nrm = np.array([np.sin(np.radians(60)) * np.sin(np.radians(200)), np.sin(np.radians(60)) * np.cos(np.radians(200)), np.cos(np.radians(60))])
    s = X @ nrm + 40
    lab = np.where(np.abs(s) < 10, "ore", np.where(s > 0, "hw", "fw"))
    g = np.exp(1.0 + 0.8 * np.sin(X[:, 0] / 60) + 0.6 * np.cos(X[:, 1] / 80) + rng.normal(0, 0.4, n))
    return X, lab, g, rng


def test_classifier_beats_majority():
    X, lab, _, rng = _data()
    te = rng.random(len(X)) < 0.25
    c = svgp.SVGPClassifier(n_inducing=150, iterations=400, batch_size=512, lr=0.02, seed=1, device="cpu").fit(X[~te], lab[~te])
    p = c.predict_proba(X[te], samples=32)
    assert np.allclose(p.sum(1), 1)
    acc = (np.array(c.classes)[p.argmax(1)] == lab[te]).mean()
    assert acc > max((lab[te] == k).mean() for k in c.classes) + 0.15


def test_regressor_unbiased_calibrated_and_block():
    X, _, g, rng = _data(seed=2)
    te = rng.random(len(X)) < 0.25
    r = svgp.SVGPRegressor(n_inducing=150, iterations=600, batch_size=512, lr=0.02, seed=1, device="cpu").fit(X[~te], g[~te])
    m, q = r.predict(X[te], return_quantiles=True)
    assert abs(m.mean() / g[te].mean() - 1) < 0.08
    assert np.sqrt(((m - g[te]) ** 2).mean()) < 0.8 * np.sqrt(((g[~te].mean() - g[te]) ** 2).mean())
    assert 0.65 < svgp.coverage(g[te], q[10], q[90]) < 0.95
    b = r.predict(X[te][:10], block=(20, 20, 20), disc=(2, 2, 2))
    assert b.shape == (10,) and np.isfinite(b).all() and b.min() >= g.min()


def test_anamorphosis_round_trip():
    v = np.random.default_rng(3).lognormal(0, 1, 500)
    a = svgp.Anamorphosis(v)
    assert np.allclose(a.inverse(a.transform(v)), v)


def test_compositional_valid_and_ratio():
    X, _, g, rng = _data(seed=4)
    zn = g * 3; pb = zn * 0.4 * np.exp(rng.normal(0, 0.15, len(g))); cu = np.exp(rng.normal(-0.5, 0.5, len(g)))
    Y = np.c_[zn, pb, cu]; Y[:20, 2] = 0
    te = rng.random(len(X)) < 0.25
    c = svgp.SVGPCompositional(n_inducing=120, iterations=500, batch_size=512, lr=0.02, seed=1, device="cpu").fit(X[~te], Y[~te])
    lv = svgp.interval_levels((80,))
    m, q = c.predict(X[te], return_quantiles=True, levels=lv)
    assert m.shape == (te.sum(), 3) and (m > 0).all() and (m.sum(1) < 100).all()
    assert abs(np.median(m[:, 0] / m[:, 1]) / np.median(zn / pb) - 1) < 0.1
    assert abs(m[:, 0].mean() / zn[te].mean() - 1) < 0.1
    assert 0.65 < svgp.coverage_curve(Y[te, 0], {p: q[p][:, 0] for p in lv}, (80,))[80] < 0.97
