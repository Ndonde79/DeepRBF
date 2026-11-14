# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Drift option (none / constant / linear) on NativeExactRBFImplicitModel and the gen_cauchy covariance kernel."""
import os
import sys

sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))

import numpy as np
import pytest

import deeprbf_recovered as n

M = n.NativeExactRBFImplicitModel

VALID_DRIFTS = {
    "spherical": ["none", "constant", "linear"],
    "gen_cauchy": ["none", "constant", "linear"],
}


@pytest.fixture(autouse=True)
def _reset_solver():
    yield
    M.force_iterative_solver(False)


def _data(npts, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (npts, 3))
    y = np.sin(3.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2] + 0.1 * rng.standard_normal(npts)
    return x, y


def test_default_drift_is_linear():
    m = M("gaussian", 1.0, 1e-6)
    assert m.drift() == "linear"


@pytest.mark.parametrize("kernel,eps", [("spherical", 0.6), ("gen_cauchy", 0.5), ("gen_cauchy:6:1.5", 0.5)])
@pytest.mark.parametrize("drift", ["none", "constant", "linear"])
def test_exact_interpolation_zero_regularization(kernel, eps, drift):
    x, y = _data(80, 1)
    m = M(kernel, eps, 0.0)
    m.set_drift(drift)
    assert m.drift() == drift
    m.fit(x.tolist(), y.tolist())
    assert m.solver_backend().startswith("dense")
    p = np.asarray(m.predict(x.tolist()))
    assert np.max(np.abs(p - y)) < 1e-6


def test_spherical_no_drift_is_zero_beyond_range():
    x, y = _data(60, 2)
    eps = 0.3
    m = M("spherical", eps, 0.0)
    m.set_drift("none")
    m.fit(x.tolist(), (y + 5.0).tolist())
    far = [[1.0 + 3.5 * eps, 0.5, 0.5], [-3.5 * eps, -3.5 * eps, 0.0], [0.5, 0.5, 1.0 + 4.0 * eps]]
    assert np.all(np.asarray(m.predict(far)) == 0.0)
    # with a constant drift the far-field value is the drift constant, not zero
    mc = M("spherical", eps, 0.0)
    mc.set_drift("constant")
    mc.fit(x.tolist(), (y + 5.0).tolist())
    pc = np.asarray(mc.predict(far))
    assert np.allclose(pc, pc[0]) and abs(pc[0]) > 1.0


@pytest.mark.parametrize("kernel,eps", [("spherical", 0.5), ("gen_cauchy", 0.4)])
@pytest.mark.parametrize("drift", ["none", "constant", "linear"])
def test_dense_and_iterative_agree(kernel, eps, drift):
    x, y = _data(600, 3)
    q = np.random.default_rng(4).uniform(0.0, 1.0, (200, 3)).tolist()
    preds = []
    for force in (False, True):
        M.force_iterative_solver(force)
        m = M(kernel, eps, 1e-6)
        m.set_drift(drift)
        m.fit(x.tolist(), y.tolist())
        assert m.solver_backend().startswith("iterative" if force else "dense")
        preds.append(np.asarray(m.predict(q)))
    assert np.max(np.abs(preds[0] - preds[1])) < 1e-4


@pytest.mark.parametrize("kernel", ["multiquadric", "linear", "cubic", "thin_plate"])
def test_refuse_no_drift_for_non_pd_kernels(kernel):
    m = M(kernel, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive definite"):
        m.set_drift("none")
    assert m.drift() == "linear"


@pytest.mark.parametrize("kernel", ["cubic", "thin_plate"])
def test_refuse_constant_drift_for_order_two_kernels(kernel):
    m = M(kernel, 1.0, 0.0)
    with pytest.raises(ValueError, match="linear"):
        m.set_drift("constant")


@pytest.mark.parametrize("kernel", ["multiquadric", "linear", "gaussian"])
def test_constant_drift_accepted(kernel):
    m = M(kernel, 0.2, 0.0)
    m.set_drift("constant")
    x, y = _data(40, 5)
    m.fit(x.tolist(), y.tolist())
    assert np.max(np.abs(np.asarray(m.predict(x.tolist())) - y)) < 1e-6


def test_refuse_unknown_drift():
    with pytest.raises(ValueError):
        M("gaussian", 1.0, 0.0).set_drift("quadratic")


def test_gen_cauchy_kernel_value():
    for eps in (0.3, 1.0, 250.0):
        assert n.kernel_value("gen_cauchy", 0.0, eps) == pytest.approx(1.0, abs=1e-15)
        assert n.kernel_value("gen_cauchy", eps, eps) == pytest.approx(0.04, rel=1e-12)
        assert n.kernel_value("gen_cauchy:5:2", eps, eps) == pytest.approx(0.04, rel=1e-12)
    # long tail: still positive well past the range, monotone decreasing
    r = np.linspace(0.0, 3.0, 31)
    v = np.array([n.kernel_value("gen_cauchy", float(t), 1.0) for t in r])
    assert np.all(np.diff(v) < 0) and v[-1] > 0.0
    # explicit default shape string matches the bare name
    assert n.kernel_value("gen_cauchy:9:1.667", 0.7, 1.0) == pytest.approx(n.kernel_value("gen_cauchy", 0.7, 1.0), rel=1e-14)


def test_gen_cauchy_shape_setter_and_invalid_shapes():
    m = M("gen_cauchy", 1.0, 0.0)
    m.set_kernel_shape(6.0, 1.5)
    assert m.kernel().startswith("gen_cauchy:6:1.5")
    with pytest.raises(ValueError):
        m.set_kernel_shape(9.0, 2.5)          # power must be <= 2 for positive definiteness
    with pytest.raises(ValueError):
        m.set_kernel_shape(-1.0, 1.0)
    with pytest.raises(ValueError):
        M("spherical", 1.0, 0.0).set_kernel_shape(9.0, 1.667)
    with pytest.raises(Exception):
        M("gen_cauchy:abc", 1.0, 0.0)


def test_spherical_and_exponential_kernel_values():
    assert n.kernel_value("spherical", 0.0, 2.0) == 1.0
    assert n.kernel_value("spherical", 2.0, 2.0) == 0.0
    assert n.kernel_value("exponential", 1.0, 1.0) == pytest.approx(np.exp(-3.0))
