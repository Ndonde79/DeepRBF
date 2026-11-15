# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Regularisation must smooth for every kernel (3.4.2). For order-1 CPD kernels (linear, multiquadric) the engine
applies the regularisation with a negative sign; before 3.4.2 a positive nu drove those systems towards singularity
and the fit error grew far beyond the noise."""
import unittest
import numpy as np
import deeprbf_recovered as d


def _fit(kernel, drift, nu, X, f, solver=None):
    if solver == "iterative":
        d.NativeExactRBFImplicitModel.force_iterative_solver(True)
    try:
        m = d.NativeExactRBFImplicitModel(kernel, 0.3, nu)
        m.set_drift(drift)
        m.fit(X.tolist(), f.tolist())
        return np.asarray(m.predict(X.tolist())), m.solver_backend()
    finally:
        d.NativeExactRBFImplicitModel.force_iterative_solver(False)


class RegularizationSign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.uniform(0, 1, (400, 3))
        cls.truth = np.sin(4 * cls.X[:, 0])
        cls.f = cls.truth + 0.3 * rng.normal(size=400)

    def check(self, kernel, drift, solver=None):
        rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        for nu in (1e-3, 1e-2, 1e-1, 1.0):
            p, backend = _fit(kernel, drift, nu, self.X, self.f, solver)
            with self.subTest(kernel=kernel, nu=nu, backend=backend):
                self.assertLess(rms(p, self.f), 0.35)          # never worse than the noise level
                self.assertLess(rms(p, self.truth), 0.33)      # near or below the noise level (the pre-3.4.2 bug gave 8-18)
        p0, _ = _fit(kernel, drift, 0.0, self.X, self.f, solver)
        self.assertLess(rms(p0, self.f), 1e-6)                 # nu = 0 still interpolates exactly

    def test_linear_dense(self): self.check("linear", "constant")
    def test_multiquadric_dense(self): self.check("multiquadric", "constant")
    def test_cubic_dense(self): self.check("cubic", "linear")
    def test_gaussian_dense(self): self.check("gaussian", "none")
    def test_linear_iterative(self): self.check("linear", "constant", "iterative")


if __name__ == "__main__":
    unittest.main()
