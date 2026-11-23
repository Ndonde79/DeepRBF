"""3.5.1: two-level preconditioner forms for iterative fits.

Before 3.5.1 the iterative path stalled for the cubic and multiquadric kernels on drillhole data (the additive two-level
form; the cubic also with the H-matrix hybrid form). The symmetric form (coarse, local, coarse) converges; "auto" uses it
for those kernels and keeps the 3.5.0 form for the others.
"""
import unittest

import numpy as np
import deeprbf_recovered as d

M = d.NativeExactRBFImplicitModel


def drillholes(n, seed=3):
    """Holes on a jittered 25 m grid, 2 m samples down a 60 degree dip, 60 per hole; a smooth contact. Units of 100 m."""
    rng = np.random.default_rng(seed)
    per = 60
    holes = int(np.ceil(n / per))
    g = int(np.ceil(np.sqrt(holes)))
    pts = []
    for k in range(holes):
        x0, y0 = (k % g) * 25.0 + rng.normal(0, 3), (k // g) * 25.0 + rng.normal(0, 3)
        t = np.arange(per) * 2.0
        pts.append(np.c_[np.full(per, x0), y0 + 0.5 * t, -0.866 * t])
    x = np.vstack(pts)[:n]
    y = np.tanh((x[:, 2] + 60 + 15 * np.sin(x[:, 0] / 80)) / 8.0)
    return x / 100.0, y


def reset():
    M.set_preconditioner_form("auto")
    M.force_iterative_solver(False)
    M.set_hmatrix("auto")
    M.set_iterative_solver(150, 3000, 1e-6)
    M.set_dense_direct_limit(8000)


class PreconditionerFormTest(unittest.TestCase):
    def tearDown(self):
        reset()

    def fit(self, kernel, eps, drift, n, form="auto", hmatrix="off", max_iter=3000):
        reset()
        M.set_preconditioner_form(form)
        M.force_iterative_solver(True)
        M.set_hmatrix(hmatrix)
        M.set_iterative_solver(150, max_iter, 1e-6)
        x, y = drillholes(n)
        m = M(kernel, eps, 1e-6)
        m.set_drift(drift)
        m.fit(x.tolist(), y.tolist())
        return m, x, y

    def dense(self, kernel, eps, drift, x, y):
        reset()
        m = M(kernel, eps, 1e-6)
        m.set_drift(drift)
        m.fit(x.tolist(), y.tolist())
        return m

    def test_api(self):
        reset()
        self.assertEqual(M.preconditioner_form(), "auto")
        for f in ["symmetric", "hybrid", "additive", "auto"]:
            M.set_preconditioner_form(f)
            self.assertEqual(M.preconditioner_form(), f)
        with self.assertRaises(ValueError):
            M.set_preconditioner_form("bogus")

    def test_cubic_auto_converges_and_matches_dense(self):
        m, x, y = self.fit("cubic", 1.0, "linear", 4000)
        self.assertTrue(m.solver_backend().endswith("_symmetric"), m.solver_backend())
        self.assertEqual(m.iterative_info()["preconditioner_symmetric"], 1.0)
        q = x[::13] + 0.004
        a = np.asarray(m.predict(q.tolist()))
        b = np.asarray(self.dense("cubic", 1.0, "linear", x, y).predict(q.tolist()))
        self.assertLess(np.abs(a - b).max(), 1e-3)

    def test_cubic_additive_still_stalls(self):
        # Documents the 3.5.0 behaviour that motivated the change (fails within a short iteration budget).
        with self.assertRaises(RuntimeError):
            self.fit("cubic", 1.0, "linear", 4000, form="additive", max_iter=150)

    def test_multiquadric_auto_converges(self):
        m, x, y = self.fit("multiquadric", 0.3, "constant", 2000)
        self.assertTrue(m.solver_backend().endswith("_symmetric"), m.solver_backend())
        self.assertLess(m.iterative_info()["sampled_rel_residual"], 1e-4)

    def test_auto_keeps_350_form_for_gen_cauchy(self):
        m, x, y = self.fit("gen_cauchy", 1.5, "constant", 3000)
        self.assertFalse(m.solver_backend().endswith("_symmetric"), m.solver_backend())
        self.assertEqual(m.iterative_info()["preconditioner_symmetric"], 0.0)


if __name__ == "__main__":
    unittest.main()
