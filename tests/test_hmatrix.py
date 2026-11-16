"""3.5.0: H-matrix/ACA product and its use in the iterative RBF solver."""
import unittest

import numpy as np
import deeprbf_recovered as d

M = d.NativeExactRBFImplicitModel


def drillholes(n_holes=40, per_hole=60, spacing=25.0, seed=1):
    """Holes on a jittered grid, 2 m samples down a 60 degree dip, plus a few isolated samples."""
    rng = np.random.default_rng(seed)
    g = int(np.ceil(np.sqrt(n_holes)))
    pts = []
    for k in range(n_holes):
        x0, y0 = (k % g) * spacing + rng.normal(0, 3), (k // g) * spacing + rng.normal(0, 3)
        t = np.arange(per_hole) * 2.0
        pts.append(np.c_[np.full(per_hole, x0), y0 + 0.5 * t, -0.866 * t])
    pts.append(rng.uniform([0, 0, -100], [g * spacing, g * spacing, 0], (25, 3)))
    x = np.vstack(pts)
    y = np.tanh((x[:, 2] + 50 + 15 * np.sin(x[:, 0] / 80)) / 8.0)
    return x, y


def gen_cauchy(r, eps, a=9.0, p=1.667):
    b = eps / (0.04 ** (-p / a) - 1.0) ** (1.0 / p)
    return (1.0 + (r / b) ** p) ** (-a / p)


KERNELS = {
    "gen_cauchy": gen_cauchy,
    "gaussian": lambda r, e: np.exp(-(r / e) ** 2),
    "exponential": lambda r, e: np.exp(-3.0 * r / e),
    "linear": lambda r, e: r,
    "cubic": lambda r, e: r ** 3,
    "multiquadric": lambda r, e: np.hypot(r, e),
}


def reset():
    M.set_hmatrix("auto")
    M.force_iterative_solver(False)
    M.set_dense_direct_limit(8000)
    M.set_dense_memory_limit_mb(1024.0)


class HMatrixProduct(unittest.TestCase):
    def test_product_matches_exact(self):
        x, _ = drillholes()
        r = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        w = np.random.default_rng(0).normal(size=len(x))
        for name, f in KERNELS.items():
            for eps in (30.0, 150.0):
                h = d.HMatrix(x.tolist(), name, eps, 1e-9)
                ref = f(r, eps) @ w
                got = np.asarray(h.apply(w.tolist()))
                err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
                self.assertLess(err, 1e-7, f"{name} eps={eps}: relative error {err:.1e}")
                st = h.stats()
                self.assertGreater(st["n_lowrank"], 0)
                self.assertLess(st["compression"], 1.0)

    def test_near_field_recomputed_when_over_budget(self):
        x, _ = drillholes(n_holes=16)
        w = np.random.default_rng(1).normal(size=len(x))
        stored = d.HMatrix(x.tolist(), "gen_cauchy", 100.0, 1e-9)
        recomputed = d.HMatrix(x.tolist(), "gen_cauchy", 100.0, 1e-9, near_memory_mb=0.0)
        self.assertTrue(stored.stats()["near_stored"])
        self.assertFalse(recomputed.stats()["near_stored"])
        a, b = np.asarray(stored.apply(w.tolist())), np.asarray(recomputed.apply(w.tolist()))
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-13)

    def test_identical_points_and_bad_input(self):
        x = np.zeros((300, 3)); x[150:, 0] = 10.0          # two piles of identical points
        h = d.HMatrix(x.tolist(), "gen_cauchy", 50.0)
        y = np.asarray(h.apply(np.ones(300).tolist()))
        ref = 150 * gen_cauchy(0.0, 50.0) + 150 * gen_cauchy(10.0, 50.0)
        self.assertLess(np.abs(y - ref).max(), 1e-9 * ref)
        with self.assertRaises(Exception):
            d.HMatrix([[0, 0, float("nan")]], "gen_cauchy", 50.0)
        with self.assertRaises(Exception):
            d.HMatrix(x.tolist(), "gen_cauchy", 50.0, tol=0.0)


class HMatrixSolver(unittest.TestCase):
    def tearDown(self):
        reset()

    def test_auto_uses_hmatrix_for_gen_cauchy_and_matches_dense(self):
        x, y = drillholes(n_holes=70)          # 4,225 points: above the auto threshold
        q = np.random.default_rng(5).uniform(x.min(0), x.max(0), (500, 3))
        reset()
        M.set_dense_direct_limit(10 ** 6); M.set_dense_memory_limit_mb(1e6)
        dense = M("gen_cauchy", 150.0, 1e-6); dense.set_drift("constant"); dense.fit(x.tolist(), y.tolist())
        reset()
        M.force_iterative_solver(True)
        it = M("gen_cauchy", 150.0, 1e-6); it.set_drift("constant"); it.fit(x.tolist(), y.tolist())
        info = it.iterative_info()
        self.assertEqual(info["hmatrix"], 1.0)
        self.assertEqual(info["hmatrix_fallback"], 0.0)
        self.assertIn("hmatrix_aca", it.solver_backend())
        self.assertLess(info["sampled_rel_residual"], 1e-4)
        a, b = np.asarray(dense.predict(q.tolist())), np.asarray(it.predict(q.tolist()))
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-4)

    def test_auto_keeps_exact_product_for_gaussian(self):
        x, y = drillholes(n_holes=70)
        reset(); M.force_iterative_solver(True)
        m = M("gaussian", 3.0, 1e-6); m.set_drift("constant"); m.fit(x.tolist(), y.tolist())
        self.assertEqual(m.iterative_info()["hmatrix"], 0.0)

    def test_failed_check_falls_back_to_exact_product(self):
        x, y = drillholes(n_holes=30)
        reset(); M.force_iterative_solver(True)
        M.set_hmatrix("on", tol=1e-2)                    # deliberately coarse: the residual check must fail
        m = M("gen_cauchy", 150.0, 1e-6); m.set_drift("constant"); m.fit(x.tolist(), y.tolist())
        info = m.iterative_info()
        self.assertEqual(info["hmatrix_fallback"], 1.0)
        self.assertGreater(info["hmatrix_sampled_rel_residual"], info["hmatrix_check_limit"])
        self.assertLess(info["sampled_rel_residual"], 1e-4)
        self.assertIn("after_hmatrix_check", m.solver_backend())

    def test_set_hmatrix_validates(self):
        for bad in [dict(mode="sometimes"), dict(mode="on", tol=2.0), dict(mode="on", eta=-1.0),
                    dict(mode="on", leaf_size=2), dict(mode="on", near_memory_mb=-5.0)]:
            with self.assertRaises(Exception):
                M.set_hmatrix(**bad)
        reset()
        self.assertEqual(M.hmatrix_mode(), "auto")


if __name__ == "__main__":
    unittest.main()
