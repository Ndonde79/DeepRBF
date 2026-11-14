# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""3.3.2 large-RBF solver: iterative path on drillhole-like data, solver controls, explicit predict backend."""
import unittest
import os, sys
_dll_handle = os.add_dll_directory(os.path.join(sys.prefix, "Library", "bin")) if os.name == "nt" else None
import numpy as np
import deeprbf_recovered as d

M = d.NativeExactRBFImplicitModel


def drillholes(holes=40, samples=60, seed=7):
    """Inclined holes with 2 m spacing through a dipping lens: +1 inside, -1 outside, 0 at down-hole contacts."""
    rng = np.random.default_rng(seed); pts = []; val = []
    for _ in range(holes):
        c = rng.uniform(-150, 150, 2); az = rng.uniform(0, 2 * np.pi); dip = np.radians(rng.uniform(55, 80))
        u = np.array([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), -np.sin(dip)])
        p = np.array([c[0], c[1], 0.0]) + np.outer(np.arange(samples) * 2.0, u)
        s = p[:, 2] + 0.4 * p[:, 0] + 60.0                                    # lens: |s| < 6
        f = np.where(np.abs(s) < 6.0, 1.0, -1.0)
        k = np.flatnonzero(f[1:] != f[:-1])
        pts += [p, (p[k] + p[k + 1]) / 2]; val += [f, np.zeros(len(k))]
    x = np.vstack(pts) / 100.0
    return x, np.concatenate(val)


class LargeSolver(unittest.TestCase):
    def tearDown(self):
        M.force_iterative_solver(False); M.set_iterative_solver(150, 3000, 1e-6)
        M.set_dense_memory_limit_mb(1024.0); M.set_predict_backend("auto")

    def test_controls_validate(self):
        self.assertEqual(M.predict_backend(), "auto")
        for bad in (lambda: M.set_predict_backend("fast"), lambda: M.set_iterative_solver(0, 10, 1e-6),
                    lambda: M.set_iterative_solver(10, 10, 2.0), lambda: M.set_dense_memory_limit_mb(0.0)):
            with self.assertRaises((ValueError, RuntimeError)): bad()

    def test_iterative_matches_dense_on_drillhole_data(self):
        x, y = drillholes()
        dense = M("linear", 1.0, 0.0); dense.fit(x.tolist(), y.tolist())
        M.force_iterative_solver(True)
        it = M("linear", 1.0, 0.0); it.fit(x.tolist(), y.tolist())
        self.assertIn("iterative", it.solver_backend())
        q = np.random.default_rng(3).uniform(x.min(0), x.max(0), (400, 3))
        a = np.asarray(dense.predict(q.tolist())); b = np.asarray(it.predict(q.tolist()))
        self.assertLess(np.abs(a - b).max(), 1e-4 * max(1.0, np.abs(a).max()))
        fit = np.asarray(it.predict(x.tolist())); nz = y != 0
        self.assertTrue(np.all(np.sign(fit[nz]) == np.sign(y[nz])))
        self.assertLess(np.abs(fit - y).max(), 1e-3)

    def test_predict_backend_is_explicit(self):
        x, y = drillholes(holes=12, samples=30)
        m = M("gaussian", 0.5, 1e-3); m.fit(x.tolist(), y.tolist())
        q = np.random.default_rng(4).uniform(x.min(0), x.max(0), (200, 3)).tolist()
        M.set_predict_backend("direct"); exact = np.asarray(m.predict(q))
        self.assertEqual(m.last_predict_backend(), "direct_openmp")
        M.set_predict_backend("scalfmm")
        if not d.__has_scalfmm__:
            with self.assertRaises(RuntimeError): m.predict(q)
        else:
            approx = np.asarray(m.predict(q))
            self.assertEqual(m.last_predict_backend(), "scalfmm")
            self.assertLess(np.linalg.norm(approx - exact) / np.linalg.norm(exact), 2e-3)


if __name__ == "__main__":
    unittest.main()
