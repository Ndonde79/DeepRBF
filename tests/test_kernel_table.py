"""3.5.0: the tabulated kernel used in the O(n^2) loops must agree with the closed form."""
import unittest

import numpy as np
import deeprbf_recovered as d

TABULATED = ["gen_cauchy", "gen_cauchy:9:0.5", "gen_cauchy:3:2", "gaussian", "exponential",
             "inverse_multiquadric", "multiquadric", "thin_plate"]
SCALES = [(150.0, 3000.0), (5.0, 10000.0), (1000.0, 50.0), (0.01, 5.0e5)]


def gen_cauchy(r, eps, a=9.0, p=1.667):
    b = eps / (0.04 ** (-p / a) - 1.0) ** (1.0 / p)
    return (1.0 + (r / b) ** p) ** (-a / p)


class KernelTable(unittest.TestCase):
    def test_table_matches_closed_form(self):
        for k in TABULATED:
            for eps, dmax in SCALES:
                r = d.kernel_table_check(k, eps, dmax, 100000)
                self.assertTrue(r["tabulated"], f"{k} eps={eps} dmax={dmax}: table switched off")
                self.assertLess(r["max_rel_err"], 1e-13, f"{k} eps={eps} dmax={dmax}")

    def test_cheap_kernels_use_closed_form(self):
        for k in ["linear", "cubic", "spherical"]:
            self.assertFalse(d.kernel_table_check(k, 10.0, 100.0, 10)["tabulated"])

    def test_predict_matches_numpy_sum(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 500, (600, 3))
        y = np.sin(x[:, 0] / 70) + 0.01 * x[:, 2]
        m = d.NativeExactRBFImplicitModel("gen_cauchy", 120.0, 1e-6)
        m.set_drift("constant")
        m.fit(x.tolist(), y.tolist())
        w = np.asarray(m.weights())
        q = rng.uniform(-50, 550, (400, 3))
        r = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=2)
        ref = gen_cauchy(r, 120.0) @ w[:600] + w[600]
        got = np.asarray(m.predict(q.tolist()))
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), 1e-12)


if __name__ == "__main__":
    unittest.main()
