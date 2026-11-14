# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
from __future__ import annotations

import numpy as np


def _transform(x: np.ndarray, anisotropy: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (x - origin) @ anisotropy.T


def _kernel(r: np.ndarray, epsilon: float, kernel: str) -> np.ndarray:
    kernel = kernel.lower()
    er = epsilon * r
    if kernel == "gaussian":
        return np.exp(-(er ** 2))
    if kernel == "multiquadric":
        return np.sqrt(1.0 + er ** 2)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + er ** 2)
    if kernel == "linear":
        return r
    if kernel == "cubic":
        return r ** 3
    raise ValueError(f"Unsupported kernel: {kernel}")


def _distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def fit_rbf_fallback(points, values, epsilon, kernel, anisotropy, origin, smoothing):
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if points.ndim != 2:
        raise ValueError("points must be 2D")
    if len(points) != len(values):
        raise ValueError("points and values length mismatch")
    if len(points) < 2:
        raise ValueError("at least two points are required")

    xt = _transform(points, anisotropy, origin)
    r = _distance_matrix(xt, xt)
    k = _kernel(r, epsilon, kernel)
    k.flat[:: k.shape[0] + 1] += float(smoothing)

    p = np.column_stack([np.ones(len(points)), xt])
    z = np.zeros((p.shape[1], p.shape[1]))
    a = np.block([[k, p], [p.T, z]])
    rhs = np.concatenate([values, np.zeros(p.shape[1])])

    sol = np.linalg.solve(a, rhs)
    w = sol[: len(points)]
    poly = sol[len(points):]
    return {
        "centers": points.copy(),
        "weights": w,
        "poly": poly,
        "epsilon": float(epsilon),
        "kernel": str(kernel),
        "anisotropy": anisotropy.copy(),
        "origin": origin.copy(),
    }


def predict_rbf_fallback(points, centers, weights, poly, epsilon, kernel, anisotropy, origin):
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    xt = _transform(points, anisotropy, origin)
    ct = _transform(centers, anisotropy, origin)
    r = _distance_matrix(xt, ct)
    k = _kernel(r, epsilon, kernel)
    p = np.column_stack([np.ones(len(points)), xt])
    return k @ weights + p @ poly
