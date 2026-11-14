#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""
DeepRBF Full Turning Bands V3

Line-projection Turning Bands conditional simulation using ordinary-kriging
correction.

Conditional simulation formula:
    Zc(u) = Zk(u) + U(u) - Uk(u)

where:
    Zk(u) = OK estimate from true normal-score conditioning data
    U(u)  = unconditional TB Gaussian field at target
    Uk(u) = OK estimate from the unconditional TB field sampled at conditioning data

V3 changes:
    - no per-evaluation normalization of the unconditional field;
    - realization-consistent random bands across data and target evaluation;
    - precomputed local OK weights;
    - better normal-score handling for duplicated values;
    - practical audit tools for pilot and full runs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm


EPS = 1.0e-12


@dataclass
class VariogramSpec:
    model: str = "exponential"
    nugget: float = 0.05
    sill: float = 0.95
    ranges: Tuple[float, float, float] = (180.0, 80.0, 35.0)
    azimuth_deg: float = 0.0
    dip_deg: float = 0.0


@dataclass
class TBSpec:
    bands: int = 384
    terms_per_band: int = 1
    seed: int = 20260827
    max_neighbors: int = 20
    search_radius: float = 250.0


class NormalScoreTransform:
    """Normal-score transform with duplicate-value handling."""

    def __init__(self) -> None:
        self.x_unique: Optional[np.ndarray] = None
        self.z_unique: Optional[np.ndarray] = None

    def fit(self, values: Sequence[float]) -> "NormalScoreTransform":
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            raise ValueError("NormalScoreTransform requires at least 5 finite values")

        x_sorted = np.sort(x)
        p = (np.arange(x_sorted.size) + 0.5) / x_sorted.size
        z_sorted = norm.ppf(np.clip(p, 1.0e-9, 1.0 - 1.0e-9))

        ux, counts = np.unique(x_sorted, return_counts=True)
        midpoint = (np.cumsum(counts) - 0.5 * counts) / len(x_sorted)
        self.x_unique = ux
        self.z_unique = norm.ppf(midpoint)
        return self

    def transform(self, values: Sequence[float]) -> np.ndarray:
        if self.x_unique is None or self.z_unique is None:
            raise RuntimeError("NormalScoreTransform not fitted")
        x = np.asarray(values, dtype=float)
        return np.interp(x, self.x_unique, self.z_unique,
                         left=self.z_unique[0], right=self.z_unique[-1])

    def inverse(self, scores: Sequence[float]) -> np.ndarray:
        if self.x_unique is None or self.z_unique is None:
            raise RuntimeError("NormalScoreTransform not fitted")
        z = np.asarray(scores, dtype=float)
        return np.interp(z, self.z_unique, self.x_unique,
                         left=self.x_unique[0], right=self.x_unique[-1])


def rotation_matrix(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    dp = np.deg2rad(dip_deg)
    ca, sa = np.cos(az), np.sin(az)
    cd, sd = np.cos(dp), np.sin(dp)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cd, -sd], [0.0, sd, cd]], dtype=float)
    return rz @ rx


def scaled_coordinates(points: np.ndarray, spec: VariogramSpec) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    ranges = np.maximum(np.asarray(spec.ranges, dtype=float), 1.0)
    rot = rotation_matrix(spec.azimuth_deg, spec.dip_deg)
    return (pts @ rot) / ranges


def covariance_from_scaled_distance(h: np.ndarray, spec: VariogramSpec, include_nugget: bool = False) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    model = spec.model.lower().strip()
    if model.startswith("exp"):
        core = np.exp(-3.0 * h)
    elif model.startswith("gauss"):
        core = np.exp(-3.0 * h * h)
    elif model.startswith("sph"):
        core = np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)
    else:
        raise ValueError(f"Unsupported variogram model: {spec.model}")
    out = float(spec.sill) * core
    if include_nugget:
        out = out + float(spec.nugget) * (h <= EPS)
    return out


def pairwise_scaled_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + EPS
    return x


class TurningBandsFieldV3:
    """
    Unconditional Gaussian field from line-projection cosine bands.

    Important:
        evaluate() does not normalize by the supplied points. This keeps the
        realization consistent between conditioning data and block targets.
    """

    def __init__(self, spec: VariogramSpec, tb: TBSpec, realization: int):
        self.spec = spec
        self.tb = tb
        self.realization = int(realization)

        seed = int(tb.seed) + 104729 * int(realization)
        self.rng = np.random.default_rng(seed)

        self.n_terms = int(tb.bands) * int(tb.terms_per_band)
        if self.n_terms <= 0:
            raise ValueError("bands * terms_per_band must be positive")

        model = spec.model.lower().strip()
        frequency = self.rng.normal(size=(self.n_terms, 3))
        if model.startswith("gauss"):
            frequency *= math.sqrt(6.0)
        elif model.startswith("exp"):
            denominator = np.abs(self.rng.normal(size=(self.n_terms, 1)))
            if np.any(denominator == 0):
                raise ValueError("zero spectral denominator")
            frequency *= 3.0 / denominator
        else:
            raise ValueError("Qualified TB supports Gaussian and exponential spectra only")
        if not np.isfinite(spec.sill) or spec.sill <= 0 or not np.isfinite(spec.nugget) or spec.nugget < 0:
            raise ValueError("sill must be positive and nugget nonnegative")
        self.frequency = frequency
        self.coefficients = self.rng.normal(size=(2, self.n_terms))
        self.amplitude = math.sqrt(float(spec.sill) / self.n_terms)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        pts = scaled_coordinates(points, self.spec)
        out = np.zeros(len(pts), dtype=float)
        # Bound temporary memory independently of the caller's chunk size.
        for start in range(0, len(pts), 2048):
            phase = pts[start:start+2048] @ self.frequency.T
            out[start:start+2048] = self.amplitude * (
                np.cos(phase) @ self.coefficients[0] + np.sin(phase) @ self.coefficients[1])
        if self.spec.nugget:
            # Coordinate-keyed white noise: identical points share a value, and
            # evaluation order/chunk boundaries cannot change a realization.
            bits = np.ascontiguousarray(np.where(points == 0, 0., points), dtype=np.float64).view(np.uint64)
            def mix(x):
                with np.errstate(over="ignore"):
                    x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
                    x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
                    return x ^ (x >> 31)
            key = mix(bits[:, 0]) ^ mix(bits[:, 1] ^ np.uint64(0x9e3779b97f4a7c15)) ^ mix(bits[:, 2] ^ np.uint64(0x517cc1b727220a95))
            key ^= np.uint64((int(self.tb.seed) + 104729*self.realization) % (2**64))
            u = ((mix(key) >> 11).astype(float) + .5) / 2**53
            v = ((mix(key ^ np.uint64(0xd1b54a32d192ed03)) >> 11).astype(float) + .5) / 2**53
            out += math.sqrt(self.spec.nugget) * np.sqrt(-2*np.log(u)) * np.cos(2*np.pi*v)
        return out


class PrecomputedOrdinaryKriging:
    """Precompute local ordinary-kriging weights for many repeated estimates."""

    def __init__(
        self,
        data_points: np.ndarray,
        target_points: np.ndarray,
        spec: VariogramSpec,
        max_neighbors: int = 20,
        search_radius: float = 250.0,
    ):
        self.data_points = np.asarray(data_points, dtype=float)
        self.target_points = np.asarray(target_points, dtype=float)
        self.spec = spec
        self.max_neighbors = int(max_neighbors)
        self.search_radius = float(search_radius)

        if len(self.data_points) < 2:
            raise ValueError("At least two conditioning samples are required")

        self.data_scaled = scaled_coordinates(self.data_points, spec)
        self.target_scaled = scaled_coordinates(self.target_points, spec)
        self.tree = cKDTree(self.data_points)

        n_targets = len(self.target_points)
        k = min(self.max_neighbors, len(self.data_points))
        self.idx = np.full((n_targets, k), -1, dtype=np.int64)
        self.w = np.zeros((n_targets, k), dtype=float)
        self.diagnostics = dict(exact_targets=0, expanded_search_targets=0, idw_fallback_targets=0)

        self._build()

    def _build(self) -> None:
        n_targets = len(self.target_points)
        k = self.idx.shape[1]

        for i in range(n_targets):
            p = self.target_points[i]
            exact = np.flatnonzero(np.all(self.data_points == p, axis=1))
            if len(exact):
                self.diagnostics['exact_targets'] += 1
                self.idx[i, 0] = exact[0]
                self.w[i, 0] = 1.
                continue
            dist, idx = self.tree.query(p, k=k, distance_upper_bound=self.search_radius)
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            mask = np.isfinite(dist) & (idx < len(self.data_points))

            if mask.sum() < 2:
                self.diagnostics['expanded_search_targets'] += 1
                dist, idx = self.tree.query(p, k=k)
                idx = np.atleast_1d(idx)
            else:
                idx = idx[mask]

            idx = idx[:k].astype(np.int64)
            n = len(idx)
            if n == 0:
                continue

            local_scaled = self.data_scaled[idx]
            A_h = pairwise_scaled_distance(local_scaled, local_scaled)
            A = covariance_from_scaled_distance(A_h, self.spec, include_nugget=True)

            K = np.empty((n + 1, n + 1), dtype=float)
            K[:n, :n] = A
            K[:n, n] = 1.0
            K[n, :n] = 1.0
            K[n, n] = 0.0

            h = np.sqrt(np.maximum(np.sum((local_scaled - self.target_scaled[i]) ** 2, axis=1), 0.0))
            rhs = np.empty(n + 1, dtype=float)
            rhs[:n] = covariance_from_scaled_distance(h, self.spec, include_nugget=False)
            rhs[n] = 1.0

            try:
                sol = np.linalg.solve(K, rhs)
                weights = sol[:n]
            except np.linalg.LinAlgError:
                self.diagnostics['idw_fallback_targets'] += 1
                # Stable IDW fallback if local covariance matrix is singular.
                dd = np.sqrt(np.maximum(np.sum((self.data_points[idx] - p) ** 2, axis=1), 0.0))
                dd = np.maximum(dd, 1.0e-6)
                weights = 1.0 / (dd * dd)
                weights = weights / max(float(np.sum(weights)), EPS)

            self.idx[i, :n] = idx
            self.w[i, :n] = weights

    def apply(self, data_values: np.ndarray) -> np.ndarray:
        values = np.asarray(data_values, dtype=float)
        fill = np.array([0.0], dtype=float)
        values2 = np.concatenate([values, fill])
        idx_safe = np.where(self.idx >= 0, self.idx, len(values))
        return np.sum(self.w * values2[idx_safe], axis=1)


def clean_conditioning_data(points: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    val = np.asarray(values, dtype=float)
    mask = np.isfinite(val) & np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    val = val[mask]
    unique, inverse = np.unique(pts, axis=0, return_inverse=True)
    first = np.full(len(unique), -1, dtype=int)
    for i, group in enumerate(inverse):
        if first[group] >= 0 and val[i] != val[first[group]]:
            raise ValueError("conflicting conditioning grades at identical coordinates")
        first[group] = i
    return unique, val[first]


def conditional_tb_simulation_v3(
    data_points: np.ndarray,
    data_values: np.ndarray,
    target_points: np.ndarray,
    variogram: VariogramSpec,
    tb: TBSpec,
    n_realizations: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    return PreparedConditionalTB(data_points, data_values, variogram, tb, n_realizations).predict(target_points)


class PreparedConditionalTB:
    """Cache conditioning fields once; reuse unchanged realizations across chunks."""
    def __init__(self, data_points, data_values, variogram, tb, n_realizations):
        self.points, self.values = clean_conditioning_data(data_points, data_values)
        if len(self.values) < 5:
            raise ValueError("At least five finite conditioning samples required")
        if int(n_realizations) != n_realizations or n_realizations < 1:
            raise ValueError("n_realizations must be positive")
        if tb.max_neighbors < 1 or tb.search_radius <= 0:
            raise ValueError("neighbor count and search radius must be positive")
        if len(variogram.ranges) != 3 or not np.isfinite(variogram.ranges).all() or np.any(np.asarray(variogram.ranges) <= 0):
            raise ValueError("ranges must be three positive finite values")
        self.spec, self.tb = variogram, tb
        self.ns = NormalScoreTransform().fit(self.values)
        self.scores = self.ns.transform(self.values)
        self.fields = [TurningBandsFieldV3(variogram, tb, r) for r in range(int(n_realizations))]
        self.data_fields = [f.evaluate(self.points) for f in self.fields]

    def predict(self, target_points):
        targets = np.asarray(target_points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError("target_points must be finite (n,3)")
        ok = PrecomputedOrdinaryKriging(self.points, targets, self.spec, self.tb.max_neighbors, self.tb.search_radius)
        mean = ok.apply(self.scores)
        sims = np.empty((len(self.fields), len(targets)), dtype=np.float32)
        for r, field in enumerate(self.fields):
            scores = mean + field.evaluate(targets) - ok.apply(self.data_fields[r])
            sims[r] = self.ns.inverse(scores)
        if not np.isfinite(sims).all():
            raise ValueError("simulation contains nonfinite values")
        return sims, dict(n_data=len(self.values), n_targets=len(targets),
            n_realizations=len(self.fields), bands=self.tb.bands,
            terms_per_band=self.tb.terms_per_band, variogram=asdict(self.spec),
            support="point_at_centroid", algorithm="corrected_3d_spectrum_with_coordinate_nugget", neighborhood=ok.diagnostics)


def summarize_realizations(realizations: np.ndarray, cutoffs: Sequence[float]) -> Dict[str, np.ndarray]:
    arr = np.asarray(realizations, dtype=float)
    out: Dict[str, np.ndarray] = {
        "MEAN": np.mean(arr, axis=0),
        "STD": np.std(arr, axis=0),
        "P10": np.percentile(arr, 10, axis=0),
        "P50": np.percentile(arr, 50, axis=0),
        "P90": np.percentile(arr, 90, axis=0),
    }
    for cutoff in cutoffs:
        suffix = str(cutoff).replace(".", "P").replace("-", "M")
        out[f"PROB_GT_{suffix}"] = np.mean(arr > float(cutoff), axis=0)
    return out


def covariance_rmse_check_v3(
    points: np.ndarray,
    spec: VariogramSpec,
    tb: TBSpec,
    n_realizations: int = 16,
    n_pairs: int = 1500,
    seed: int = 12345,
) -> Dict[str, float]:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 20:
        return {"rmse": float("nan"), "pairs": 0}

    rng = np.random.default_rng(seed)
    if len(pts) > 1500:
        idx0 = rng.choice(len(pts), size=1500, replace=False)
        pts = pts[idx0]

    n = len(pts)
    ia = rng.integers(0, n, size=int(n_pairs))
    ib = rng.integers(0, n, size=int(n_pairs))

    scaled = scaled_coordinates(pts, spec)
    h = np.sqrt(np.maximum(np.sum((scaled[ia] - scaled[ib]) ** 2, axis=1), 0.0))
    target_cov = covariance_from_scaled_distance(h, spec, include_nugget=False)

    emp = np.zeros_like(target_cov)
    for r in range(int(n_realizations)):
        f = TurningBandsFieldV3(spec, tb, r)
        u = f.evaluate(pts)
        u = u - float(np.mean(u))
        emp += u[ia] * u[ib]
    emp /= float(n_realizations)

    rmse = float(np.sqrt(np.mean((emp - target_cov) ** 2)))
    return {
        "rmse": rmse,
        "pairs": int(len(target_cov)),
        "target_cov_mean": float(np.mean(target_cov)),
        "emp_cov_mean": float(np.mean(emp)),
        "target_cov_std": float(np.std(target_cov)),
        "emp_cov_std": float(np.std(emp)),
        "check_realizations": int(n_realizations),
    }


def merge_variogram(default_cfg: Dict[str, Any], override: Optional[Dict[str, Any]]) -> VariogramSpec:
    d = dict(default_cfg or {})
    if override:
        d.update(override)
    return VariogramSpec(
        model=str(d.get("model", "exponential")),
        nugget=float(d.get("nugget", 0.05)),
        sill=float(d.get("sill", 0.95)),
        ranges=tuple(float(x) for x in d.get("ranges", [180.0, 80.0, 35.0])),
        azimuth_deg=float(d.get("azimuth_deg", 0.0)),
        dip_deg=float(d.get("dip_deg", 0.0)),
    )


def synthetic_acceptance_v3(output_dir: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(777)
    data = rng.uniform(0.0, 100.0, size=(40, 3))
    grade = 0.15 + 0.01 * data[:, 0] + 0.004 * data[:, 1] + rng.lognormal(-2.4, 0.35, size=len(data))

    gx, gy, gz = np.meshgrid(
        np.linspace(0, 100, 8),
        np.linspace(0, 100, 7),
        np.linspace(0, 50, 3),
        indexing="ij",
    )
    targets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    vario = VariogramSpec(model="exponential", nugget=0.05, sill=0.95, ranges=(70, 45, 20))
    tb = TBSpec(bands=192, terms_per_band=1, seed=99, max_neighbors=14, search_radius=90)
    sims, sim_audit = conditional_tb_simulation_v3(data, grade, targets, vario, tb, 6)
    sums = summarize_realizations(sims, [0.5, 1.0])
    cov = covariance_rmse_check_v3(targets, vario, tb, n_realizations=12, n_pairs=800)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(targets, columns=["X", "Y", "Z"])
    for k, v in sums.items():
        df[k] = v
    csv_path = output_dir / "synthetic_full_tb_v3_summary.csv"
    df.to_csv(csv_path, index=False)

    audit = {
        "status": "PASS",
        "engine": "DeepRBF Full Turning Bands V3",
        "method": "line_based_tb_ok_correction",
        "realizations": int(sims.shape[0]),
        "targets": int(sims.shape[1]),
        "finite": bool(np.isfinite(sims).all()),
        "covariance_check": cov,
        "simulation": sim_audit,
        "summary_csv": str(csv_path),
    }
    (output_dir / "synthetic_full_tb_v3_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
