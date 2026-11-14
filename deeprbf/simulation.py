# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Spectral turning bands and quadrature-based Gaussian direct block simulation.

Known-mean, stationary, univariate Gaussian algorithms. No nugget/process
white noise or automatic raw-grade block back-transform is assumed.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _points(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        a = np.empty((0, 3))
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{name} must have shape (n, 3) and be finite")
    return a


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _conditioning(x, y, noise):
    x = _points(x, "sample_xyz")
    y = np.asarray(y, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("sample_values must be finite and match sample_xyz")
    noise = np.asarray(noise, dtype=float)
    if noise.ndim == 0:
        noise = np.full(len(x), float(noise))
    if noise.shape != (len(x),) or not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("measurement_variance must be finite, nonnegative and scalar or sample-sized")
    # Collapse repeated noiseless observations, but keep noisy replicates.
    indices, known = [], {}
    for i, point in enumerate(x):
        if noise[i] == 0:
            key = tuple(point)
            if key in known:
                if y[i] != y[known[key]]:
                    raise ValueError("conflicting noiseless observations at identical coordinates")
                continue
            known[key] = i
        indices.append(i)
    return x[indices], y[indices], noise[indices]


def _factor(cov):
    try:
        return cho_factor(cov, lower=True, check_finite=True)
    except np.linalg.LinAlgError as error:
        raise ValueError("conditioning covariance is singular or ill-conditioned; check spacing, covariance and band count") from error


class GaussianCovariance:
    """Spherical, exponential or Gaussian covariance without a nugget.

    h = norm((rotation @ displacement) / ranges). Exponential is exp(-3h),
    Gaussian exp(-3h**2). Rotation rows define principal axes in world space.
    This is an explicit matrix convention, not an azimuth/dip convention.
    """
    def __init__(self, model="gaussian", ranges=(1., 1., 1.), sill=1., rotation=None):
        if model not in ("gaussian", "exponential", "spherical"):
            raise ValueError("unsupported covariance model")
        ranges = np.asarray(ranges, dtype=float)
        if ranges.ndim == 0:
            ranges = np.repeat(ranges, 3)
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if ranges.shape != (3,) or not np.isfinite(ranges).all() or np.any(ranges <= 0):
            raise ValueError("ranges must be three positive finite lengths")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1.):
            raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
        if not np.isfinite(sill) or sill <= 0:
            raise ValueError("sill must be finite and positive")
        self.model, self.ranges, self.sill = model, ranges.copy(), float(sill)
        self.rotation = rotation.copy()
        self.ranges.setflags(write=False)
        self.rotation.setflags(write=False)

    def coordinates(self, points):
        with np.errstate(over="raise", invalid="raise"):
            return (_points(points, "points") @ self.rotation.T) / self.ranges

    def matrix(self, points, other=None):
        a = self.coordinates(points)
        b = a if other is None else self.coordinates(other)
        h = cdist(a, b)
        if self.model == "gaussian":
            correlation = np.exp(-3 * h**2)
        elif self.model == "exponential":
            correlation = np.exp(-3 * h)
        else:
            h = np.minimum(h, 1.)
            correlation = 1 - 1.5*h + .5*h**3
        return self.sill * correlation


class TurningBandsSimulator:
    """Finite spectral bands with Gaussian coefficients and conditional correction.

    Gaussian and exponential spectra are supported. Conditional moments are
    exact for the sampled finite-band covariance, which approximates the
    requested continuous covariance. Increase bands and check convergence.
    Covariance is fixed by spectrum_seed; seed controls realization draws.
    """
    def __init__(self, covariance=None, *, bands=512, spectrum_seed=42, mean=0.):
        self.covariance = GaussianCovariance() if covariance is None else covariance
        if self.covariance.model not in ("gaussian", "exponential"):
            raise ValueError("TB supports Gaussian and exponential spectra; spherical is not approximated silently")
        self.bands = _positive_integer(bands, "bands")
        if not np.isfinite(mean):
            raise ValueError("mean must be finite")
        self.mean = float(mean)
        rng = np.random.default_rng(spectrum_seed)
        # A 3-D isotropic spectral vector is a random direction times its
        # correctly distributed radial frequency, not a scalar normal/Cauchy
        # frequency attached to an independent direction.
        self.frequency = rng.normal(size=(self.bands, 3))
        if self.covariance.model == "gaussian":
            self.frequency *= np.sqrt(6.)
        else:
            denominator = np.abs(rng.normal(size=(self.bands, 1)))
            if np.any(denominator == 0):
                raise ValueError("zero spectral denominator; choose another spectrum_seed")
            self.frequency *= 3. / denominator
        self.frequency.setflags(write=False)

    def features(self, points):
        phase = self.covariance.coordinates(points) @ self.frequency.T
        if not np.isfinite(phase).all():
            raise ValueError("spectral phase overflow")
        return np.sqrt(self.covariance.sill / self.bands) * np.concatenate((np.cos(phase), np.sin(phase)), axis=1)

    def covariance_matrix(self, points, other=None):
        a = self.features(points)
        b = a if other is None else self.features(other)
        return a @ b.T

    def simulate(self, target_xyz, *, sample_xyz=(), sample_values=(),
                 measurement_variance=0., n_realizations=1, seed=42, chunk_size=4096):
        target = _points(target_xyz, "target_xyz")
        count = _positive_integer(n_realizations, "n_realizations")
        chunk = _positive_integer(chunk_size, "chunk_size")
        x, y, noise = _conditioning(sample_xyz, sample_values, measurement_variance)
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(size=(2*self.bands, count))
        if len(x):
            data = self.features(x)
            system = data @ data.T + np.diag(noise)
            factor = _factor(system)
            residual = y[:, None] - self.mean - data @ coefficients
            if np.any(noise):
                residual -= np.sqrt(noise[:, None]) * rng.normal(size=(len(x), count))
            coefficients += data.T @ cho_solve(factor, residual)
        result = np.empty((len(target), count))
        for start in range(0, len(target), chunk):
            result[start:start+chunk] = self.mean + self.features(target[start:start+chunk]) @ coefficients
        # Remove solver roundoff only at exact noiseless observations.
        known = {tuple(point): value for point, value, v in zip(x, y, noise) if v == 0}
        for i, point in enumerate(target):
            if tuple(point) in known:
                result[i] = known[tuple(point)]
        return result


@dataclass
class BlockMoments:
    mean: np.ndarray
    covariance: np.ndarray


class DirectBlockSimulator:
    """Joint simulation of Gaussian block averages using integrated covariances.

    Gauss-Legendre quadrature approximates the support integrals. Blocks are
    axis-aligned in world coordinates; zero side lengths give lower-dimensional
    or point supports. Dense block covariance makes this a moderate-size method.
    """
    def __init__(self, covariance=None, *, mean=0., quadrature_order=3, max_blocks=512):
        self.covariance = GaussianCovariance() if covariance is None else covariance
        self.order = _positive_integer(quadrature_order, "quadrature_order")
        if self.order > 8:
            raise ValueError("quadrature_order above 8 is not supported")
        self.max_blocks = _positive_integer(max_blocks, "max_blocks")
        if not np.isfinite(mean):
            raise ValueError("mean must be finite")
        self.mean = float(mean)

    def _supports(self, centers, sizes):
        centers = _points(centers, "block_centers")
        if len(centers) > self.max_blocks:
            raise ValueError("block count exceeds max_blocks; dense DBS is not a large-grid engine")
        sizes = np.asarray(sizes, dtype=float)
        if sizes.shape == (3,):
            sizes = np.broadcast_to(sizes, centers.shape)
        if sizes.shape != centers.shape or not np.isfinite(sizes).all() or np.any(sizes < 0):
            raise ValueError("block_sizes must be nonnegative and shape (3,) or (n,3)")
        nodes, weights = np.polynomial.legendre.leggauss(self.order)
        supports = []
        for center, size in zip(centers, sizes):
            axes = [(nodes*.5*length, weights*.5) if length else (np.array([0.]), np.array([1.])) for length in size]
            indices = list(product(*(range(len(axis[0])) for axis in axes)))
            points = np.array([[center[k]+axes[k][0][idx[k]] for k in range(3)] for idx in indices])
            mass = np.array([np.prod([axes[k][1][idx[k]] for k in range(3)]) for idx in indices])
            supports.append((points, mass))
        return centers, sizes, supports

    def conditional_moments(self, block_centers, block_sizes, *, sample_xyz=(), sample_values=(), measurement_variance=0.):
        centers, sizes, supports = self._supports(block_centers, block_sizes)
        x, y, noise = _conditioning(sample_xyz, sample_values, measurement_variance)
        nb = len(supports)
        cbb = np.empty((nb, nb))
        cbd = np.empty((nb, len(x)))
        for i, (a, wa) in enumerate(supports):
            if len(x):
                cbd[i] = wa @ self.covariance.matrix(a, x)
            for j in range(i+1):
                b, wb = supports[j]
                cbb[i, j] = cbb[j, i] = wa @ self.covariance.matrix(a, b) @ wb
        mean = np.full(nb, self.mean)
        if len(x):
            factor = _factor(self.covariance.matrix(x) + np.diag(noise))
            mean += cbd @ cho_solve(factor, y-self.mean)
            cbb -= cbd @ cho_solve(factor, cbd.T)
        cbb = .5*(cbb+cbb.T)
        known = {tuple(point): value for point, value, v in zip(x, y, noise) if v == 0}
        for i, (center, size) in enumerate(zip(centers, sizes)):
            if np.all(size == 0) and tuple(center) in known:
                mean[i] = known[tuple(center)]
                cbb[i, :] = cbb[:, i] = 0.
        return BlockMoments(mean, cbb)

    def simulate(self, block_centers, block_sizes, *, sample_xyz=(), sample_values=(), measurement_variance=0., n_realizations=1, seed=42):
        count = _positive_integer(n_realizations, "n_realizations")
        moments = self.conditional_moments(block_centers, block_sizes, sample_xyz=sample_xyz, sample_values=sample_values, measurement_variance=measurement_variance)
        if len(moments.mean) == 0:
            return np.empty((0, count))
        values, vectors = np.linalg.eigh(moments.covariance)
        tolerance = 1e-10*self.covariance.sill
        if values.min() < -tolerance:
            raise ValueError("conditional block covariance is not positive semidefinite")
        root = vectors * np.sqrt(np.maximum(values, 0.))
        rng = np.random.default_rng(seed)
        return moments.mean[:, None] + root @ rng.normal(size=(len(values), count))


def simulate_tb(sample_xyz, sample_values, target_xyz, *, covariance=None, bands=512, spectrum_seed=42, mean=0., n_realizations=1, seed=42, chunk_size=4096, measurement_variance=0.):
    """Convenience entry point for conditional spectral turning bands."""
    return TurningBandsSimulator(covariance, bands=bands, spectrum_seed=spectrum_seed, mean=mean).simulate(target_xyz, sample_xyz=sample_xyz, sample_values=sample_values, n_realizations=n_realizations, seed=seed, chunk_size=chunk_size, measurement_variance=measurement_variance)


def simulate_dbs(sample_xyz, sample_values, block_centers, block_sizes, *, covariance=None, mean=0., quadrature_order=3, max_blocks=512, n_realizations=1, seed=42, measurement_variance=0.):
    """Convenience entry point for Gaussian direct block simulation."""
    return DirectBlockSimulator(covariance, mean=mean, quadrature_order=quadrature_order, max_blocks=max_blocks).simulate(block_centers, block_sizes, sample_xyz=sample_xyz, sample_values=sample_values, n_realizations=n_realizations, seed=seed, measurement_variance=measurement_variance)
