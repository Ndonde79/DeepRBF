# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np

EPS = 1.0e-12
UNKNOWN = -2147483648


# ---------------------------------------------------------------------------
# Configuration / diagnostics
# ---------------------------------------------------------------------------
@dataclass
class SNESIMConfig:
    max_conditioning_nodes: int = 32
    min_replicates: int = 10
    n_multigrids: int = 3
    search_radii: Tuple[float, float, float] = (8.0, 8.0, 4.0)
    search_angles_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    affinity: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    servo_lambda: float = 0.0
    target_proportions: Optional[Mapping[int, float]] = None
    soft_tau: float = 1.0
    seed: int = 31001
    n_realizations: int = 1
    random_path: bool = True
    rotation_bin_deg: float = 5.0
    affinity_bin: float = 0.10
    max_tree_cache: int = 64
    strict_hard_data: bool = True


@dataclass
class SNESIMDiagnostics:
    simulated_nodes: int = 0
    hard_nodes: int = 0
    inactive_nodes: int = 0
    fallback_to_prior: int = 0
    conditioning_queries: int = 0
    backoff_events: int = 0
    total_backoff_nodes: int = 0
    replicate_sum: int = 0
    tree_builds: int = 0
    tree_cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_replicates"] = (
            float(self.replicate_sum) / self.conditioning_queries
            if self.conditioning_queries else 0.0
        )
        d["mean_backoff_nodes"] = (
            float(self.total_backoff_nodes) / self.backoff_events
            if self.backoff_events else 0.0
        )
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= EPS:
        return np.full_like(p, 1.0 / len(p), dtype=float)
    return p / s


def _rotation_matrix(azimuth_deg: float,
                     dip_deg: float,
                     plunge_deg: float) -> np.ndarray:
    a, d, p = np.deg2rad([azimuth_deg, dip_deg, plunge_deg])
    ca, sa = math.cos(a), math.sin(a)
    cd, sd = math.cos(d), math.sin(d)
    cp, sp = math.cos(p), math.sin(p)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cd, -sd], [0.0, sd, cd]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return Rz @ Rx @ Ry


def build_template_offsets(
    ndim: int,
    radii: Sequence[float],
    max_nodes: int,
    angles_deg: Sequence[float] = (0.0, 0.0, 0.0),
    affinity: Sequence[float] = (1.0, 1.0, 1.0),
    grid_stride: int = 1,
) -> np.ndarray:
    if ndim not in (2, 3):
        raise ValueError("SNESIM supports 2D or 3D training images.")
    if len(radii)!=3 or len(affinity)!=3 or len(angles_deg)!=3 or not np.isfinite([*radii,*affinity,*angles_deg]).all():raise ValueError('Finite template triples required')
    if any(v<0 for v in radii) or any(v<=0 for v in affinity) or type(max_nodes) is not int or max_nodes<1 or type(grid_stride) is not int or grid_stride<1:raise ValueError('Invalid template controls')
    rx, ry, rz = [float(x) for x in radii]
    ax, ay, az = [float(x) for x in affinity]
    R = _rotation_matrix(*angles_deg)
    if ndim==2 and not np.allclose(R[2,:2],0,atol=1e-12):raise ValueError('2D templates require rotation within the image plane')
    # Bounding box of the rotated ellipsoid; unrotated bounds omit long-axis nodes.
    extent=np.asarray(affinity)*np.linalg.norm(R@np.diag([rx,ry,rz if ndim==3 else 0]),axis=1)
    xlim,ylim,zlim=[int(math.ceil(v)) for v in extent]
    if ndim==2:zlim=0
    if (2*xlim+1)*(2*ylim+1)*(2*zlim+1)>2000000:raise ValueError('Template candidate workspace exceeded')

    cand = []
    for z in range(-zlim, zlim + 1):
        for y in range(-ylim, ylim + 1):
            for x in range(-xlim, xlim + 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                v = np.array([x / ax, y / ay, z / az], dtype=float)
                local = R.T @ v
                q = ((local[0] / max(rx, EPS)) ** 2 +
                     (local[1] / max(ry, EPS)) ** 2)
                if ndim == 3:
                    q += (local[2] / max(rz, EPS)) ** 2
                if q <= 1.0 + 1e-12:
                    dist = float(np.linalg.norm(v))
                    cand.append((dist, z, y, x))

    cand.sort(key=lambda t: (t[0],
                             abs(t[1]) + abs(t[2]) + abs(t[3]),
                             t[1], t[2], t[3]))
    cand = cand[: max(1, int(max_nodes))]
    if not cand:raise ValueError('Template contains no noncentral integer-grid neighbors')

    if ndim == 2:
        return np.array([[y * grid_stride, x * grid_stride]
                         for _, z, y, x in cand], dtype=int)
    return np.array([[z * grid_stride, y * grid_stride, x * grid_stride]
                     for _, z, y, x in cand], dtype=int)


# ---------------------------------------------------------------------------
# Fast pattern database — vectorized scan + indexed query (np.isin)
# ---------------------------------------------------------------------------
class PatternDatabase:
    """
    Training-image pattern database with hash-indexed query.

    Construction:
        `_scan` builds the pattern matrix and center-category vector with
        vectorized NumPy indexing (no Python-level loop over cells).

    Query:
        An inverted index from (template_position, category) -> np.ndarray
        of sorted pattern indices. Shortest-first intersections and a bounded
        exact-event cache preserve replicate counts and backoff semantics.
    """

    def __init__(self,
                 training_image: np.ndarray,
                 categories: np.ndarray,
                 offsets: np.ndarray) -> None:
        self.ti = np.asarray(training_image, dtype=np.int64)
        self.categories = np.asarray(categories, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=int)
        self.patterns, self.centers = self._scan()
        self._query_cache = {}
        self.query_cache_hits = 0
        self.prior_counts = np.array(
            [(self.centers == c).sum() for c in self.categories], dtype=np.int64
        )
        self.prior = _normalize_prob(self.prior_counts.astype(float))

        # ---- inverted index: (position_idx, category) -> pattern indices ----
        self._idx_by_pos_cat: Dict[Tuple[int, int], np.ndarray] = {}
        n_pat, n_pos = self.patterns.shape
        for pos in range(n_pos):
            col = self.patterns[:, pos]
            for cat in self.categories:
                key = (int(pos), int(cat))
                self._idx_by_pos_cat[key] = np.where(col == cat)[0]

    def _scan(self) -> Tuple[np.ndarray, np.ndarray]:
        ti = self.ti
        shape = np.array(ti.shape, dtype=int)
        offsets = self.offsets
        n_off = int(offsets.shape[0])

        if ti.ndim not in (2, 3):
            raise ValueError("training_image must be 2D or 3D")
        if offsets.ndim != 2 or offsets.shape[1] != ti.ndim:
            raise ValueError("offsets dimensionality mismatch")

        was_2d = (ti.ndim == 2)
        if was_2d:
            ti3 = ti[None, :, :]
            offsets3 = np.concatenate(
                [np.zeros((n_off, 1), dtype=int), offsets], axis=1
            )
            shape3 = np.array(ti3.shape, dtype=int)
        else:
            ti3 = ti
            offsets3 = offsets
            shape3 = shape

        min_off = offsets3.min(axis=0)
        max_off = offsets3.max(axis=0)

        lo = np.maximum(0, -min_off)
        hi = np.minimum(shape3 - 1, shape3 - 1 - max_off)
        if np.any(lo > hi):
            raise ValueError(
                "Training image/template combination produced zero valid "
                "pattern centers (template does not fit)."
            )

        if math.prod((hi-lo+1).tolist())*max(1,n_off)>32000000:raise ValueError('Pattern workspace exceeds 32 million entries')
        ranges = [np.arange(int(lo[d]), int(hi[d]) + 1) for d in range(3)]
        ZZ, YY, XX = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
        centers = np.stack([ZZ.ravel(), YY.ravel(), XX.ravel()], axis=1)

        if centers.shape[0]*max(1,n_off)>32000000:raise ValueError('Pattern workspace exceeds 32 million entries')
        pattern_vals = np.empty((centers.shape[0], n_off), dtype=np.int64)
        for j in range(n_off):
            idx = centers + offsets3[j]
            pattern_vals[:, j] = ti3[idx[:, 0], idx[:, 1], idx[:, 2]]

        center_vals = ti3[centers[:, 0], centers[:, 1], centers[:, 2]]

        cats = self.categories
        valid_center = np.isin(center_vals, cats)
        valid_pattern = ~np.any(pattern_vals == UNKNOWN, axis=1)
        valid = valid_center & valid_pattern

        pattern_vals = pattern_vals[valid]
        center_vals = center_vals[valid]

        if pattern_vals.shape[0] == 0:
            raise ValueError(
                "Training image/template combination produced zero valid "
                "pattern centers."
            )
        return pattern_vals, center_vals

    def query(self,
              informed_positions: np.ndarray,
              informed_values: np.ndarray,
              min_replicates: int):
        if len(informed_positions) == 0:
            return self.prior.copy(), int(self.patterns.shape[0]), 0

        pos = np.asarray(informed_positions, dtype=int)
        vals = np.asarray(informed_values, dtype=np.int64)
        key = (int(min_replicates), tuple(pos), tuple(vals))
        cached = self._query_cache.get(key)
        if cached is not None:
            self.query_cache_hits += 1
            return cached[0].copy(), cached[1], cached[2]
        idx_lookup = self._idx_by_pos_cat
        cats = self.categories

        # Drop the furthest conditioning nodes until enough replicates.
        for keep in range(len(pos), -1, -1):
            if keep == 0:
                return self.prior.copy(), int(self.patterns.shape[0]), len(pos)

            candidates = None
            ok = True
            lists = [idx_lookup.get((int(pos[k]), int(vals[k]))) for k in range(keep)]
            lists.sort(key=lambda v: 0 if v is None else len(v))
            for idx in lists:
                if idx is None or idx.size == 0:
                    ok = False
                    break
                if candidates is None:
                    candidates = idx
                else:
                    # Both inverted lists contain sorted unique pattern indices.
                    candidates = np.intersect1d(candidates, idx, assume_unique=True)
                    if candidates.size == 0:
                        ok = False
                        break

            if not ok or candidates is None:
                continue

            nrep = int(candidates.size)
            if nrep >= min_replicates:
                center_vals = self.centers[candidates]
                counts = np.array(
                    [(center_vals == c).sum() for c in cats],
                    dtype=float,
                )
                probability = _normalize_prob(counts)
                if len(self._query_cache) >= 4096:
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[key] = (probability.copy(), nrep, len(pos)-keep)
                return probability, nrep, len(pos) - keep

        return self.prior.copy(), int(self.patterns.shape[0]), len(pos)


# ---------------------------------------------------------------------------
# Category / prior helpers
# ---------------------------------------------------------------------------
def _prepare_categories(training_image: np.ndarray,
                        categories: Optional[Sequence[int]]) -> np.ndarray:
    if categories is None:
        vals = np.unique(training_image)
        vals = vals[vals != UNKNOWN]
        return vals.astype(np.int64)
    return np.asarray(categories, dtype=np.int64)


def _target_vector(categories: np.ndarray,
                   target: Optional[Mapping[int, float]],
                   prior: np.ndarray) -> np.ndarray:
    if target is None:
        return prior.copy()
    p = np.array([float(target.get(int(c), 0.0)) for c in categories],
                 dtype=float)
    return _normalize_prob(p)


def _apply_target_prior(p: np.ndarray,
                        training_prior: np.ndarray,
                        target_prior: np.ndarray) -> np.ndarray:
    ratio = target_prior / np.maximum(training_prior, EPS)
    return _normalize_prob(p * ratio)


def _apply_servo(p: np.ndarray,
                 current: np.ndarray,
                 target: np.ndarray,
                 servo_lambda: float) -> np.ndarray:
    if servo_lambda <= 0.0:
        return p
    if not (0.0 <= servo_lambda < 1.0):
        raise ValueError("servo_lambda must be in [0,1).")
    mu = servo_lambda / max(1.0 - servo_lambda, EPS)
    corrected = p + mu * (target - current)
    corrected = np.maximum(corrected, 0.0)
    return _normalize_prob(corrected)


def _apply_soft(p: np.ndarray,
                soft: Optional[np.ndarray],
                training_prior: np.ndarray,
                tau: float) -> np.ndarray:
    if soft is None:
        return p
    soft = _normalize_prob(np.asarray(soft, dtype=float))
    ratio = np.maximum(soft, EPS) / np.maximum(training_prior, EPS)
    return _normalize_prob(p * np.power(ratio, max(float(tau), 0.0)))


def _quantize_tuple(values: Sequence[float], step: float):
    if step <= 0:
        return tuple(float(v) for v in values)
    return tuple(round(float(v) / step) * step for v in values)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------
class SNESIMMineralSimulator:
    def __init__(self,
                 training_image: np.ndarray,
                 config: Optional[SNESIMConfig] = None,
                 categories: Optional[Sequence[int]] = None) -> None:
        ti = np.asarray(training_image)
        if ti.ndim not in (2, 3):
            raise ValueError("training_image must be 2D or 3D.")
        if not np.issubdtype(ti.dtype,np.integer) or ti.size==0 or ti.size>32000000:raise ValueError('Nonempty integer training image within workspace limit required')
        if categories is not None:
            supplied=np.asarray(categories)
            if supplied.ndim!=1 or not np.issubdtype(supplied.dtype,np.integer) or len(np.unique(supplied))!=len(supplied) or UNKNOWN in supplied:raise ValueError('Unique integer categories required')
        self.training_image = ti.astype(np.int64, copy=True)
        self.config = config or SNESIMConfig()
        for key in ('max_conditioning_nodes','min_replicates','n_multigrids','n_realizations','max_tree_cache'):
            value=getattr(self.config,key)
            if type(value) is not int or value<1:raise ValueError(key+' must be a positive integer')
        if self.config.n_multigrids>12 or self.config.max_conditioning_nodes>128:raise ValueError('Template or multigrid limit exceeded')
        if not np.isfinite([self.config.servo_lambda,self.config.soft_tau]).all() or not 0<=self.config.servo_lambda<1 or self.config.soft_tau<0:raise ValueError('Invalid servo or soft conditioning weight')
        self.categories = _prepare_categories(self.training_image, categories)
        if len(self.categories) < 2:
            raise ValueError("SNESIM requires at least two categories.")
        if not np.isin(ti, np.r_[self.categories,UNKNOWN]).all():raise ValueError('Undeclared training image category')
        if self.config.target_proportions is not None:
            prior=self.config.target_proportions
            if set(prior)!=set(self.categories.tolist()) or not np.isfinite(list(prior.values())).all() or min(prior.values())<0 or not np.isclose(sum(prior.values()),1.):raise ValueError('Target proportions must cover the catalogue and sum to one')
        self.cat_to_idx = {int(c): i for i, c in enumerate(self.categories)}
        self._tree_cache: Dict[Any, PatternDatabase] = {}

    def _node_orientation(self,
                          idx: Tuple[int, ...],
                          rotation_map: Optional[np.ndarray],
                          affinity_map: Optional[np.ndarray]):
        angles = self.config.search_angles_deg
        affinity = self.config.affinity

        if rotation_map is not None:
            rv = np.asarray(rotation_map[idx])
            if rv.ndim == 0:
                angles = (float(rv), angles[1], angles[2])
            else:
                vals = list(np.ravel(rv).astype(float))
                while len(vals) < 3:
                    vals.append(0.0)
                angles = tuple(vals[:3])

        if affinity_map is not None:
            av = np.asarray(affinity_map[idx])
            if av.ndim == 0:
                affinity = (float(av), 1.0, 1.0)
            else:
                vals = list(np.ravel(av).astype(float))
                while len(vals) < 3:
                    vals.append(1.0)
                affinity = tuple(max(v, EPS) for v in vals[:3])

        angles = _quantize_tuple(angles, self.config.rotation_bin_deg)
        affinity = _quantize_tuple(affinity, self.config.affinity_bin)
        return angles, affinity

    def _db_for(self, angles, affinity, stride,
                diagnostics: SNESIMDiagnostics):
        key = (angles, affinity, int(stride),
               int(self.config.max_conditioning_nodes))
        if key in self._tree_cache:
            diagnostics.tree_cache_hits += 1
            return self._tree_cache[key]

        offsets = build_template_offsets(
            self.training_image.ndim,
            self.config.search_radii,
            self.config.max_conditioning_nodes,
            angles,
            affinity,
            grid_stride=stride,
        )
        db = self._make_pattern_database(offsets)
        diagnostics.tree_builds += 1
        if len(self._tree_cache) >= self.config.max_tree_cache:
            first = next(iter(self._tree_cache))
            del self._tree_cache[first]
        self._tree_cache[key] = db
        return db

    def _make_pattern_database(self, offsets):
        return PatternDatabase(self.training_image, self.categories, offsets)

    def _extract_conditioning(self, sim, idx, offsets):
        center = np.array(idx, dtype=int)
        shape = np.array(sim.shape, dtype=int)
        positions, values = [], []
        for j, off in enumerate(offsets):
            pt = center + off
            if np.any(pt < 0) or np.any(pt >= shape):
                continue
            v = int(sim[tuple(pt)])
            if v != UNKNOWN:
                positions.append(j)
                values.append(v)
        return (np.asarray(positions, dtype=int),
                np.asarray(values, dtype=np.int64))

    def simulate(self,
                 simulation_shape: Sequence[int],
                 hard_data: Optional[np.ndarray] = None,
                 simulation_mask: Optional[np.ndarray] = None,
                 soft_probabilities: Optional[np.ndarray] = None,
                 rotation_map: Optional[np.ndarray] = None,
                 affinity_map: Optional[np.ndarray] = None,
                 n_realizations: Optional[int] = None,
                 seed: Optional[int] = None,
                 return_diagnostics: bool = True):
        if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<1 for v in simulation_shape):raise ValueError('Positive integer simulation dimensions required')
        shape = tuple(int(x) for x in simulation_shape)
        if len(shape) != self.training_image.ndim:
            raise ValueError(
                "simulation_shape dimensionality must match training_image.")

        nr = self.config.n_realizations if n_realizations is None else n_realizations
        base_seed = self.config.seed if seed is None else seed
        if type(nr) is not int or nr<1 or nr*math.prod(shape)>32000000:raise ValueError('Invalid or excessive realization count')
        if type(base_seed) is not int or base_seed<0:raise ValueError('Nonnegative integer seed required')

        if hard_data is not None:
            hard = np.asarray(hard_data)
            if not np.issubdtype(hard.dtype,np.integer):raise ValueError('Hard categories must be integer codes')
            if hard.shape != shape:
                raise ValueError("hard_data shape mismatch.")
            hard = hard.astype(np.int64, copy=False)
        else:
            hard = np.full(shape, UNKNOWN, dtype=np.int64)

        if simulation_mask is None:
            mask = np.ones(shape, dtype=bool)
        else:
            mask = np.asarray(simulation_mask, dtype=bool)
            if mask.shape != shape:
                raise ValueError("simulation_mask shape mismatch.")

        if soft_probabilities is not None:
            softp = np.asarray(soft_probabilities, dtype=float)
            if softp.shape != shape + (len(self.categories),):
                raise ValueError(
                    f"soft_probabilities must have shape "
                    f"{shape + (len(self.categories),)}")
        else:
            softp = None
        if softp is not None and (not np.isfinite(softp).all() or (softp<0).any() or not np.allclose(softp[mask].sum(-1),1.,atol=1e-8)):raise ValueError('Active soft probabilities must be finite, nonnegative and sum to one')
        for mapping,name,positive in [(rotation_map,'rotation_map',False),(affinity_map,'affinity_map',True)]:
            if mapping is not None and (not np.isfinite(mapping).all() or (positive and (np.asarray(mapping)<=0).any())):raise ValueError('Invalid '+name)

        if (rotation_map is not None and
                np.asarray(rotation_map).shape[:len(shape)] != shape):
            raise ValueError(
                "rotation_map leading dimensions must equal simulation_shape.")
        if (affinity_map is not None and
                np.asarray(affinity_map).shape[:len(shape)] != shape):
            raise ValueError(
                "affinity_map leading dimensions must equal simulation_shape.")

        hv = np.unique(hard[hard != UNKNOWN])
        invalid = [int(v) for v in hv if int(v) not in self.cat_to_idx]
        if invalid:
            raise ValueError(
                "Hard data contain categories absent from the training "
                f"image / categories list: {invalid}")

        realizations = np.full((nr,) + shape, UNKNOWN, dtype=np.int64)
        diagnostics_all = []

        for r in range(nr):
            rng = np.random.default_rng(base_seed + 104729 * r)
            sim = np.full(shape, UNKNOWN, dtype=np.int64)
            hard_mask = (hard != UNKNOWN) & mask
            sim[hard_mask] = hard[hard_mask]

            diag = SNESIMDiagnostics(
                hard_nodes=int(hard_mask.sum()),
                inactive_nodes=int((~mask).sum()),
            )

            base_db = self._db_for(
                _quantize_tuple(self.config.search_angles_deg,
                                self.config.rotation_bin_deg),
                _quantize_tuple(self.config.affinity,
                                self.config.affinity_bin),
                1,
                diag,
            )
            train_prior = base_db.prior
            target_prior = _target_vector(self.categories,
                                          self.config.target_proportions,
                                          train_prior)

            counts = np.array(
                [int(np.sum(sim[hard_mask] == c)) for c in self.categories],
                dtype=float)
            informed_total = float(counts.sum())

            levels = list(range(max(1, self.config.n_multigrids) - 1, -1, -1))
            for lev in levels:
                stride = 2 ** lev
                nodes = []
                for idx in np.ndindex(*shape):
                    if not mask[idx] or sim[idx] != UNKNOWN:
                        continue
                    if all((idx[d] % stride) == 0
                           for d in range(len(shape))):
                        nodes.append(idx)

                if self.config.random_path:
                    rng.shuffle(nodes)

                for idx in nodes:
                    angles, affinity = self._node_orientation(
                        idx, rotation_map, affinity_map)
                    db = self._db_for(angles, affinity, stride, diag)
                    positions, values = self._extract_conditioning(
                        sim, idx, db.offsets)
                    p, nrep, dropped = db.query(
                        positions, values,
                        max(1, int(self.config.min_replicates)))
                    diag.conditioning_queries += 1
                    diag.replicate_sum += int(nrep)
                    if dropped:
                        diag.backoff_events += 1
                        diag.total_backoff_nodes += int(dropped)
                    if len(positions) == 0:
                        diag.fallback_to_prior += 1

                    p = _apply_target_prior(p, db.prior, target_prior)
                    current = _normalize_prob(
                        counts if informed_total > 0 else target_prior)
                    p = _apply_servo(p, current, target_prior,
                                     float(self.config.servo_lambda))
                    local_soft = softp[idx] if softp is not None else None
                    p = _apply_soft(p, local_soft, db.prior,
                                    self.config.soft_tau)

                    chosen_i = int(rng.choice(np.arange(len(self.categories)),
                                              p=p))
                    chosen = int(self.categories[chosen_i])
                    sim[idx] = chosen
                    counts[chosen_i] += 1.0
                    informed_total += 1.0
                    diag.simulated_nodes += 1

            remaining = np.argwhere(mask & (sim == UNKNOWN))
            for arr_idx in remaining:
                idx = tuple(int(x) for x in arr_idx)
                p = target_prior.copy()
                local_soft = softp[idx] if softp is not None else None
                p = _apply_soft(p, local_soft, train_prior,
                                self.config.soft_tau)
                chosen_i = int(rng.choice(np.arange(len(self.categories)),
                                          p=p))
                sim[idx] = int(self.categories[chosen_i])
                diag.simulated_nodes += 1

            sim[hard_mask] = hard[hard_mask]
            realizations[r] = sim

            d = diag.to_dict()
            active = sim[mask]
            d["realization"] = r
            d["category_proportions"] = {
                int(c): float(np.mean(active == c)) for c in self.categories
            }
            diagnostics_all.append(d)

        if return_diagnostics:
            return realizations, diagnostics_all
        return realizations


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------
def simulate_snesim(
    training_image: np.ndarray,
    simulation_shape: Sequence[int],
    *,
    config: Optional[SNESIMConfig] = None,
    categories: Optional[Sequence[int]] = None,
    hard_data: Optional[np.ndarray] = None,
    simulation_mask: Optional[np.ndarray] = None,
    soft_probabilities: Optional[np.ndarray] = None,
    rotation_map: Optional[np.ndarray] = None,
    affinity_map: Optional[np.ndarray] = None,
    n_realizations: Optional[int] = None,
    seed: Optional[int] = None,
    return_diagnostics: bool = True,
    backend: str = 'python',
):
    if backend not in ('python','native'):raise ValueError('backend must be python or native')
    if backend=='native':
        return _simulate_native(training_image,simulation_shape,config,categories,hard_data,simulation_mask,
            soft_probabilities,rotation_map,affinity_map,n_realizations,seed,return_diagnostics)
    engine = SNESIMMineralSimulator(training_image,
                                    config=config,
                                    categories=categories)
    return engine.simulate(
        simulation_shape,
        hard_data=hard_data,
        simulation_mask=simulation_mask,
        soft_probabilities=soft_probabilities,
        rotation_map=rotation_map,
        affinity_map=affinity_map,
        n_realizations=n_realizations,
        seed=seed,
        return_diagnostics=return_diagnostics,
    )


def _simulate_native(ti,shape,config,categories,hard,mask,soft,rotation_map,affinity_map,n_realizations,seed,return_diagnostics):
    """Explicit native static-template path; unsupported features fail visibly."""
    from deeprbf_recovered import SnesimNativeConfig,snesim_native_static
    cfg=SNESIMConfig() if config is None else config
    if rotation_map is not None or affinity_map is not None:
        raise ValueError('Native static SNESIM does not support varying rotation/affinity maps; use backend=python')
    if tuple(cfg.search_angles_deg)!=(0.,0.,0.) or tuple(cfg.affinity)!=(1.,1.,1.) or not cfg.random_path:
        raise ValueError('Native static SNESIM requires unrotated unit-affinity random-path templates')
    ti=np.asarray(ti);shape=tuple(shape)
    if not np.issubdtype(ti.dtype,np.integer) or ti.ndim not in (2,3) or len(shape)!=ti.ndim:
        raise ValueError('Integer 2D/3D TI and matching simulation shape required')
    if any(not isinstance(v,(int,np.integer)) or v<1 for v in shape) or math.prod(shape)>32000000:
        raise ValueError('Invalid or excessive SNESIM shape')
    if np.any(ti<np.iinfo(np.int32).min) or np.any(ti>np.iinfo(np.int32).max):raise ValueError('Native category codes must fit int32')
    cats=np.unique(ti[ti!=UNKNOWN]) if categories is None else np.asarray(categories)
    if not np.issubdtype(cats.dtype,np.integer) or np.any(cats<np.iinfo(np.int32).min) or np.any(cats>np.iinfo(np.int32).max):raise ValueError('Native category codes must fit int32')
    hard=np.full(shape,UNKNOWN,np.int32) if hard is None else np.asarray(hard)
    if hard.shape!=shape or not np.issubdtype(hard.dtype,np.integer) or np.any(hard<np.iinfo(np.int32).min) or np.any(hard>np.iinfo(np.int32).max):raise ValueError('Invalid native hard data')
    mask=np.ones(shape,bool) if mask is None else np.asarray(mask,bool)
    if mask.shape!=shape:raise ValueError('Simulation mask shape mismatch')
    nr=cfg.n_realizations if n_realizations is None else n_realizations
    base_seed=cfg.seed if seed is None else seed
    if not isinstance(nr,int) or nr<1 or nr*math.prod(shape)>32000000:raise ValueError('Invalid/excessive SNESIM realizations')
    if not isinstance(base_seed,int) or base_seed<0:raise ValueError('Nonnegative integer seed required')
    native=SnesimNativeConfig();native.max_conditioning_nodes=cfg.max_conditioning_nodes
    native.min_replicates=cfg.min_replicates;native.n_multigrids=cfg.n_multigrids
    native.radius_x,native.radius_y,native.radius_z=cfg.search_radii;native.servo_lambda=cfg.servo_lambda
    proportions=[] if cfg.target_proportions is None else [cfg.target_proportions.get(int(c),0.) for c in cats]
    output=[];diagnostics=[];known=(hard!=UNKNOWN)&mask
    for realization in range(nr):
        native.seed=base_seed+104729*realization
        if soft is not None and np.asarray(soft).shape!=shape+(len(cats),):raise ValueError('Soft probability shape mismatch')
        result=snesim_native_static(ti,hard,cats,proportions,native,mask.astype(int).ravel(),
                                   [] if soft is None else np.asarray(soft,float).ravel(),cfg.soft_tau)
        values=np.asarray(result['values'])
        if not np.array_equal(values[known],hard[known]) or not np.isin(values[mask],cats).all() or np.any(values[~mask]!=UNKNOWN):
            raise RuntimeError('Native SNESIM conditioning/category/mask audit failed')
        output.append(values);diag=dict(result['diagnostics'],backend='native_cpp_indexed_static',hard_mismatches=0,
            hard_nodes=int(known.sum()),inactive_nodes=int((~mask).sum()),realization=realization)
        diagnostics.append(diag)
    sims=np.stack(output)
    return (sims,diagnostics) if return_diagnostics else sims


# ---------------------------------------------------------------------------
# Mineral VMS preset
# ---------------------------------------------------------------------------
def mineral_vms_preset(seed: int = 31001,
                       n_realizations: int = 20) -> SNESIMConfig:
    return SNESIMConfig(
        max_conditioning_nodes=36,
        min_replicates=12,
        n_multigrids=4,
        search_radii=(12.0, 8.0, 5.0),
        search_angles_deg=(0.0, 0.0, 0.0),
        affinity=(1.0, 0.65, 0.40),
        servo_lambda=0.10,
        seed=seed,
        n_realizations=n_realizations,
    )


def status():
    return {
        "module": "DeepRBF Stage 11 SNESIM",
        "algorithm": "multiple-point categorical sequential simulation",
        "mineral_deposit_oriented": True,
        "dimensions": [2, 3],
        "features": [
            "training-image multiple-point patterns",
            "conditional pattern probabilities",
            "minimum-replicate backoff",
            "multi-grid",
            "hard conditioning",
            "target proportions",
            "servosystem",
            "soft conditioning",
            "rotation maps",
            "affinity maps",
            "simulation masks",
        ],
        "continuous_grade_simulation": False,
        "recommended_grade_followup": ["SGS", "Turning Bands"],
    }
