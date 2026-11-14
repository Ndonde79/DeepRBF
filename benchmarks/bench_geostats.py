# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""DeepRBF 3.3.4 geostatistics benchmark (B7) for the Computers & Geosciences paper.

G1  kriging_gslib: agreement with a direct solve of the same system; speed from 10^4 to 10^6 targets
G2  sgs_gslib, 3-D: 400,000 nodes, nested anisotropic variogram, skewed grades; time per realisation,
    histogram and directional variogram reproduction, single grid vs multiple grid; 3.3.3 run_sgs for comparison
G3  sis_gslib, 3-D: 400,000 nodes; time, proportions, indicator variogram; 3.3.3 run_sis speed on 5,000-20,000 nodes
Every case runs in its own process with a time limit; one JSON line per case in results/bench_geostats.jsonl.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
OUT = RES / "bench_geostats.jsonl"
import deeprbf_recovered as n


def emit(r):
    r["t_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S"); r["native"] = n.__version__
    with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps(r, default=float)[:300], flush=True)


def child(case, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "--case", json.dumps(case)], capture_output=True, text=True, timeout=timeout, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return dict(status="timeout", seconds=round(time.perf_counter() - t0, 1))
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    if p.returncode != 0 or not lines: return dict(status="error", stderr=p.stderr[-800:])
    return json.loads(lines[-1][2:])


def vario():   # nested, anisotropic, rotated: a typical gold model in normal-score units
    return n.GSLIBVariogram(0.15, [n.GSLIBStructure("spherical", 0.55, 25, 15, 5, 30, 20, 0),
                                   n.GSLIBStructure("spherical", 0.30, 80, 50, 15, 30, 20, 0)])


def search():
    return n.SearchEllipsoid(160, 100, 30, 30, 20, 0)


def data(nd, seed):
    rg = np.random.default_rng(seed)
    X = np.c_[rg.uniform(0, 400, nd), rg.uniform(0, 400, nd), rg.uniform(0, 100, nd)]
    return X, np.exp(0.9 * rg.normal(size=nd) - 0.4)


def grid(nx, ny, nz, s):
    return np.stack(np.meshgrid(np.arange(nx) * s + s / 2, np.arange(ny) * s + s / 2, np.arange(nz) * s / 2 + s / 4, indexing="ij"), -1).reshape(-1, 3)


def dir_vario(G, vals, shape, axis_step):   # semivariogram along grid axes, lags 1..L nodes
    A = vals.reshape(shape)
    out = {}
    for ax, name in [(0, "x"), (1, "y"), (2, "z")]:
        g = []
        for L in axis_step:
            if L >= A.shape[ax]: g.append(None); continue
            a = np.take(A, range(0, A.shape[ax] - L), axis=ax); b = np.take(A, range(L, A.shape[ax]), axis=ax)
            g.append(round(float(0.5 * np.mean((a - b) ** 2)), 4))
        out[name] = g
    return out


def case_g1(c):
    X, v = data(2000, 1)
    T = np.random.default_rng(2).uniform([0, 0, 0], [400, 400, 100], (c["targets"], 3))
    t = time.perf_counter()
    r = n.kriging_gslib(X.tolist(), np.log(v).tolist(), T.tolist(), vario(), search(), max_data=24, min_data=1, kriging_type="ordinary")
    secs = time.perf_counter() - t
    out = dict(seconds=secs, targets_per_s=c["targets"] / secs, estimated=int(np.isfinite(r["estimate"]).sum()))
    if c["targets"] <= 10000:   # direct solve of the same system on 200 targets
        V = vario(); lv = np.log(v); errs = []
        for i in range(200):
            q = T[i]
            # same neighbours: the engine's anisotropic search, reproduced with the search rotation
            R = np.asarray(n.gslib_rotation(30, 20, 0, 160, 100, 30)).reshape(3, 3)
            d = np.linalg.norm((X - q) @ R.T, axis=1); nb = np.argsort(d)[:24]; nb = nb[d[nb] <= 1.0]
            P = X[nb]; k = len(nb)
            A = np.ones((k + 1, k + 1)); A[k, k] = 0
            A[:k, :k] = [[V.covariance(a.tolist(), b.tolist()) for b in P] for a in P]
            b = np.r_[[V.covariance(a.tolist(), q.tolist()) for a in P], 1.0]
            w = np.linalg.solve(A, b); errs.append(abs(w[:k] @ lv[nb] - r["estimate"][i]))
        out["max_abs_diff_vs_direct"] = float(np.max(errs))
    return out


def case_g2(c):
    X, v = data(1500, 3)
    G = grid(*c["shape"], 4.0)
    t = time.perf_counter()
    if c["engine"] == "sgs_gslib":
        R = n.sgs_gslib(X.tolist(), v.tolist(), G.tolist(), vario(), search(), n_realizations=c["nreal"], max_data=16, max_nodes=16,
                        kriging_type="simple", normal_score=True, multigrid_levels=c["levels"], seed=11)
    else:   # 3.3.3 run_sgs_with_range: isotropic range, ordinary kriging, standardised values
        R = np.asarray(n.run_sgs_with_range(X.tolist(), v.tolist(), G.tolist(), n_realizations=c["nreal"], max_neighbors=32, min_neighbors=1,
                                            search_radius=80.0, variogram_range=80.0, sill=1.0, nugget=0.15, seed=11))
    secs = time.perf_counter() - t
    q = [0.1, 0.5, 0.9, 0.99]
    ys = np.empty_like(R)
    from scipy.stats import norm
    for r in range(R.shape[1]):
        ys[:, r] = norm.ppf((np.argsort(np.argsort(R[:, r])) + 0.5) / R.shape[0])
    dv = [dir_vario(G, ys[:, r], tuple(c["shape"]), [1, 3, 6, 12]) for r in range(min(R.shape[1], 4))]
    mean_dv = {a: [round(float(np.mean([d[a][i] for d in dv if d[a][i] is not None])), 3) if dv[0][a][i] is not None else None for i in range(4)] for a in "xyz"}
    return dict(nodes=len(G), realisations=R.shape[1], seconds=secs, seconds_per_realisation=secs / R.shape[1],
                data_quantiles=np.round(np.quantile(v, q), 3).tolist(), sim_quantiles=np.round(np.quantile(R, q), 3).tolist(),
                sim_mean=float(R.mean()), data_mean=float(v.mean()), normal_score_variogram_by_axis=mean_dv,
                lags_nodes=[1, 3, 6, 12], node_spacing_m=[4.0, 4.0, 2.0])


def case_g3(c):
    X, v = data(1500, 4)
    cat = (v > np.quantile(v, 0.75)).astype(int)
    G = grid(*c["shape"], 4.0)
    t = time.perf_counter()
    if c["engine"] == "sis_gslib":
        V = n.GSLIBVariogram(0.1, [n.GSLIBStructure("spherical", 0.9, 60, 40, 12, 30, 20, 0)])
        R = n.sis_gslib(X.tolist(), cat.tolist(), G.tolist(), [0, 1], [V, V], search(), n_realizations=c["nreal"], max_data=16, max_nodes=16,
                        multigrid_levels=3, seed=5)
    else:
        R = np.asarray(n.run_sis(X.tolist(), cat.tolist(), G.tolist(), [0, 1], n_realizations=c["nreal"], max_neighbors=16, min_neighbors=1,
                                 search_radius=60.0, sill=1.0, nugget=0.1, seed=5))
    secs = time.perf_counter() - t
    return dict(nodes=len(G), realisations=R.shape[1], seconds=secs, seconds_per_realisation=secs / R.shape[1],
                data_proportion=float(cat.mean()), sim_proportion=float((R == 1).mean()),
                proportion_sd_between_realisations=float((R == 1).mean(0).std()))


CASES = dict(G1=case_g1, G2=case_g2, G3=case_g3)

if __name__ == "__main__":
    if "--case" in sys.argv:
        c = json.loads(sys.argv[sys.argv.index("--case") + 1])
        print("@@" + json.dumps(CASES[c["g"]](c), default=float), flush=True); sys.exit(0)
    t0 = time.perf_counter()
    emit(dict(bench="G0", threads=n.get_openmp_threads(), mkl=bool(getattr(n, "__has_mkl__", False))))
    for T in [10000, 100000, 1000000]:
        emit(dict(bench="G1", targets=T, **child(dict(g="G1", targets=T), 1800)))
    for eng, shape, lev, nreal in [("sgs_gslib", [100, 100, 40], 3, 10), ("sgs_gslib", [100, 100, 40], 0, 10), ("run_sgs_3.3.3", [50, 50, 20], 0, 4)]:
        emit(dict(bench="G2", engine=eng, shape=shape, multigrid_levels=lev, **child(dict(g="G2", engine=eng, shape=shape, levels=lev, nreal=nreal), 3000)))
    for eng, shape, nreal in [("sis_gslib", [100, 100, 40], 10), ("run_sis_3.3.3", [25, 25, 8], 2), ("run_sis_3.3.3", [50, 50, 8], 2), ("sis_gslib", [50, 50, 8], 2)]:
        emit(dict(bench="G3", engine=eng, shape=shape, **child(dict(g="G3", engine=eng, shape=shape, nreal=nreal), 2400)))
    emit(dict(bench="END", total_minutes=round((time.perf_counter() - t0) / 60, 1)))
