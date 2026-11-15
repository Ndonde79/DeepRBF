# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""DeepRBF 3.3.3 benchmark suite for the Computers & Geosciences software paper.

Every result is appended as one JSON line to results/bench_results.jsonl, so a partial run is usable.
All random inputs are seeded. Heavy cases run in a child process with a time limit, so one slow case
cannot stop the suite; a timed-out case is recorded with status "timeout".

B0  environment and build flags
B1  scattered-data interpolation accuracy: 3-D Franke function, kernels x N
B2  solver scaling: dense direct, GMRES + two-level Schwarz, sparse Cholesky/Schur (spherical)
B3  drift: extrapolation of an indicator field beyond the data with none / constant / linear drift
B4  prediction backends: exact direct sum, compact neighbour sum, ScalFMM (if built)
B5  synthetic orebody recovery: two parallel lenses, drill spacing x nugget, anisotropic indicator RBF vs nearest neighbour
B6  Kilamongo South public case: the three shell methods of deeprbf.workflow.shell

Usage:  python bench_deeprbf.py            (full suite)
        python bench_deeprbf.py B1 B5      (selected benchmarks)
        python bench_deeprbf.py --quick    (small sizes, for a smoke test)
"""
import json, os, platform, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
OUT = RES / os.environ.get("BENCH_OUT", "bench_results.jsonl")
QUICK = "--quick" in sys.argv
ONLY = {a.upper() for a in sys.argv[1:] if not a.startswith("--")}
KILAMONGO = Path(os.environ.get("KILAMONGO_DIR", HERE.parent / "examples" / "kilamongo"))

import deeprbf_recovered as n

M = n.NativeExactRBFImplicitModel


def emit(rec):
    rec["t_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["quick"] = QUICK
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float)[:240], flush=True)


def want(tag):
    return not ONLY or tag in ONLY


def reset():
    M.set_sparse_solver(False)
    M.force_iterative_solver(False)
    M.set_dense_direct_limit(8000)
    M.set_predict_backend("auto")


def child(case, timeout):
    """Run one case in a fresh process; return its result dict, or a timeout/error record."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "--case", json.dumps(case)] + (["--quick"] if QUICK else []),
                           capture_output=True, text=True, timeout=timeout, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return dict(status="timeout", seconds=round(time.perf_counter() - t0, 1), timeout=timeout)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    if p.returncode != 0 or not lines:
        return dict(status="error", returncode=p.returncode, stderr=p.stderr[-800:])
    return json.loads(lines[-1][2:])


# ----------------------------------------------------------------------------------------- shared helpers
def fit(kernel, eps, reg, x, y, drift=None):
    m = M(kernel, eps, reg)
    if drift:
        m.set_drift(drift)
    t = time.perf_counter()
    m.fit(np.asarray(x).tolist(), np.asarray(y).tolist())
    return m, time.perf_counter() - t


def predict(m, q):
    t = time.perf_counter()
    v = np.asarray(m.predict(np.asarray(q).tolist()))
    return v, time.perf_counter() - t


def franke3(p):
    x, y, z = (9 * p[:, 0] - 2), (9 * p[:, 1] - 2), (9 * p[:, 2] - 2)
    return (0.75 * np.exp(-(x ** 2 + y ** 2 + z ** 2) / 4) + 0.75 * np.exp(-((9 * p[:, 0] + 1) ** 2) / 49 - (9 * p[:, 1] + 1) / 10 - (9 * p[:, 2] + 1) / 10)
            + 0.5 * np.exp(-((x - 5) ** 2 + (y - 1) ** 2 + (z - 3) ** 2) / 4) - 0.2 * np.exp(-((x - 2) ** 2) - (y - 5) ** 2 - (z - 2) ** 2))


# ----------------------------------------------------------------------------------------- B0
def b0():
    info = dict(bench="B0", deeprbf_native=getattr(n, "__version__", "?"), python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(), logical_cpus=os.cpu_count(), openmp_threads=n.get_openmp_threads())
    for k in ("__has_mkl__", "__has_openmp__", "__has_scalfmm__"):
        info[k.strip("_")] = bool(getattr(n, k, False))
    try:
        import deeprbf
        info["deeprbf_python"] = getattr(deeprbf, "__version__", "?")
    except Exception as e:
        info["deeprbf_python"] = f"import failed: {e}"
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        info["cpu"] = platform.processor()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    emit(info)


# ----------------------------------------------------------------------------------------- B1
B1_KERNELS = [("linear", 1.0, 0.0), ("cubic", 1.0, 0.0), ("thin_plate", 1.0, 0.0), ("gaussian", 0.3, 1e-8),
              ("multiquadric", 0.2, 0.0), ("spherical", 0.6, 1e-10), ("exponential", 0.6, 1e-10), ("gen_cauchy", 0.6, 1e-10)]


def case_b1(c):
    rng = np.random.default_rng(c["seed"])
    x = rng.uniform(0, 1, (c["n"], 3))
    y = franke3(x)
    q = np.random.default_rng(99).uniform(0.05, 0.95, (20000, 3))
    m, tf = fit(c["kernel"], c["eps"], c["reg"], x, y)
    v, tp = predict(m, q)
    t = franke3(q)
    err = v - t
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), max_abs=float(np.abs(err).max()), rel_rmse=float(np.sqrt(np.mean(err ** 2)) / t.std()),
                fit_s=tf, predict_s=tp, backend=m.solver_backend(), predict_backend=m.last_predict_backend())


def b1():
    sizes = [250, 1000] if QUICK else [500, 2000, 8000]
    for N in sizes:
        for k, eps, reg in B1_KERNELS:
            c = dict(b="B1", kernel=k, eps=eps, reg=reg, n=N, seed=1)
            emit(dict(bench="B1", kernel=k, epsilon=eps, regularization=reg, n=N, **child(c, 900)))


# ----------------------------------------------------------------------------------------- B2
def case_b2(c):
    reset()
    rng = np.random.default_rng(c["seed"])
    L = (c["n"] / 2000.0) ** (1 / 3) * 10.0                      # constant point density: ~2 points per unit^3
    x = rng.uniform(0, L, (c["n"], 3))
    y = np.sin(0.6 * x[:, 0]) + np.cos(0.4 * x[:, 1]) * np.sin(0.3 * x[:, 2]) + 0.05 * x[:, 0]
    if c["solver"] == "dense":
        M.set_dense_direct_limit(10 ** 7)
    elif c["solver"] == "iterative":
        M.force_iterative_solver(True)
    elif c["solver"] == "sparse":
        M.set_sparse_solver(True)
    m, tf = fit(c["kernel"], c["eps"], c["reg"], x, y)
    q = rng.uniform(0, L, (50000, 3))
    v, tp = predict(m, q[:2000])
    resid = np.asarray(m.predict(x[:2000].tolist())) - y[:2000]
    out = dict(fit_s=tf, predict_2000_s=tp, backend=m.solver_backend(), max_abs_residual_at_data=float(np.abs(resid).max()))
    if c["solver"] == "sparse":
        out["sparse_info"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(m.sparse_solve_info()).items()}
    np.save(RES / f"_b2_{c['solver']}_{c['kernel']}_{c['n']}.npy", v)
    reset()
    return out


def b2():
    sizes = [1000, 3000] if QUICK else [2000, 5000, 10000, 20000]
    plan = [("dense", "linear", 1.0, 0.0), ("iterative", "linear", 1.0, 0.0),
            ("dense", "spherical", 3.0, 1e-6), ("sparse", "spherical", 3.0, 1e-6), ("iterative", "spherical", 3.0, 1e-6)]
    for N in sizes:
        for solver, k, eps, reg in plan:
            if solver == "dense" and N > 12000:
                emit(dict(bench="B2", solver=solver, kernel=k, n=N, status="skipped", reason="dense matrix above 1.1 GB")); continue
            c = dict(b="B2", solver=solver, kernel=k, eps=eps, reg=reg, n=N, seed=2)
            r = child(c, 1200)
            ref = RES / f"_b2_dense_{k}_{N}.npy"
            mine = RES / f"_b2_{solver}_{k}_{N}.npy"
            if solver != "dense" and ref.exists() and mine.exists() and r.get("status") is None:
                a, b = np.load(ref), np.load(mine)
                r["rel_diff_vs_dense"] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))
            emit(dict(bench="B2", solver=solver, kernel=k, epsilon=eps, regularization=reg, n=N, **r))


# ----------------------------------------------------------------------------------------- B3
def case_b3(c):
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, (1500, 3))
    lens = lambda p: np.abs(p[:, 2] - (50 + 0.2 * (p[:, 0] - 50))) < 6                # a 12 m thick dipping lens
    ind = lens(x).astype(float)
    m, tf = fit(c["kernel"], c["eps"], c["reg"], x, ind, drift=c["drift"])
    far = np.c_[rng.uniform(150, 300, (5000, 1)), rng.uniform(0, 100, (5000, 2))]   # 50-200 m beyond the data, in strike
    across = np.c_[rng.uniform(0, 100, (5000, 2)), np.where(rng.random(5000) < 0.5, rng.uniform(150, 300, 5000),
                                                             rng.uniform(-200, -50, 5000))]   # 50-200 m above / below
    v, _ = predict(m, far)
    va, _ = predict(m, across)
    inside = np.c_[rng.uniform(0, 100, (5000, 2)), rng.uniform(0, 100, (5000, 1))]
    vi, _ = predict(m, inside)
    ti = lens(inside)
    out = dict(far_mean=float(v.mean()), far_p99=float(np.quantile(v, 0.99)), far_frac_above_0_5=float((v > 0.5).mean()),
               across_frac_above_0_5=float((va > 0.5).mean()),
               inside_accuracy=float(((vi > 0.5) == ti).mean()), fit_s=tf)
    # 3.4.2: the same data coded as signed distance (distance to the nearest sample of the other class, ore negative);
    # inside where the field is <= 0, so a far field that drifts to or below zero counts as ore.
    from scipy.spatial import cKDTree
    o = ind > 0.5
    sd = cKDTree(x[o]).query(x)[0]                                                  # waste: distance to nearest ore
    sd[o] = -cKDTree(x[~o]).query(x[o])[0]                                         # ore: minus distance to nearest waste
    ms, _ = fit(c["kernel"], c["eps"], c["reg"], x, sd, drift=c["drift"])
    s_far, _ = predict(ms, far)
    s_acr, _ = predict(ms, across)
    s_in, _ = predict(ms, inside)
    out.update(sd_far_frac_inside=float((s_far <= 0).mean()), sd_across_frac_inside=float((s_acr <= 0).mean()),
               sd_inside_accuracy=float(((s_in <= 0) == ti).mean()))
    return out


def b3():
    for k, eps, reg in [("spherical", 25.0, 0.05), ("gen_cauchy", 25.0, 0.05), ("exponential", 25.0, 0.05), ("linear", 1.0, 0.05)]:
        for drift in ["none", "constant", "linear"]:
            if k == "linear" and drift == "none":
                continue
            r = child(dict(b="B3", kernel=k, eps=eps, reg=reg, drift=drift), 600)
            emit(dict(bench="B3", kernel=k, epsilon=eps, regularization=reg, drift=drift, **r))


# ----------------------------------------------------------------------------------------- B4
def case_b4(c):
    reset()
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 100, (c["n"], 3))
    y = np.sin(x[:, 0] / 9) + np.cos(x[:, 1] / 13)
    if c["kernel"] == "spherical":
        M.set_sparse_solver(True)
    m, _ = fit(c["kernel"], c["eps"], c["reg"], x, y)
    M.set_sparse_solver(False)
    q = rng.uniform(0, 100, (c["targets"], 3))
    M.set_predict_backend(c["backend"])
    try:
        v, tp = predict(m, q)
    finally:
        M.set_predict_backend("auto")
    np.save(RES / f"_b4_{c['kernel']}_{c['backend']}.npy", v[:20000])
    return dict(predict_s=tp, used=m.last_predict_backend(), targets_per_s=c["targets"] / tp)


def b4():
    T = 100000 if QUICK else 1000000
    for k, eps, reg, backs in [("gaussian", 8.0, 1e-6, ["direct", "scalfmm"]), ("spherical", 10.0, 1e-6, ["direct", "auto"])]:
        for b in backs:
            if b == "scalfmm" and not getattr(n, "__has_scalfmm__", False):
                emit(dict(bench="B4", kernel=k, backend=b, status="skipped", reason="ScalFMM not in this build")); continue
            r = child(dict(b="B4", kernel=k, eps=eps, reg=reg, n=4000, targets=T, backend=b), 1200)
            ref, mine = RES / f"_b4_{k}_direct.npy", RES / f"_b4_{k}_{b}.npy"
            if b != "direct" and ref.exists() and mine.exists() and r.get("status") is None:
                a, bb = np.load(ref), np.load(mine)
                r["rel_l2_vs_direct"] = float(np.linalg.norm(a - bb) / np.linalg.norm(a))
            emit(dict(bench="B4", kernel=k, epsilon=eps, n=4000, targets=T, backend=b, **r))


# ----------------------------------------------------------------------------------------- B5
DIP, DIPAZ = 50.0, 90.0                                           # both lenses dip 50 deg to the east


def lens_frame(p):
    """Coordinates in the lens frame: (strike, down-dip, across-lens)."""
    d, az = np.radians(DIP), np.radians(DIPAZ)
    nrm = np.array([np.sin(d) * np.sin(az), np.sin(d) * np.cos(az), np.cos(d)])        # pole to the plane
    stk = np.array([np.sin(az - np.pi / 2), np.cos(az - np.pi / 2), 0.0])
    dd = np.cross(nrm, stk)
    q = p - np.array([200.0, 200.0, 150.0])
    return np.c_[q @ stk, q @ dd, q @ nrm]


def truth(p):
    """Two parallel, undulating, pinching lenses, 3-15 m thick, 40 m apart, in a 400 x 400 x 300 m volume."""
    s, u, w = lens_frame(p).T
    out = np.zeros(len(p), bool)
    for off, t0, amp in [(-20.0, 9.0, 6.0), (20.0, 6.0, 3.0)]:
        mid = off + 8 * np.sin(s / 40) + 6 * np.cos(u / 55)
        th = t0 + amp * np.sin(s / 70 + u / 90)
        out |= (np.abs(w - mid) < th / 2) & (np.hypot(s / 170, u / 150) < 1)
    return out


def case_b5(c):
    rng = np.random.default_rng(c["seed"])
    sp = c["spacing"]
    cx, cy = np.meshgrid(np.arange(sp / 2, 400, sp), np.arange(sp / 2, 400, sp))
    cx, cy = cx.ravel() + rng.normal(0, 2, cx.size), cy.ravel() + rng.normal(0, 2, cy.size)
    zs = np.arange(1.0, 300, 2.0)                                 # vertical holes, 2 m composites
    X = np.c_[np.repeat(cx, zs.size), np.repeat(cy, zs.size), np.tile(zs, cx.size)]
    lab = truth(X)
    obs = lab.copy()
    k = int(round(c["nugget"] * lab.sum()))                       # nugget: k ore composites logged as waste and k waste
    if k:                                                         # composites logged as ore (balanced, so the ore
        obs[rng.choice(np.flatnonzero(lab), k, replace=False)] = False   # fraction is unbiased)
        obs[rng.choice(np.flatnonzero(~lab), k, replace=False)] = True
    g = np.stack(np.meshgrid(np.arange(2.5, 400, 5), np.arange(2.5, 400, 5), np.arange(2.5, 300, 5), indexing="ij"), -1).reshape(-1, 3)
    tg = truth(g)
    out = dict(holes=int(cx.size), composites=int(X.shape[0]), ore_fraction=float(obs.mean()))

    def score(pred):
        tp, fp, fn = int((pred & tg).sum()), int((pred & ~tg).sum()), int((~pred & tg).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return dict(f1=round(2 * pr * rc / max(pr + rc, 1e-12), 4), precision=round(pr, 4), recall=round(rc, 4),
                    volume_ratio=round(float(pred.sum() / tg.sum()), 3))

    from scipy.spatial import cKDTree
    _, j = cKDTree(X).query(g)                                    # nearest-neighbour (polygonal) indicator
    out["nearest_neighbour"] = score(obs[j])
    # DeepRBF indicator RBF in the lens frame: spherical covariance, range 2 x spacing along the lenses and
    # 8 m across them (anisotropy by scaling the across-lens axis), constant drift, nugget as regularisation.
    R, r_across = 2.0 * sp, 8.0
    sc = np.array([1.0, 1.0, R / r_across])
    Xf, gf = lens_frame(X) * sc, lens_frame(g) * sc
    reset()
    M.set_sparse_solver(True)
    for name, reg in [("rbf_exact", 1e-6), ("rbf_nugget", 0.5)]:
        t = time.perf_counter()
        m = M("spherical", R, reg)
        m.set_drift("constant")
        m.fit(Xf.tolist(), obs.astype(float).tolist())
        v = np.asarray(m.predict(gf.tolist()))
        out[name] = dict(**score(v > 0.5), seconds=round(time.perf_counter() - t, 1))
        # 3.4.2: iso calibrated so that the shell volume matches the ore proportion of the (regularly spaced, hence
        # declustered) composites, as Section 5.5 recommends
        t_cal = float(np.quantile(v, 1.0 - obs.mean()))
        out[name + "_calibrated"] = dict(**score(v > t_cal), iso=round(t_cal, 4))
    reset()
    return out


def b5():
    spacings = [50] if QUICK else [25, 50, 100]
    nuggets = [0.0, 0.15] if QUICK else [0.0, 0.1, 0.25]
    for sp in spacings:
        for nug in nuggets:
            r = child(dict(b="B5", spacing=sp, nugget=nug, seed=5), 1800)
            emit(dict(bench="B5", spacing_m=sp, nugget_flip_prob=nug, **r))


# ----------------------------------------------------------------------------------------- B6
def b6():
    if not KILAMONGO.exists():
        emit(dict(bench="B6", status="skipped", reason=f"{KILAMONGO} not found")); return
    for js in ["kilamongo_indicator.json", "kilamongo_signed_distance.json", "kilamongo_contacts.json"]:
        f = KILAMONGO / js
        if not f.exists():
            emit(dict(bench="B6", settings=js, status="skipped", reason="settings file missing")); continue
        t = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-m", "deeprbf.workflow.shell", js], cwd=str(KILAMONGO), capture_output=True, text=True, timeout=1800)
            rep = None
            cfg = json.load(open(f))
            out_dir = KILAMONGO / cfg.get("output_dir", "out")
            reps = sorted(out_dir.glob("*_report.json"), key=lambda q: q.stat().st_mtime)
            if reps:
                rep = json.load(open(reps[-1]))
            keep = {k: rep.get(k) for k in ("name", "method", "data", "honour", "volume_m3", "bodies", "reference", "folds", "uncertainty", "seconds")} if rep else None
            emit(dict(bench="B6", settings=js, returncode=p.returncode, wall_s=round(time.perf_counter() - t, 1), report=keep, stdout_tail=p.stdout[-600:]))
        except subprocess.TimeoutExpired:
            emit(dict(bench="B6", settings=js, status="timeout"))


CASES = dict(B1=case_b1, B2=case_b2, B3=case_b3, B4=case_b4, B5=case_b5)

if __name__ == "__main__":
    if "--case" in sys.argv:
        c = json.loads(sys.argv[sys.argv.index("--case") + 1])
        print("@@" + json.dumps(CASES[c["b"]](c), default=float), flush=True)
        sys.exit(0)
    t0 = time.perf_counter()
    for tag, fn in [("B0", b0), ("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4), ("B5", b5), ("B6", b6)]:
        if want(tag):
            try:
                fn()
            except Exception:
                emit(dict(bench=tag, status="error", traceback=traceback.format_exc()[-1500:]))
    emit(dict(bench="END", total_minutes=round((time.perf_counter() - t0) / 60, 1)))
