# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Sparse variational Gaussian processes (SVGP) for geology - ore-shell / rock-unit probabilities and grade estimation.

Independent PyTorch implementation of the standard SVGP (Titsias 2009; Hensman, Fusi & Lawrence 2013; Hensman, Matthews &
Ghahramani 2015) following the geological use described by Goncalves & Nwaila (2026, Artificial Intelligence in
Geosciences 7(2) 100218, Thalanga VHMS). No geoML / GPflow code is used.

Model
  f_l(x) ~ GP(0, k_l)  for each latent l = 1..L, k_l Matern (nu 1/2, 3/2 or 5/2) with ARD length scales in a rotated frame
  inducing variables u_l = f_l(Z) at M shared, learnable locations Z (k-means initialised); whitened q(v_l) = N(m_l, S_l)
  ELBO = N/B * sum_batch E_q[log p(y | f)] - sum_l KL(q(v_l) || N(0, I)), trained with Adam on mini-batches
Likelihoods
  softmax (rock units / ore-waste, MC), bernoulli (probit, Gauss-Hermite), gaussian, student_t (Gauss-Hermite)
Compositions (1.1)
  several grades of one sample (e.g. Zn, Pb, Cu % and the rest to 100 %) -> centred log-ratios -> PCA -> one GP per
  component, Gaussian noise each; back-transformed by Monte Carlo through inverse PCA, exp and closure, so estimates are
  always valid compositions and the ratios between metals are modelled jointly.
Grades
  values -> normal scores (anamorphosis) -> GP with Gaussian noise; estimates back-transformed by Gauss-Hermite quadrature
  over the predictive distribution (point: latent + noise; block: latent only, averaged over discretisation points),
  so heavy tails come back through the anamorphosis rather than being smoothed away.
"""
import math
import os
import subprocess
import sys

import numpy as np

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(24)            # Gauss-Hermite nodes for E[g(N(mu, var))]

# On Windows a broken torch install (DLL entry-point errors, or an OpenMP runtime that clashes with numpy's) can
# abort the whole Python process rather than raise. Probe torch next to numpy in a child process first, so a bad
# install only disables SVGP. Set DEEPRBF_TORCH_PROBE=0 to skip the probe.
_PROBE = ("import numpy as np, scipy.linalg, torch; np.polynomial.hermite.hermgauss(24); "
          "np.linalg.eigvalsh(np.eye(3)); float(torch.zeros(2, dtype=torch.float64).sum())")


def _torch_probe_failure():
    if sys.platform != "win32" or os.environ.get("DEEPRBF_TORCH_PROBE", "1") == "0":
        return None
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300)
    except Exception:                         # cannot probe: fall back to a normal import attempt
        return None
    if r.returncode == 0:
        return None
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
    return RuntimeError(f"torch fails to load next to numpy in this environment (exit code {r.returncode}): {tail[0][:200]}")


_TORCH_ERROR = _torch_probe_failure()
if _TORCH_ERROR is not None:
    torch = None
else:
    try:
        import torch
    except Exception as _e:                   # torch is optional for the rest of DeepRBF
        torch = None
        _TORCH_ERROR = _e

__all__ = ["SVGPClassifier", "SVGPRegressor", "SVGPCompositional", "Anamorphosis", "svgp_available", "coverage", "coverage_curve"]
__svgp_version__ = "1.1.0"   # pure-Python module inside DeepRBF 3.3.1 (no native rebuild needed); 1.1: compositions, interval levels


def svgp_available():
    return (torch is not None), ("torch " + torch.__version__ if torch is not None else f"torch not importable: {_TORCH_ERROR}")


def _device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ----------------------------------------------------------------------------- normal-score transform
class Anamorphosis:
    """Empirical normal-score transform with linear tails in score space beyond the data (no extrapolated grades
    outside [min, max])."""

    def __init__(self, values):
        v = np.sort(np.asarray(values, float)[np.isfinite(values)])
        if len(v) < 3:
            raise ValueError("need at least 3 values for a normal-score transform")
        from scipy.stats import norm
        n = len(v)
        p = (np.arange(n) + 0.5) / n
        self.z, self.v = norm.ppf(p), v
        # collapse ties: one score per distinct value (mean of its scores)
        u, inv = np.unique(self.v, return_inverse=True)
        zs = np.bincount(inv, weights=self.z) / np.bincount(inv)
        self.v, self.z = u, zs

    def transform(self, x):
        return np.interp(np.asarray(x, float), self.v, self.z)

    def inverse(self, z):
        return np.interp(np.asarray(z, float), self.z, self.v)


# ----------------------------------------------------------------------------- kernels and the SVGP core
class _Matern:
    def __init__(self, nu):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be 0.5, 1.5 or 2.5")
        self.nu = nu

    def __call__(self, A, B, ls, var):
        d = (A[:, None, :] - B[None, :, :]) / ls
        r = torch.sqrt((d * d).sum(-1) + 1e-12)
        if self.nu == 0.5:
            k = torch.exp(-r)
        elif self.nu == 1.5:
            s = math.sqrt(3) * r; k = (1 + s) * torch.exp(-s)
        else:
            s = math.sqrt(5) * r; k = (1 + s + s * s / 3) * torch.exp(-s)
        return var * k


class _SVGP(torch.nn.Module if torch is not None else object):
    def __init__(self, Z, L, ls0, nu=1.5, var0=1.0, learn_inducing=True):
        super().__init__()
        M, D = Z.shape
        self.Z = torch.nn.Parameter(Z.clone(), requires_grad=learn_inducing)
        self.log_ls = torch.nn.Parameter(torch.log(ls0).repeat(L, 1))               # L x D
        self.log_var = torch.nn.Parameter(torch.full((L,), math.log(var0), dtype=Z.dtype, device=Z.device))
        self.m = torch.nn.Parameter(torch.zeros(L, M, dtype=Z.dtype, device=Z.device))
        self.Ls_raw = torch.nn.Parameter(torch.eye(M, dtype=Z.dtype, device=Z.device).repeat(L, 1, 1) * 1e-1)           # whitened: S starts small
        self.k = _Matern(nu); self.L, self.M = L, M

    def _Ls(self):
        T = torch.tril(self.Ls_raw)
        d = torch.diagonal(T, dim1=-2, dim2=-1)
        return T - torch.diag_embed(d) + torch.diag_embed(torch.nn.functional.softplus(d) + 1e-5)

    def _Lzz(self):
        out = []
        for l in range(self.L):
            K = self.k(self.Z, self.Z, torch.exp(self.log_ls[l]), torch.exp(self.log_var[l]))
            jit = 1e-5 if K.dtype == torch.float64 else 1e-3
            out.append(torch.linalg.cholesky(K + jit * torch.exp(self.log_var[l]) * torch.eye(self.M, device=K.device, dtype=K.dtype)))
        return out

    def marginals(self, X, Lzz=None):
        """Predictive mean and variance of every latent at X: N x L each."""
        Lzz = Lzz or self._Lzz(); Ls = self._Ls()
        mu, var = [], []
        for l in range(self.L):
            Kzx = self.k(self.Z, X, torch.exp(self.log_ls[l]), torch.exp(self.log_var[l]))
            A = torch.linalg.solve_triangular(Lzz[l], Kzx, upper=False)                 # M x N
            mu.append(A.T @ self.m[l])
            SA = Ls[l].T @ A
            v = torch.exp(self.log_var[l]) - (A * A).sum(0) + (SA * SA).sum(0)
            var.append(torch.clamp(v, min=1e-8))
        return torch.stack(mu, 1), torch.stack(var, 1)

    def kl(self):
        Ls = self._Ls()
        tr = (Ls * Ls).sum((1, 2))
        logdet = 2 * torch.log(torch.diagonal(Ls, dim1=-2, dim2=-1)).sum(-1)
        return 0.5 * (tr + (self.m * self.m).sum(-1) - self.M - logdet).sum()


def _kmeans(X, M, seed):
    from sklearn.cluster import MiniBatchKMeans
    M = min(M, len(X))
    km = MiniBatchKMeans(n_clusters=M, random_state=seed, n_init=3, batch_size=4096).fit(X)
    return km.cluster_centers_


class _Base:
    def __init__(self, frame=None, origin=None, n_inducing=512, nu=1.5, iterations=3000, batch_size=2048, lr=0.01,
                 lengthscale=None, seed=0, device=None, verbose=False):
        self.R = np.eye(3) if frame is None else np.asarray(frame, float)
        self.origin = None if origin is None else np.asarray(origin, float)
        self.M, self.nu, self.iters, self.bs, self.lr = int(n_inducing), float(nu), int(iterations), int(batch_size), float(lr)
        self.ls0, self.seed, self.dev, self.verbose = lengthscale, int(seed), device, verbose

    def _local(self, X):
        X = np.asarray(X, float)
        if self.origin is None:
            self.origin = X.mean(0)
        return (X - self.origin) @ self.R

    def _init(self, Xl, L):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device = _device(self.dev)
        # consumer GPUs run float64 at ~1/64 of float32 speed: single precision on CUDA, double on CPU
        import os
        f32 = self.device.type == "cuda" or os.environ.get("DEEPRBF_SVGP_FLOAT32") == "1"
        self.dtype = torch.float32 if f32 else torch.float64
        Z = torch.tensor(_kmeans(Xl, self.M, self.seed), dtype=self.dtype, device=self.device)
        ext = np.maximum(Xl.max(0) - Xl.min(0), 1.0)
        ls0 = np.asarray(self.ls0, float) if self.ls0 is not None else ext / 8.0
        self.model = _SVGP(Z, L, torch.tensor(ls0, dtype=self.dtype, device=self.device), nu=self.nu).to(device=self.device, dtype=self.dtype)

    def _train(self, Xt, loglik, N):
        import time
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        hist = []; t0 = time.time()
        if self.verbose:
            print(f"[svgp] {N:,} samples, {self.M} inducing, {self.model.L} latent, {self.iters} steps on {self.device} ({self.dtype})", flush=True)
        for it in range(self.iters):
            idx = torch.randint(0, N, (min(self.bs, N),), generator=g).to(self.device)
            mu, var = self.model.marginals(Xt[idx])
            elbo = N / len(idx) * loglik(idx, mu, var) - self.model.kl()
            opt.zero_grad(); (-elbo / N).backward(); opt.step()
            if it % 100 == 0 or it == self.iters - 1:
                hist.append(float(elbo.detach()) / N)
                if self.verbose and (it % 500 == 0 or it == self.iters - 1):
                    print(f"[svgp] step {it}: ELBO/N {hist[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
        self.elbo_history = hist

    def _marginals(self, Q, chunk=4096):
        Ql = torch.tensor(self._local(Q), dtype=self.dtype, device=self.device)
        mus, vs = [], []
        with torch.no_grad():
            Lzz = self.model._Lzz()
            for s in range(0, len(Ql), chunk):
                m, v = self.model.marginals(Ql[s:s + chunk], Lzz)
                mus.append(m.cpu().numpy()); vs.append(v.cpu().numpy())
        return np.vstack(mus) if mus else np.zeros((0, self.model.L)), np.vstack(vs) if vs else np.zeros((0, self.model.L))

    def hyperparameters(self):
        return {"lengthscales_m": np.exp(self.model.log_ls.detach().cpu().numpy()).round(2).tolist(),
                "variance": np.exp(self.model.log_var.detach().cpu().numpy()).round(4).tolist(),
                "elbo_per_sample": [round(x, 4) for x in getattr(self, "elbo_history", [])][-5:]}


# ----------------------------------------------------------------------------- classification: rock units / ore shells
class SVGPClassifier(_Base):
    """P(class | x) for K classes with K latent GPs and a softmax likelihood (MC, reparameterised).
    With two classes this is an ore-shell model: P(ore) and the shell at P = 0.5 (or any threshold)."""

    def __init__(self, classes=None, mc_samples=8, **kw):
        super().__init__(**kw); self.classes, self.S = classes, int(mc_samples)

    def fit(self, X, labels):
        lab = np.asarray(labels, object)
        self.classes = list(self.classes) if self.classes is not None else sorted(set(lab))
        idx = {c: i for i, c in enumerate(self.classes)}
        keep = np.array([v in idx for v in lab])
        Xl = self._local(np.asarray(X, float)[keep]); y = np.array([idx[v] for v in lab[keep]])
        self._init(Xl, len(self.classes))
        Xt = torch.tensor(Xl, dtype=self.dtype, device=self.device); yt = torch.tensor(y, device=self.device)
        # class proportions as a fixed mean offset: rare classes need not be pulled up from zero by the GP
        pri = np.bincount(y, minlength=len(self.classes)) / len(y)
        self.logprior = torch.tensor(np.log(np.maximum(pri, 1e-4)), dtype=self.dtype, device=self.device)
        S = self.S

        def loglik(i, mu, var):
            eps = torch.randn((S,) + mu.shape, dtype=mu.dtype, device=mu.device)
            f = mu + torch.sqrt(var) * eps + self.logprior
            lp = torch.log_softmax(f, -1).gather(-1, yt[i].expand(S, -1).unsqueeze(-1)).squeeze(-1)
            return lp.mean(0).sum()
        self._train(Xt, loglik, len(Xt))
        return self

    def predict_proba(self, Q, samples=64):
        mu, var = self._marginals(Q)
        rng = np.random.default_rng(self.seed)
        lp = self.logprior.cpu().numpy()
        out = np.zeros_like(mu)
        for _ in range(samples):
            f = mu + np.sqrt(var) * rng.standard_normal(mu.shape) + lp
            f -= f.max(1, keepdims=True); e = np.exp(f); out += e / e.sum(1, keepdims=True)
        return out / samples


# ----------------------------------------------------------------------------- regression: grades
class SVGPRegressor(_Base):
    """Grade estimation: normal scores -> GP + Gaussian noise; estimates are expectations of the back-transformed
    predictive (Gauss-Hermite). transform='none' fits the raw values with a Student-t likelihood instead."""

    def __init__(self, transform="nscore", df=4.0, **kw):
        super().__init__(**kw); self.transform, self.df = transform, float(df)

    def fit(self, X, y):
        y = np.asarray(y, float); ok = np.isfinite(y)
        X, y = np.asarray(X, float)[ok], y[ok]
        if self.transform == "nscore":
            self.ana = Anamorphosis(y); t = self.ana.transform(y); self.shift, self.scale = 0.0, 1.0
        else:
            self.ana = None; self.shift, self.scale = float(np.median(y)), float(np.std(y) or 1.0); t = (y - self.shift) / self.scale
        Xl = self._local(X); self._init(Xl, 1)
        Xt = torch.tensor(Xl, dtype=self.dtype, device=self.device); yt = torch.tensor(t, dtype=self.dtype, device=self.device)
        self.log_noise = torch.nn.Parameter(torch.tensor(math.log(0.3), dtype=self.dtype, device=self.device))
        self.model.register_parameter("log_noise", self.log_noise)
        gx = torch.tensor(_GH_X, dtype=self.dtype, device=self.device); gw = torch.tensor(_GH_W / math.sqrt(math.pi), dtype=self.dtype, device=self.device)
        nu = self.df

        def loglik(i, mu, var):
            mu, var = mu[:, 0], var[:, 0]; s2 = torch.exp(2 * self.log_noise)
            if self.transform == "nscore":
                return (-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(s2) - 0.5 * ((yt[i] - mu) ** 2 + var) / s2).sum()
            f = mu[:, None] + math.sqrt(2) * torch.sqrt(var)[:, None] * gx[None]            # Student-t, quadrature
            r = (yt[i][:, None] - f) ** 2 / (nu * s2)
            lp = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - 0.5 * torch.log(s2)
                  - (nu + 1) / 2 * torch.log1p(r))
            return (lp * gw[None]).sum()
        self._train(Xt, loglik, len(Xt))
        self.noise_var = float(torch.exp(2 * self.log_noise).detach())
        return self

    def _back(self, mu, var, levels=(10, 50, 90)):
        """E[g(Y)], Y ~ N(mu, var), g = inverse transform; also the quantiles at `levels` (percent)."""
        from scipy.stats import norm
        sd = np.sqrt(np.maximum(var, 1e-12))
        nodes = mu[:, None] + math.sqrt(2) * sd[:, None] * _GH_X[None]
        g = self.ana.inverse(nodes) if self.ana is not None else nodes * self.scale + self.shift
        mean = (g * (_GH_W / math.sqrt(math.pi))[None]).sum(1)
        q = {}
        for p in levels:
            z = mu + float(norm.ppf(p / 100.0)) * sd
            q[p] = self.ana.inverse(z) if self.ana is not None else z * self.scale + self.shift
        return mean, q

    def predict(self, Q, block=None, disc=(1, 1, 1), return_quantiles=False, levels=(10, 50, 90)):
        """Point estimates (latent + noise: the value of a sample there) or, with block=(dx, dy, dz) and disc, the block
        mean (latent only, averaged over the discretisation points)."""
        Q = np.asarray(Q, float)
        if block is None:
            mu, var = self._marginals(Q); mu, var = mu[:, 0], var[:, 0] + self.noise_var
            mean, q = self._back(mu, var, levels)
            return (mean, q) if return_quantiles else mean
        off = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) / n - 0.5 for n in disc], indexing="ij"), -1).reshape(-1, 3) * np.asarray(block, float)
        P = (Q[:, None, :] + off[None]).reshape(-1, 3)
        mu, var = self._marginals(P)
        mean, _ = self._back(mu[:, 0], var[:, 0])
        return mean.reshape(len(Q), len(off)).mean(1)


def coverage(true, lo, hi):
    """Share of true values inside [lo, hi] - for calibration plots (nominal vs observed coverage)."""
    t = np.asarray(true, float); ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    return float(((t[ok] >= np.asarray(lo)[ok]) & (t[ok] <= np.asarray(hi)[ok])).mean()) if ok.any() else float("nan")


def coverage_curve(true, quantiles, nominal=(50, 80, 90, 95)):
    """Observed coverage of central intervals: {nominal %: observed share}. `quantiles` = {percent: array}, holding the
    (100 - c) / 2 and (100 + c) / 2 levels for every nominal c."""
    return {c: round(coverage(true, quantiles[(100 - c) / 2], quantiles[(100 + c) / 2]), 4) for c in nominal}


def interval_levels(nominal=(50, 80, 90, 95)):
    return tuple(sorted({(100 - c) / 2 for c in nominal} | {50.0} | {(100 + c) / 2 for c in nominal}))


# ----------------------------------------------------------------------------- compositions: joint grades
class SVGPCompositional(_Base):
    """Joint grades of several parts of one composition (e.g. Zn, Pb, Cu in %) by log-ratios.

    The parts plus a filler ('rest' = total - sum) are closed to `total`, zeros replaced by half the smallest positive value
    of that part, transformed to centred log-ratios (CLR) and rotated by PCA; the D - 1 informative components are
    standardised and each gets its own GP (shared inducing points, own kernel and noise). Predictions are Monte Carlo
    draws of the components (latent + noise for a sample, latent only for a block), back through inverse PCA, exp and
    closure. Estimates are draw means; quantiles are draw quantiles."""

    def __init__(self, total=100.0, filler=True, samples=256, **kw):
        super().__init__(**kw); self.total, self.filler, self.S = float(total), bool(filler), int(samples)

    def _clr(self, Y):
        Y = np.maximum(np.asarray(Y, float), self.dl[None])
        if self.filler:
            rest = np.maximum(self.total - Y.sum(1, keepdims=True), self.dl_rest)
            Y = np.hstack([Y, rest])
        L = np.log(Y)
        return L - L.mean(1, keepdims=True)

    def fit(self, X, Y):
        Y = np.asarray(Y, float); X = np.asarray(X, float)
        ok = np.isfinite(Y).all(1); X, Y = X[ok], Y[ok]
        if len(Y) < 10:
            raise ValueError("need at least 10 complete samples")
        pos = np.where(Y > 0, Y, np.nan)
        self.dl = 0.5 * np.nanmin(pos, 0)
        self.dl = np.where(np.isfinite(self.dl), self.dl, 1e-6)
        rest = self.total - Y.sum(1)
        self.dl_rest = 0.5 * float(np.min(rest[rest > 0])) if (rest > 0).any() else 1e-3
        C = self._clr(Y)
        self.c_mean = C.mean(0)
        U, sv, Vt = np.linalg.svd(C - self.c_mean, full_matrices=False)
        k = C.shape[1] - 1                                  # CLR rows sum to zero: one direction carries nothing
        self.V = Vt[:k]                                     # k x D
        T = (C - self.c_mean) @ self.V.T
        self.t_sd = T.std(0) + 1e-12; T = T / self.t_sd
        Xl = self._local(X); self._init(Xl, k)
        Xt = torch.tensor(Xl, dtype=self.dtype, device=self.device); Tt = torch.tensor(T, dtype=self.dtype, device=self.device)
        self.log_noise = torch.nn.Parameter(torch.full((k,), math.log(0.3), dtype=self.dtype, device=self.device))
        self.model.register_parameter("log_noise", self.log_noise)

        def loglik(i, mu, var):
            s2 = torch.exp(2 * self.log_noise)[None]
            return (-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(s2) - 0.5 * ((Tt[i] - mu) ** 2 + var) / s2).sum()
        self._train(Xt, loglik, len(Xt))
        self.noise_var = torch.exp(2 * self.log_noise).detach().cpu().numpy()
        self.n_parts = Y.shape[1]
        return self

    def _draws(self, mu, var, rng):
        """S x N x parts draws of the composition (parts only, filler dropped)."""
        T = mu[None] + np.sqrt(np.maximum(var, 1e-12))[None] * rng.standard_normal((self.S,) + mu.shape)
        C = (T * self.t_sd) @ self.V + self.c_mean
        E = np.exp(C - C.max(-1, keepdims=True))
        Y = self.total * E / E.sum(-1, keepdims=True)
        return Y[..., :self.n_parts]

    def predict(self, Q, return_quantiles=False, levels=(10, 50, 90), chunk=2000):
        """Point (sample-support) estimates: N x parts means, and with return_quantiles a dict
        {percent: N x parts} of draw quantiles."""
        Q = np.asarray(Q, float); rng = np.random.default_rng(self.seed)
        mean = np.zeros((len(Q), self.n_parts)); q = {p: np.zeros((len(Q), self.n_parts)) for p in levels}
        for s in range(0, len(Q), chunk):
            mu, var = self._marginals(Q[s:s + chunk]); D = self._draws(mu, var + self.noise_var[None], rng)
            mean[s:s + chunk] = D.mean(0)
            for p in levels:
                q[p][s:s + chunk] = np.percentile(D, p, axis=0)
        return (mean, q) if return_quantiles else mean
