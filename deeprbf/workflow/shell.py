# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Indicator grade-shell workflow: drillhole CSVs + one JSON settings file -> DeepRBF indicator RBF -> closed wireframe
(Surpac .str/.dtm and DXF) clipped to bounds and topography -> report (composites, honouring of the data, volumes,
optional withheld-hole folds, optional comparison with a reference wireframe such as a shell from other software).

    python -m deeprbf.workflow.shell settings.json

Settings (JSON; paths relative to the settings file):
{
  "name": "Kilamongo_Au_0.3",
  "drillholes": {"collar": "collar.csv", "survey": "survey.csv", "intervals": "assay.csv", "value": "AU",
                 "columns": {"hole": "BHID", "x": "XCOLLAR", "y": "YCOLLAR", "z": "ZCOLLAR", "depth": "AT", "dip": "DIP",
                             "azimuth": "BRG", "from_": "FROM", "to": "TO"},
                 "dip_positive_down": true, "composite_length": 1.0, "min_fraction": 0.5},
  "cutoff": 0.3,
  "trend": {"dip": 17.975, "dip_azimuth": 67.69, "pitch": 90, "ranges": [52.63, 43.86, 11.70]},
  "interpolant": {"kernel": "spherical", "range_factor": 1.04, "sill": 12.0, "nugget": 0.309, "drift": "none"},
  "surface": {"iso": 0.25, "spacing": 2.0, "min_volume": 1000.0, "padding": 20.0, "bounds": null, "topography": "topo.dxf"},
  "validation": {"folds": 0, "reference": "Au_Ind_kila_0.3.dtm"},
  "output_dir": "out"
}
Notes: ranges are maximum, intermediate, minimum (common implicit-modelling convention; pitch 90 = maximum down-dip). The nugget is
used as regularization nugget/(sill - nugget). Kernel "spherical" (compact; sparse solver for large data) or
"gen_cauchy[:alpha:power]" (long tail, global; slower for large data) or "exponential". "range_factor" scales the
ranges for the kernel: 1.04 makes a spherical kernel follow a spheroidal alpha 9 curve (fitted, not exact)."""
from __future__ import annotations
import json, os, sys, time, numpy as np, pandas as pd
from . import drillholes as dh, mesh as ms, surfaces as sf, distance as dist, contacts as ct
from .anisotropy import Frame
import deeprbf_recovered as native

def _p(base, path): return None if path in (None, '') else (path if os.path.isabs(path) else os.path.join(base, path))

def prepare(cfg, base='.'):
    d = cfg['drillholes']; col, sur, itv = dh.load_tables(_p(base, d['collar']), _p(base, d['survey']), _p(base, d['intervals']), d['value'], d.get('columns'))
    if not d.get('dip_positive_down', True): sur = sur.assign(dip=-sur.dip)
    comp = dh.composite(itv, d.get('composite_length', 1.0), d.get('min_fraction', 0.5)); comp, skipped = dh.desurvey(col, sur, comp)
    return comp, dict(holes=int(comp.hole.nunique()), composites=int(len(comp)), holes_skipped_no_collar_or_survey=skipped, survey_dips_clipped_to_90=getattr(dh.desurvey, 'dips_clipped', 0),
                      composite_length=d.get('composite_length', 1.0), metres=float(itv.eval('to - `from`').clip(lower=0).sum()))

class IndicatorModel:
    def __init__(self, cfg, origin):
        t, k = cfg['trend'], cfg['interpolant']; f = k.get('range_factor', 1.0)
        self.frame = Frame(t['dip'], t['dip_azimuth'], t.get('pitch', 90.0), np.asarray(t['ranges'], float) * f, origin)
        sill, nug = float(k.get('sill', 1.0)), float(k.get('nugget', 0.0))
        if not 0 <= nug < sill: raise ValueError('need 0 <= nugget < sill')
        self.kernel, self.reg, self.drift = k.get('kernel', 'spherical'), nug / (sill - nug), k.get('drift', 'none')
    def fit(self, xyz, ind):
        P = self.frame(xyz).round(7); P, inv = np.unique(P, axis=0, return_inverse=True)
        v = np.bincount(inv.ravel(), ind) / np.bincount(inv.ravel())                      # works for any values, not only 0/1                      # duplicates in model space averaged
        self.m = native.NativeExactRBFImplicitModel(self.kernel, 1.0, self.reg); self.m.set_drift(self.drift)
        t = time.time(); self.m.fit(P.tolist(), v.tolist()); self.fit_seconds = time.time() - t
        return self
    def predict(self, xyz):
        return np.asarray(self.m.predict(self.frame(xyz).tolist()))

def constraints(cfg, comp, ore, frame):
    """Points and values the interpolant is fitted to, by method, and the sign that makes the field positive inside:
    indicator: composites with 0/1 (inside where field >= iso); signed_distance: BOUNDSIM codes, ore negative
    (inside where field <= 0); contacts: hole contacts at 0, off-surface points at +/-delta along normals estimated from
    neighbouring holes, and interval mid-points (ore negative)."""
    method = cfg.get('method', 'indicator'); X = comp[['X', 'Y', 'Z']].to_numpy()
    if method == 'indicator': return X, ore.astype(float), 1.0, {}
    if method == 'signed_distance':
        return X, dist.signed_distance_codes(X, ore, frame=frame, holes=comp.hole.to_numpy()), -1.0, {}
    if method == 'contacts':
        c = cfg.get('contacts', {}); L = cfg['drillholes'].get('composite_length', 1.0)
        con = ct.extract_contacts(comp, ore, max_gap=c.get('max_gap', 2.0 * L))
        nrm, info = ct.estimate_normals(con, k=c.get('k', 8), frame=frame, same_kind=c.get('same_kind', True))
        p1, v1 = ct.constraint_points(con, nrm, c.get('delta', 2.0)); p2, v2 = ct.interval_points(con, nrm, c.get('max_half', 10.0))
        return np.vstack([p1, p2]), np.r_[v1, v2], -1.0, dict(contacts=int(len(con)), normals_fallback=int(info.fallback.sum()), median_planarity=round(float(info.planarity.median()), 3))
    raise ValueError(f'unknown method {method}: indicator | signed_distance | contacts')

def _prf(p, o):
    tp = int((p & o).sum()); pr = tp / max(int(p.sum()), 1); rc = tp / max(int(o.sum()), 1)
    return dict(precision=round(pr, 4), recall=round(rc, 4), f1=round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0)

def run(cfg, base='.'):
    t0 = time.time(); name = cfg.get('name', 'shell'); out = _p(base, cfg.get('output_dir', 'out')); os.makedirs(out, exist_ok=True)
    s = cfg.get('surface', {}); iso, h = float(s.get('iso', 0.5)), float(s.get('spacing', 2.0)); rep = dict(name=name, settings=cfg)
    comp, rep['drillholes'] = prepare(cfg, base); X = comp[['X', 'Y', 'Z']].to_numpy(); val = comp.value.to_numpy(); ore = val >= cfg['cutoff']
    rep['drillholes']['share_at_or_above_cutoff'] = round(float(ore.mean()), 4)
    method = cfg.get('method', 'indicator'); model = IndicatorModel(cfg, X.mean(0))
    cx, cv, sign, cinfo = constraints(cfg, comp, ore, model.frame); model.fit(cx, cv)
    rep['model'] = dict(method=method, constraints=int(len(cx)), **cinfo, kernel=model.kernel, regularization=round(model.reg, 6), drift=model.drift, backend=model.m.solver_backend(), fit_seconds=round(model.fit_seconds, 1))
    topo = sf.read_points(_p(base, s['topography'])) if s.get('topography') else None
    if s.get('bounds'): lo, hi = np.asarray(s['bounds'][0], float), np.asarray(s['bounds'][1], float)
    else:
        pad = float(s.get('padding', 20.0)); lo, hi = X.min(0) - pad, X.max(0) + pad
        if topo is not None: hi[2] = max(hi[2], float(topo[:, 2].max()))
    if sign < 0: iso = 0.0                                                         # distance-type fields: boundary at 0, field = -prediction
    pred, ax = ms.evaluate(model.predict, lo, hi, h, slab=max(1, int(400000 // (len(ms.grid_axes(lo, hi, h)[1]) * len(ms.grid_axes(lo, hi, h)[2])))))
    def shell_from(shift):
        g = sign * pred + shift
        if topo is not None: g = ms.clip_topography(g, ax, topo, iso)
        return ms.filter_small(g, h, iso, float(s.get('min_volume', 0.0)))
    f, bodies = shell_from(0.0)
    V, F = ms.contour(f, ax, iso)
    stem = os.path.join(out, name); ms.write_surpac(V, F, stem, name); ms.write_dxf(V, F, stem + '.dxf'); np.save(stem + '_field.npy', f)
    u = cfg.get('uncertainty')
    if u and sign < 0:                                                            # BOUNDSIM-style shells calibrated on withheld-hole contacts
        def fp(tx, tc, qx): return IndicatorModel(cfg, X.mean(0)).fit(tx, tc).predict(qx)
        cal = dist.calibrate_shift(X, ore, comp.hole.to_numpy(), model.frame, fp, targets=tuple(u.get('quantiles', (0.1, 0.5, 0.9))), folds=int(u.get('folds', 5)), seed=int(u.get('seed', 0)), contact_codes=True)
        rep['uncertainty'] = dict(method='withheld-hole field quantiles at true contacts', C=cal['C'], label=cal['label'], coverage=cal['coverage'], shells={})
        for q in u.get('quantiles', (0.1, 0.5, 0.9)):
            key = next(kk for kk in cal['C'] if float(kk) == float(q)); C, lab = cal['C'][key], cal['label'][key]
            g, b2 = shell_from(float(C)); Vq, Fq = ms.contour(g, ax, iso); sq = f'{stem}_{lab}'; ms.write_surpac(Vq, Fq, sq, f'{name}_{lab}'); ms.write_dxf(Vq, Fq, sq + '.dxf')
            rep['uncertainty']['shells'][lab] = dict(quantile=q, C=float(C), volume_m3=round(ms.volume(Vq, Fq)), bodies=b2['bodies_kept'], files=[sq + e for e in ('.str', '.dtm', '.dxf')])
    comp.assign(indicator=ore.astype(int)).to_csv(stem + '_composites.csv', index=False)
    from scipy.interpolate import RegularGridInterpolator
    at = RegularGridInterpolator(ax, f, bounds_error=False, fill_value=-1.0)(X) >= iso
    rep['wireframe'] = dict(files=[stem + e for e in ('.str', '.dtm', '.dxf')], points=int(len(V)), triangles=int(len(F)), volume_m3=round(ms.volume(V, F)),
                            grid=dict(lo=lo.tolist(), hi=hi.tolist(), spacing=h), iso=iso, topography=s.get('topography'), **bodies)
    rep['honouring'] = dict(**_prf(at, ore), gold_inside_share=round(float(val[at].sum() / max(val.sum(), 1e-12)), 4),
                            mean_inside=round(float(val[at].mean()), 4) if at.any() else None, mean_outside=round(float(val[~at].mean()), 4))
    v = cfg.get('validation', {})
    if v.get('reference'):
        T = ms.read_wireframe(_p(base, v['reference'])); iR = ms.inside(T, X)
        sub = np.stack(np.meshgrid(*[a[::2] for a in ax], indexing='ij'), -1).reshape(-1, 3) + 0.0137
        gR = ms.inside(T, sub); gD = RegularGridInterpolator(ax, f, bounds_error=False, fill_value=-1.0)(sub) >= iso; cell = (2 * h) ** 3
        rep['reference'] = dict(file=v['reference'], volume_m3=round(ms.volume(T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3))),
                                honouring=dict(**_prf(iR, ore), gold_inside_share=round(float(val[iR].sum() / max(val.sum(), 1e-12)), 4)),
                                shared_share_of_reference=round(float((gR & gD).sum() / max(gR.sum(), 1)), 3), shared_share_of_deeprbf=round(float((gR & gD).sum() / max(gD.sum(), 1)), 3),
                                grid_volume_reference_m3=round(float(gR.sum() * cell)), grid_volume_deeprbf_m3=round(float(gD.sum() * cell)))
    k = int(v.get('folds', 0))
    if k > 1:                                                                   # withheld holes: support for geometry decisions
        holes = np.random.default_rng(int(v.get('seed', 0))).permutation(comp.hole.unique()); predh = np.zeros(len(comp), bool)
        for i in range(k):
            te = comp.hole.isin(holes[i::k]).to_numpy(); mi = IndicatorModel(cfg, X.mean(0)); tc = comp[~te].reset_index(drop=True)
            tx, tv, _, _ = constraints(cfg, tc, ore[~te], mi.frame); mi.fit(tx, tv); predh[te] = sign * mi.predict(X[te]) >= iso
        rep['withheld_holes'] = dict(folds=k, **_prf(predh, ore), baseline_f1_all_ore=round(float(2 * ore.mean() / (1 + ore.mean())), 4),
                                     gold_inside_share=round(float(val[predh].sum() / max(val.sum(), 1e-12)), 4), note='scores the trend and settings on unseen holes; the wireframe itself uses all holes')
    rep['seconds'] = round(time.time() - t0, 1); rep['engine'] = dict(native_version=getattr(native, '__version__', None))
    json.dump(rep, open(stem + '_report.json', 'w'), indent=1, default=str); return rep

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1: print(__doc__); return 2
    path = os.path.abspath(argv[0]); rep = run(json.load(open(path)), os.path.dirname(path))
    w, hn = rep['wireframe'], rep['honouring']
    print(f"{rep['name']}: {rep['drillholes']['composites']} composites, {w['bodies_kept']} bodies, {w['volume_m3']/1e6:.2f} Mm3, F1 {hn['f1']}, gold inside {hn['gold_inside_share']:.1%}, {rep['seconds']} s")
    if 'reference' in rep: r = rep['reference']; print(f"reference: F1 {r['honouring']['f1']}, shared {r['shared_share_of_reference']:.0%} of reference volume")
    if 'withheld_holes' in rep: print(f"withheld holes ({rep['withheld_holes']['folds']} folds): F1 {rep['withheld_holes']['f1']} vs all-ore {rep['withheld_holes']['baseline_f1_all_ore']}")
    print('outputs:', ', '.join(w['files'])); return 0

if __name__ == '__main__':
    raise SystemExit(main())
