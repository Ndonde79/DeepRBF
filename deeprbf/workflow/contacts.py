# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Ore/waste contacts along drillholes as interpolation constraints for a signed-distance grade shell.

Idea after Wang et al. (2018, Minerals 8:443, Hermite RBF ore-body modelling): the shell passes through the ore/waste
contacts seen in each hole, and its orientation there is taken from the contacts of neighbouring holes (local plane by
PCA) instead of the hole axis. Without Hermite (gradient) constraints in the engine, the normal is imposed with the
off-surface-point formulation of Carr et al. (2001): value 0 at the contact and +/-delta at points offset by delta
along the unit normal.

Conventions
  * composites: DataFrame with hole, from, to, X, Y, Z (XYZ at composite mid-depth); rows are sorted by hole and
    depth here, the input order is not relied on.
  * ore: boolean per composite (array aligned with the rows, or the name of a boolean column).
  * contact 'top': waste above, ore below going down the hole; 'bottom': ore above, waste below.
  * hole direction (dx, dy, dz): unit down-hole vector between the two composites that define the contact.
  * normals point from ore toward waste; signed distance is negative inside ore, positive in waste.

    c = extract_contacts(comp, comp.value >= cutoff, max_gap=2.0)
    n, info = estimate_normals(c, k=8, frame=Frame(...))
    xyz, v = constraint_points(c, n, delta=2.0)
    xi, vi = interval_points(c, n)                    # optional: anchors mid-interval (recommended, see below)

Holes that start or end in ore give no contact there (the shell is left to the interpolant at such ends).
Off-surface points only fix the field within about delta of each contact; between widely spaced holes and across a
thick interval a smooth interpolant (e.g. linear kernel + linear drift) relaxes toward zero, so the middle of the ore
can come out barely negative or positive. interval_points adds one signed-distance value at the middle of each
along-hole interval between consecutive contacts, which fixes that."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

CONTACT_COLUMNS = ['hole', 'X', 'Y', 'Z', 'depth', 'kind', 'dx', 'dy', 'dz', 'gap']

def extract_contacts(comp, ore, max_gap=None):
    """Contacts between consecutive composites of different class in the same hole.

    comp: composites (hole, from, to, X, Y, Z). ore: boolean array aligned with comp rows, or a column name.
    max_gap: skip a pair whose along-hole gap (next.from - previous.to) exceeds it (missing samples); None keeps all.
    Returns DataFrame with CONTACT_COLUMNS: position = midpoint of the two composite centroids, depth = mean of their
    mid-depths, kind 'top'|'bottom', unit down-hole direction (dx, dy, dz), gap = along-hole gap between the pair."""
    o = comp[ore] if isinstance(ore, str) else ore
    o = np.asarray(o, dtype=bool).ravel()
    if len(o) != len(comp): raise ValueError('ore must have one value per composite')
    df = comp.assign(_ore=o).sort_values(['hole', 'from'], kind='stable').reset_index(drop=True)
    h = df['hole'].to_numpy(); a = df['from'].to_numpy(float); b = df['to'].to_numpy(float)
    P = df[['X', 'Y', 'Z']].to_numpy(float); oo = df['_ore'].to_numpy(bool)
    i = np.flatnonzero((h[:-1] == h[1:]) & (oo[:-1] != oo[1:])); j = i + 1
    gap = a[j] - b[i]
    if max_gap is not None:
        keep = gap <= max_gap; i, j, gap = i[keep], j[keep], gap[keep]
    d = P[j] - P[i]; L = np.linalg.norm(d, axis=1)
    d = np.where(L[:, None] > 1e-9, d / np.maximum(L, 1e-12)[:, None], [0.0, 0.0, -1.0])
    X = (P[i] + P[j]) / 2
    return pd.DataFrame(dict(hole=h[i], X=X[:, 0], Y=X[:, 1], Z=X[:, 2], depth=((a[i] + b[i]) / 2 + (a[j] + b[j]) / 2) / 2,
                             kind=np.where(oo[j], 'top', 'bottom'), dx=d[:, 0], dy=d[:, 1], dz=d[:, 2], gap=gap))[CONTACT_COLUMNS]

def _toward_waste(contacts):
    """Unit vector along the hole pointing to the waste side: up-hole at a top contact, down-hole at a bottom one."""
    d = contacts[['dx', 'dy', 'dz']].to_numpy(float)
    return np.where((contacts['kind'].to_numpy() == 'top')[:, None], -d, d)

def estimate_normals(contacts, k=8, frame=None, same_kind=True, per_hole=1, min_neighbours=3, min_planarity=0.5):
    """Unit normals (ore -> waste) at contacts from a local PCA plane through each contact and its neighbours.

    Neighbours: the k nearest contacts from OTHER holes (at most per_hole per hole, the nearest; per_hole=None for no
    limit), of the same kind when same_kind, by distance in frame(xyz) coordinates if a frame is given (e.g.
    anisotropy.Frame), else world. The plane is fitted in world coordinates to the contact plus its neighbours; the
    normal is the smallest principal axis, oriented so that it points to the waste side of the contact as seen along
    the hole (top: against the down-hole direction; bottom: with it).
    planarity = 1 - l3/l2 (l1 >= l2 >= l3 eigenvalues; 1 = perfect plane, 0 = no preferred plane). Falls back to the
    hole axis (toward waste) when fewer than min_neighbours neighbours, planarity < min_planarity, or the neighbours
    are collinear (l2 <= 1e-6 l1).
    Returns (normals (n,3), info DataFrame with planarity, neighbours, fallback (bool), cos_hole = |normal . hole
    direction| (small = contact grazed by the hole, orientation by hole side less certain))."""
    n = len(contacts)
    P = contacts[['X', 'Y', 'Z']].to_numpy(float); W = _toward_waste(contacts)
    S = np.asarray(frame(P), float) if frame is not None else P
    holes = contacts['hole'].to_numpy(); kinds = contacts['kind'].to_numpy()
    N = W.copy(); plan = np.zeros(n); nnb = np.zeros(n, int); fb = np.ones(n, bool)
    groups = [np.flatnonzero(kinds == g) for g in np.unique(kinds)] if same_kind else [np.arange(n)]
    for idx in groups:
        if not len(idx): continue
        tree = cKDTree(S[idx])
        for q in idx:
            m = len(idx); kk = min(m, (k * (per_hole or 1) + 1) * 4 + 8)
            while True:
                dist, loc = tree.query(S[q], k=kk); loc = np.atleast_1d(loc); dist = np.atleast_1d(dist)
                loc = loc[np.isfinite(dist)]; cand = idx[loc]; cand = cand[holes[cand] != holes[q]]
                if per_hole is not None:
                    cnt = {}; sel = []
                    for c in cand:                                                      # nearest-first per hole
                        hc = holes[c]; cnt[hc] = cnt.get(hc, 0) + 1
                        if cnt[hc] <= per_hole: sel.append(c)
                    cand = np.asarray(sel, int)
                if len(cand) >= k or kk >= m: break
                kk = min(m, kk * 4)
            nb = cand[:k]; nnb[q] = len(nb)
            if len(nb) < min_neighbours: continue
            Q = P[np.r_[q, nb]]; Q = Q - Q.mean(0)
            lam, vec = np.linalg.eigh(Q.T @ Q)                                          # ascending
            if lam[1] <= 1e-6 * max(lam[2], 1e-300): continue
            plan[q] = 1 - lam[0] / lam[1]
            if plan[q] < min_planarity: continue
            v = vec[:, 0]; N[q] = v if v @ W[q] >= 0 else -v; fb[q] = False
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    d = contacts[['dx', 'dy', 'dz']].to_numpy(float)
    info = pd.DataFrame(dict(planarity=plan, neighbours=nnb, fallback=fb, cos_hole=np.abs(np.sum(N * d, axis=1))), index=contacts.index)
    return N, info

def interval_points(contacts, normals, max_half=None):
    """Optional mid-interval constraints from the holes themselves: for each pair of consecutive contacts in a hole,
    the midpoint between them gets minus (ore interval, top then bottom) or plus (waste interval, bottom then top)
    half the interval thickness measured across the shell: along-hole length times the mean |normal . hole direction|
    of the two contacts. max_half caps the magnitude. Off-surface points alone only fix the field within delta of the
    contacts; these anchor it in the middle of thick or widely spaced intervals. Returns (xyz (m,3), values (m,))."""
    c = contacts.reset_index(drop=True); N = np.asarray(normals, float)
    order = np.lexsort((c['depth'].to_numpy(float), c['hole'].to_numpy())); c = c.iloc[order].reset_index(drop=True); N = N[order]
    h = c['hole'].to_numpy(); kd = c['kind'].to_numpy(); P = c[['X', 'Y', 'Z']].to_numpy(float); d = c[['dx', 'dy', 'dz']].to_numpy(float)
    i = np.flatnonzero((h[:-1] == h[1:]) & (kd[:-1] != kd[1:])); j = i + 1
    L = np.linalg.norm(P[j] - P[i], axis=1)
    cs = (np.abs(np.sum(N[i] * d[i], 1)) + np.abs(np.sum(N[j] * d[j], 1))) / 2
    half = L * cs / 2
    if max_half is not None: half = np.minimum(half, max_half)
    return (P[i] + P[j]) / 2, np.where(kd[i] == 'top', -half, half)

def constraint_points(contacts, normals, delta, prune=True):
    """Signed-distance constraints: contact points at 0, contact + delta*normal at +delta (waste side) and
    contact - delta*normal at -delta (ore side). delta: scalar or one value per contact (> 0).
    prune: drop an off-surface point whose nearest contact is not its own (Carr et al. 2001 validity check; guards
    against offsets crossing a nearby opposite contact in thin ore or waste). Returns (xyz (m,3), values (m,)) ordered
    contacts, waste-side points, ore-side points."""
    P = contacts[['X', 'Y', 'Z']].to_numpy(float) if isinstance(contacts, pd.DataFrame) else np.asarray(contacts, float)
    N = np.asarray(normals, float); n = len(P)
    if N.shape != (n, 3): raise ValueError('normals must be (n, 3) for n contacts')
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    dl = np.broadcast_to(np.asarray(delta, float), (n,)).astype(float)
    if (dl <= 0).any(): raise ValueError('delta must be positive')
    out, val = P + dl[:, None] * N, P - dl[:, None] * N
    xyz = [P]; v = [np.zeros(n)]
    for Q, s in ((out, dl), (val, -dl)):
        keep = np.ones(n, bool)
        if prune and n > 1:
            _, nearest = cKDTree(P).query(Q, k=1); dd = np.linalg.norm(Q - P[nearest], axis=1)
            keep = (nearest == np.arange(n)) | (dd >= dl * (1 - 1e-9))               # ties with own contact kept
        xyz.append(Q[keep]); v.append(s[keep])
    return np.vstack(xyz), np.concatenate(v)
