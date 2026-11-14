# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Contact-based signed-distance constraints (deeprbf.workflow.contacts) on a synthetic tilted slab: a plane dipping
30 degrees toward 090 with ore within 8 m of it, cut by 64 vertical and inclined holes at 1 m composites. Contacts
must be found where the holes cross the slab faces, PCA normals from neighbouring holes must match the slab normal and
point from ore to waste, the off-surface points must carry the right signs, and a native linear-kernel RBF fitted to
the constraints (with the mid-interval points of interval_points) must put the slab mid-plane inside (negative) and
points 20 m away outside (positive)."""
import os, sys, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
from deeprbf.workflow.contacts import extract_contacts, estimate_normals, constraint_points, interval_points
from deeprbf.workflow.anisotropy import Frame

NRM = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])               # upward slab normal (dip 30 -> 090)
P0 = np.array([100.0, 100.0, 50.0]); HALF = 8.0

def sdist(xyz): return (np.asarray(xyz, float) - P0) @ NRM                           # signed distance to the mid-plane

def direction(dip, az):
    d, a = np.radians(dip), np.radians(az)
    return np.array([np.sin(a) * np.cos(d), np.cos(a) * np.cos(d), -np.sin(d)])

def synthetic(seed=0):
    r = np.random.default_rng(seed); rows = []; holes = {}
    kinds = [(90.0, 0.0), (60.0, 270.0), (60.0, 90.0), (70.0, 0.0)]
    for i, (x, y) in enumerate([(x, y) for x in np.arange(12.5, 200, 25.0) for y in np.arange(12.5, 200, 25.0)]):
        h = f'H{i:03d}'; c = np.array([x + r.uniform(-3, 3), y + r.uniform(-3, 3), 150.0]); d = direction(*kinds[i % 4])
        u = r.uniform(0, 1); holes[h] = (c, d, u)
        for f in u + np.arange(0, 320, 1.0):
            p = c + d * (f + 0.5); rows.append((h, f, f + 1, abs(sdist(p)) < HALF, *p))
    comp = pd.DataFrame(rows, columns=['hole', 'from', 'to', 'ore', 'X', 'Y', 'Z'])
    comp['value'] = np.where(comp.ore, 2.0, 0.05)
    return comp.drop(columns='ore'), holes

def expected(holes):
    """Exact crossing depths of the slab faces along each hole: top at s = +HALF, bottom at s = -HALF."""
    out = {}
    for h, (c, d, _) in holes.items():
        s0, rate = sdist(c), d @ NRM                                                    # rate < 0 for every hole here
        out[h] = dict(top=(s0 - HALF) / -rate, bottom=(s0 + HALF) / -rate)
    return out

def angle(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))

@pytest.fixture(scope='module')
def data():
    comp, holes = synthetic(); ore = comp.value >= 1.0
    c = extract_contacts(comp, ore)
    return comp, holes, c

def test_contacts_found_where_expected(data):
    comp, holes, c = data; ex = expected(holes)
    assert len(c) == 2 * len(holes)
    for h, g in c.groupby('hole'):
        assert sorted(g.kind) == ['bottom', 'top']
        for _, row in g.iterrows():
            t = ex[h][row.kind]; cc, d, _ = holes[h]
            assert abs(row.depth - t) < 1.0
            assert np.linalg.norm(row[['X', 'Y', 'Z']].to_numpy(float) - (cc + d * t)) < 1.0
            assert np.allclose(row[['dx', 'dy', 'dz']].to_numpy(float), d, atol=1e-9)
        assert g[g.kind == 'top'].depth.iloc[0] < g[g.kind == 'bottom'].depth.iloc[0]
    assert set(c.columns) >= {'hole', 'X', 'Y', 'Z', 'depth', 'kind', 'dx', 'dy', 'dz'}
    # unordered input and a column name for ore give the same contacts
    sh = comp.assign(ore=comp.value >= 1.0).sample(frac=1.0, random_state=1)
    c2 = extract_contacts(sh, 'ore')
    pd.testing.assert_frame_equal(c.sort_values(['hole', 'depth']).reset_index(drop=True), c2.sort_values(['hole', 'depth']).reset_index(drop=True))

def test_max_gap_skips_contacts_across_missing_samples(data):
    comp, holes, c = data
    top = c[(c.hole == 'H010') & (c.kind == 'top')].depth.iloc[0]
    mid = (comp['from'] + comp['to']) / 2
    gapped = comp[~((comp.hole == 'H010') & ((mid - top).abs() < 1.6))]                 # drop the composites around it
    ore = gapped.value >= 1.0
    assert len(extract_contacts(gapped, ore)) == len(c)                                 # still a contact (mid of the gap)
    cg = extract_contacts(gapped, ore, max_gap=0.5)
    assert len(cg) == len(c) - 1 and not ((cg.hole == 'H010') & (cg.kind == 'top')).any()

def test_normals_match_slab_and_point_to_waste(data):
    _, holes, c = data
    true = np.where((c.kind == 'top').to_numpy()[:, None], NRM, -NRM)                  # ore -> waste
    for frame in (None, Frame(30.0, 90.0, 90.0, [60.0, 60.0, 10.0], P0)):
        n, info = estimate_normals(c, k=8, frame=frame)
        assert n.shape == (len(c), 3) and np.allclose(np.linalg.norm(n, axis=1), 1)
        interior = ((c.X > 30) & (c.X < 170) & (c.Y > 30) & (c.Y < 170)).to_numpy()
        err = angle(n, true)
        assert interior.sum() > 40 and err[interior].max() < 10, err[interior].max()
        assert (np.sum(n * true, axis=1) > 0).all()                                     # every normal oriented ore -> waste
        assert not info.fallback.any() and (info.planarity > 0.9).all() and (info.neighbours == 8).all()
    # the hole axis (what the PCA replaces) is much worse here: 30 degrees off for the vertical holes
    axis = np.where((c.kind == 'top').to_numpy()[:, None], -1, 1) * c[['dx', 'dy', 'dz']].to_numpy(float)
    assert angle(axis, true).mean() > 3 * err.mean()

def test_fallback_to_hole_axis_when_isolated(data):
    _, _, c = data
    one = c[c.hole.isin(['H000', 'H001'])].reset_index(drop=True)                       # 1 other hole -> < 3 neighbours
    n, info = estimate_normals(one, k=8)
    assert info.fallback.all() and (info.neighbours == 1).all()
    d = one[['dx', 'dy', 'dz']].to_numpy(float)
    assert np.allclose(n, np.where((one.kind == 'top').to_numpy()[:, None], -d, d))
    # same_kind=False lets top and bottom contacts be neighbours: the two faces are 16 m apart, so planarity drops
    n2, info2 = estimate_normals(c, k=8, same_kind=False)
    assert info2.planarity.median() < estimate_normals(c, k=8)[1].planarity.median()

def test_constraint_point_signs(data):
    _, _, c = data; n, _ = estimate_normals(c, k=8); delta = 2.0
    xyz, v = constraint_points(c, n, delta)
    assert xyz.shape == (3 * len(c), 3) and v.shape == (3 * len(c),)
    assert np.allclose(xyz[:len(c)], c[['X', 'Y', 'Z']].to_numpy(float)) and (v[:len(c)] == 0).all()
    off = v != 0; inside = np.abs(sdist(xyz[off])) < HALF
    assert ((v[off] < 0) == inside).all()                                               # negative exactly on the ore side
    assert np.allclose(np.abs(v[off]), delta)

def test_prune_drops_offsets_crossing_thin_ore():
    c = pd.DataFrame(dict(hole=['A', 'A'], X=[0.0, 0.0], Y=[0.0, 0.0], Z=[0.0, -2.0], depth=[10.0, 12.0], kind=['top', 'bottom'],
                          dx=[0.0, 0.0], dy=[0.0, 0.0], dz=[-1.0, -1.0], gap=[0.0, 0.0]))
    n = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    xyz, v = constraint_points(c, n, 3.0)                                               # 2 m of ore, delta 3: ore-side points cross
    assert len(v) == 4 and (v[2:] > 0).all()
    assert len(constraint_points(c, n, 3.0, prune=False)[1]) == 6
    with pytest.raises(ValueError): constraint_points(c, n, 0.0)

def test_interval_points(data):
    _, holes, c = data; n, _ = estimate_normals(c, k=8)
    xi, vi = interval_points(c, n)
    assert len(vi) == len(holes) and (vi < 0).all()                                     # one ore interval per hole
    assert np.allclose(vi, -HALF, atol=0.6) and np.allclose(sdist(xi), 0, atol=0.6)    # half thickness, at the mid-plane
    assert (interval_points(c, n, max_half=3.0)[1] >= -3.0).all()

def fit_predict(native, xyz, v):
    m = native.NativeExactRBFImplicitModel('linear', 1.0, 0.0); m.set_drift('linear'); m.fit(xyz.tolist(), v.tolist())
    assert np.allclose(np.asarray(m.predict(xyz.tolist())), v, atol=1e-5)               # exact interpolation
    r = np.random.default_rng(7); xy = r.uniform(15, 185, size=(400, 2))
    mid = np.c_[xy, P0[2] - (NRM[0] * (xy[:, 0] - P0[0])) / NRM[2]]                     # held-out, on the mid-plane
    far = np.vstack([mid + 20 * NRM, mid - 20 * NRM])                                   # held-out, 20 m from it
    return (np.asarray(m.predict(mid.tolist())) < 0).mean(), (np.asarray(m.predict(far.tolist())) > 0).mean()

def test_native_signed_distance_fit(data):
    native = pytest.importorskip('deeprbf_recovered')
    _, _, c = data; n, _ = estimate_normals(c, k=8); xyz, v = constraint_points(c, n, 2.0)
    inside, outside = fit_predict(native, xyz, v)                                       # contacts + off-surface points only
    assert inside > 0.85 and outside > 0.95, (inside, outside)                          # mid-ore relaxes between holes
    xi, vi = interval_points(c, n)
    inside, outside = fit_predict(native, np.vstack([xyz, xi]), np.r_[v, vi])
    assert inside > 0.95 and outside > 0.95, (inside, outside)
