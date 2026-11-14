# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Field grids to closed wireframes. Small bodies are removed on the voxel grid BEFORE contouring (removing mesh pieces
afterwards cuts holes in closed surfaces). Clipping by box and by a topography surface is applied to the field, so
surfaces close against the clip. Readers/writers: Surpac .str/.dtm, DXF 3DFACE. Point-in-mesh by vertical ray parity."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label, generate_binary_structure

def grid_axes(lo, hi, spacing):
    return [np.arange(l, h + 1e-9, spacing) for l, h in zip(lo, hi)]

def evaluate(predict, lo, hi, spacing, slab=1):
    """predict: function (n,3 world) -> (n,) values. Evaluated one x-slab at a time to bound memory."""
    ax = grid_axes(lo, hi, spacing); f = np.empty([len(a) for a in ax])
    for i in range(0, len(ax[0]), slab):
        G = np.stack(np.meshgrid(ax[0][i:i + slab], ax[1], ax[2], indexing='ij'), -1)
        f[i:i + slab] = np.asarray(predict(G.reshape(-1, 3))).reshape(G.shape[:3])
    return f, ax

def clip_topography(f, ax, topo_xyz, iso):
    """Set the field below iso above a topography given as scattered XYZ points (linear interpolation, nearest outside)."""
    from scipy.interpolate import griddata
    gx, gy = np.meshgrid(ax[0], ax[1], indexing='ij'); xy = np.c_[gx.ravel(), gy.ravel()]; t = np.asarray(topo_xyz, float)
    z = griddata(t[:, :2], t[:, 2], xy, 'linear'); bad = np.isnan(z)
    if bad.any(): z[bad] = griddata(t[:, :2], t[:, 2], xy[bad], 'nearest')
    above = ax[2][None, None, :] > z.reshape(gx.shape)[..., None]
    return np.where(above, np.minimum(f, iso - 1.0), f)

def filter_small(f, spacing, iso, min_volume):
    inside = f >= iso; lab, nb = label(inside, generate_binary_structure(3, 3))
    vol = np.bincount(lab.ravel()) * spacing ** 3; vol[0] = 0; keep = vol >= min_volume
    g = np.where(keep[lab] | ~inside, f, np.minimum(f, iso - 1e-3))
    return g, dict(bodies_found=int(nb), bodies_kept=int(keep[1:].sum()), bodies_removed=int(nb - keep[1:].sum()), body_volumes_m3=sorted(vol[1:][keep[1:]].round(1).tolist(), reverse=True))

def contour(f, ax, iso):
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:
        raise ImportError('Contouring needs scikit-image (pip install scikit-image)') from e
    h = ax[0][1] - ax[0][0]; fp = np.pad(f, 1, constant_values=min(float(np.nanmin(f)), iso) - 1.0)   # closes surfaces at the grid faces
    V, F, _, _ = marching_cubes(fp, iso, spacing=(h, ax[1][1] - ax[1][0], ax[2][1] - ax[2][0]))
    return V + np.array([a[0] for a in ax]) - np.array([h, ax[1][1] - ax[1][0], ax[2][1] - ax[2][0]]), F

def volume(V, F):
    T = V[F]; return float(abs(np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6))

def write_surpac(V, F, path_stem, name='deeprbf'):
    with open(path_stem + '.str', 'w') as s:
        s.write(f'{name},,,\n0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000\n')
        for x, y, z in V: s.write(f'1,{y:.3f},{x:.3f},{z:.3f},\n')                 # Surpac: Y, X, Z
        s.write('0, 0.000, 0.000, 0.000,\n0, 0.000, 0.000, 0.000, END\n')
    stem = path_stem.replace('\\', '/').split('/')[-1]
    with open(path_stem + '.dtm', 'w') as d:
        d.write(f'{stem}.str,\n0, 0.000, 0.000, 0.000, END\nOBJECT, 1,\nTRISOLATION, 1,\n')
        for i, (a, b, c) in enumerate(F, 1): d.write(f'{i}, {a + 1}, {b + 1}, {c + 1}, 0, 0, 0,\n')
        d.write('END\n')

def write_dxf(V, F, path, layer='SHELL'):
    with open(path, 'w') as d:
        d.write('0\nSECTION\n2\nENTITIES\n')
        for t in V[F]:
            d.write(f'0\n3DFACE\n8\n{layer}\n')
            for k, (x, y, z) in enumerate([t[0], t[1], t[2], t[2]]): d.write(f'1{k}\n{x:.3f}\n2{k}\n{y:.3f}\n3{k}\n{z:.3f}\n')
        d.write('0\nENDSEC\n0\nEOF\n')

def read_surpac(dtm, strf):
    P = []
    for line in open(strf, errors='ignore').read().splitlines()[2:]:
        f = line.split(',')
        if len(f) >= 4 and f[0].strip() not in ('0', ''): P.append((float(f[2]), float(f[1]), float(f[3])))
    T = [tuple(int(x) for x in f[1:4]) for f in ([v.strip() for v in l.split(',')] for l in open(dtm, errors='ignore').read().splitlines()) if len(f) >= 4 and f[0].isdigit() and f[1].isdigit()]
    return np.array(P)[np.array(T) - 1]

def read_dxf(path):
    L = open(path, errors='ignore').read().splitlines(); out = []; i = 0
    while i < len(L):
        if L[i].strip() == '3DFACE':
            v = {}; i += 1
            while i < len(L) - 1 and L[i].strip() != '0':
                code, val = L[i].strip(), L[i + 1].strip(); i += 2
                if code in ('10', '20', '30', '11', '21', '31', '12', '22', '32'): v[code] = float(val)
            if len(v) == 9: out.append([[v['10'], v['20'], v['30']], [v['11'], v['21'], v['31']], [v['12'], v['22'], v['32']]])
        else: i += 1
    return np.array(out)

def read_wireframe(path):
    if path.lower().endswith('.dxf'): return read_dxf(path)
    stem = path[:-4]; return read_surpac(stem + '.dtm', stem + '.str')

def inside(T, P, cell=10.0):
    """Vertical ray parity test of points P against closed triangles T (n,3,3)."""
    T = T[np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0])) > 1e-12]
    P = np.asarray(P, float); lo = P[:, :2].min(0) - cell; g = lambda xy: np.floor((xy - lo) / cell).astype(int)
    tmin = g(T[:, :, :2].min(1)); tmax = g(T[:, :, :2].max(1)); grid = {}
    for k in range(len(T)):
        for ix in range(tmin[k, 0], tmax[k, 0] + 1):
            for iy in range(tmin[k, 1], tmax[k, 1] + 1): grid.setdefault((ix, iy), []).append(k)
    pc = g(P[:, :2]); res = np.zeros(len(P), bool); order = np.lexsort((pc[:, 1], pc[:, 0])); s = pc[order]
    for grp in np.split(order, np.flatnonzero(np.any(np.diff(s, axis=0) != 0, axis=1)) + 1):
        tri = grid.get(tuple(pc[grp[0]]))
        if not tri: continue
        A = T[tri]; p = P[grp]; a, b, c = A[:, 0], A[:, 1], A[:, 2]
        v0 = (c - a)[:, :2]; v1 = (b - a)[:, :2]; v2 = p[:, None, :2] - a[None, :, :2]
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); den = d00 * d11 - d01 * d01
        d02 = (v2 * v0[None]).sum(2); d12 = (v2 * v1[None]).sum(2); u = (d11 * d02 - d01 * d12) / den; w = (d00 * d12 - d01 * d02) / den
        hit = (u >= 0) & (w >= 0) & (u + w <= 1); z = a[None, :, 2] + u * (c[None, :, 2] - a[None, :, 2]) + w * (b[None, :, 2] - a[None, :, 2])
        res[grp] = ((hit & (z > p[:, None, 2])).sum(1) % 2) == 1
    return res
