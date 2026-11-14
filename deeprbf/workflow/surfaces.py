# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Topography (or any surface) as an XYZ point cloud from the formats found in mine databases: DXF (3DFACE, POLYLINE
VERTEX, LWPOLYLINE with elevation, POINT, LINE), Surpac .str (points) or .dtm (its paired .str), CSV (columns X/Y/Z,
XP/YP/ZP, EAST/NORTH/RL or similar). Datamine .dm: read through a registered reader (see register_reader)."""
from __future__ import annotations
import os, numpy as np, pandas as pd

_READERS = {}
def register_reader(ext, fn):
    """Add a reader for another format, fn(path) -> (n,3) XYZ, e.g. register_reader('.dm', my_datamine_points)."""
    _READERS[ext.lower()] = fn

def _dxf_points(path):
    """Every 3D coordinate in the ENTITIES section: code pairs 1x/2x/3x (3DFACE corners, VERTEX, POINT, LINE ends),
    plus LWPOLYLINE vertices (10/20) at the polyline elevation (38)."""
    L = [l.strip() for l in open(path, errors='ignore').read().splitlines()]; pts = []; xy = {}; ent = None; z38 = 0.0; lw = []
    for code, val in zip(L[0::2], L[1::2]):
        if code == '0':
            if ent == 'LWPOLYLINE': pts += [(x, y, z38) for x, y in lw]
            ent, xy, z38, lw = val, {}, 0.0, []
            continue
        if code == '38': z38 = float(val); continue
        if len(code) == 2 and code[1] in '0123' and code[0] in '123':
            k = int(code[1]); v = float(val)
            if code[0] == '1': xy[k] = [v]
            elif code[0] == '2' and k in xy:
                xy[k].append(v)
                if ent == 'LWPOLYLINE' and k == 0: lw.append(tuple(xy[k]))
            elif code[0] == '3' and k in xy and len(xy[k]) == 2: pts.append((xy[k][0], xy[k][1], v)); del xy[k]
    P = np.array(pts, float)
    return np.unique(P.round(4), axis=0) if len(P) else P

def _surpac_points(path):
    if path.lower().endswith('.dtm'): path = path[:-4] + '.str'
    P = []
    for line in open(path, errors='ignore').read().splitlines()[2:]:
        f = line.split(',')
        if len(f) >= 4 and f[0].strip() not in ('0', ''):
            try: P.append((float(f[2]), float(f[1]), float(f[3])))
            except ValueError: pass
    return np.array(P, float)

def _csv_points(path):
    t = pd.read_csv(path); up = {c.upper().strip(): c for c in t.columns}
    for xs, ys, zs in (('X', 'Y', 'Z'), ('XP', 'YP', 'ZP'), ('EAST', 'NORTH', 'RL'), ('EASTING', 'NORTHING', 'ELEVATION'), ('XC', 'YC', 'ZC')):
        if xs in up and ys in up and zs in up: return t[[up[xs], up[ys], up[zs]]].to_numpy(float)
    raise ValueError(f'{path}: no X/Y/Z-type columns found ({list(t.columns)[:8]})')

def read_points(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in _READERS: return np.asarray(_READERS[ext](path), float)
    if ext == '.dxf': P = _dxf_points(path)
    elif ext in ('.str', '.dtm'): P = _surpac_points(path)
    elif ext in ('.csv', '.txt'): P = _csv_points(path)
    elif ext == '.dm': raise ValueError('Datamine .dm: register a reader with surfaces.register_reader(".dm", fn) (Kilamongo study dmread.py)')
    else: raise ValueError(f'Unsupported surface format: {ext}')
    if len(P) < 3: raise ValueError(f'{path}: fewer than 3 surface points read')
    return P
