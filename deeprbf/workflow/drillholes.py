# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Drillhole intake for workflows: collar, survey and interval tables from CSV with a column map, minimum-curvature
desurvey (dip positive down by default), and fixed-length length-weighted compositing that breaks at hole ends."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..desurvey import minimum_curvature

DEFAULT_COLUMNS = dict(hole='BHID', x='XCOLLAR', y='YCOLLAR', z='ZCOLLAR', depth='AT', dip='DIP', azimuth='BRG',
                       from_='FROM', to='TO')

def load_tables(collar_csv, survey_csv, interval_csv, value, columns=None):
    c = {**DEFAULT_COLUMNS, **(columns or {})}
    col = pd.read_csv(collar_csv); sur = pd.read_csv(survey_csv); itv = pd.read_csv(interval_csv)
    need = {'collar': (col, [c['hole'], c['x'], c['y'], c['z']]), 'survey': (sur, [c['hole'], c['depth'], c['dip'], c['azimuth']]),
            'intervals': (itv, [c['hole'], c['from_'], c['to'], value])}
    for name, (df, cols) in need.items():
        miss = [k for k in cols if k not in df.columns]
        if miss: raise ValueError(f'{name} table lacks columns {miss}; set them in "columns"')
    col = col.rename(columns={c['hole']: 'hole', c['x']: 'x', c['y']: 'y', c['z']: 'z'})[['hole', 'x', 'y', 'z']]
    sur = sur.rename(columns={c['hole']: 'hole', c['depth']: 'depth', c['dip']: 'dip', c['azimuth']: 'azimuth'})[['hole', 'depth', 'dip', 'azimuth']]
    itv = itv.rename(columns={c['hole']: 'hole', c['from_']: 'from', c['to']: 'to', value: 'value'})[['hole', 'from', 'to', 'value']]
    for df in (col, sur, itv): df['hole'] = df['hole'].astype(str)
    return col, sur, itv

def composite(intervals, length=1.0, min_fraction=0.5):
    """Fixed-length composites per hole from the first sampled depth; length-weighted mean of the sampled part;
    a composite with less than min_fraction of its length sampled is dropped. Missing values are unsampled."""
    rows = []
    for hole, g in intervals.dropna(subset=['value']).sort_values(['hole', 'from']).groupby('hole', sort=False):
        a, b, v = g['from'].to_numpy(float), g['to'].to_numpy(float), g['value'].to_numpy(float)
        ok = b > a; a, b, v = a[ok], b[ok], v[ok]
        if not len(a): continue
        edges = np.arange(a.min(), b.max() + 1e-9, length)
        if edges[-1] < b.max() - 1e-9: edges = np.r_[edges, edges[-1] + length]
        for lo, hi in zip(edges[:-1], edges[1:]):
            ov = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None); w = ov.sum()
            if w >= min_fraction * (hi - lo): rows.append((hole, lo, hi, (ov * v).sum() / w, w))
    return pd.DataFrame(rows, columns=['hole', 'from', 'to', 'value', 'sampled'])

def desurvey(collar, survey, composites):
    """XYZ at composite mid-depths by minimum curvature; holes without a collar or survey are reported and dropped."""
    out = []; skipped = []
    over = (survey.dip.abs() > 90) & (survey.dip.abs() <= 90 + 1e-3)                 # rounding just past vertical, e.g. 90.000002
    if (survey.dip.abs() > 90 + 1e-3).any(): raise ValueError('survey dips beyond +/-90 degrees')
    survey = survey.assign(dip=survey.dip.clip(-90, 90)); desurvey.dips_clipped = int(over.sum())
    col = collar.set_index('hole'); sur = dict(tuple(survey.groupby('hole')))
    for hole, g in composites.groupby('hole', sort=False):
        if hole not in col.index or hole not in sur: skipped.append(hole); continue
        s = sur[hole].drop_duplicates('depth'); mid = ((g['from'] + g['to']) / 2).to_numpy(float)
        xyz = minimum_curvature(col.loc[hole, ['x', 'y', 'z']].to_numpy(float), s.depth.to_numpy(float), s.dip.to_numpy(float), s.azimuth.to_numpy(float), mid)
        out.append(g.assign(X=xyz[:, 0], Y=xyz[:, 1], Z=xyz[:, 2]))
    return (pd.concat(out, ignore_index=True) if out else composites.iloc[:0].assign(X=[], Y=[], Z=[])), skipped
