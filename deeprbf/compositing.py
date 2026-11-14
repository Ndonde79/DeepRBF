# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Boundary-respecting length composites with explicit gaps and overlaps."""
import numpy as np


def interval_average(start,stop,values,left,right):
    start=np.asarray(start,float);stop=np.asarray(stop,float);values=np.asarray(values,float)
    if start.shape!=stop.shape or start.shape!=values.shape or start.ndim!=1:
        raise ValueError('Matching interval arrays required')
    if not np.isfinite([left,right]).all() or right<=left:raise ValueError('Invalid output interval')
    if not np.isfinite(start).all() or not np.isfinite(stop).all() or np.any(stop<=start):raise ValueError('Invalid source intervals')
    lengths=np.maximum(0,np.minimum(stop,right)-np.maximum(start,left))
    selected=lengths>1e-9
    order=np.argsort(start[selected]);a=np.maximum(start[selected][order],left);b=np.minimum(stop[selected][order],right)
    overlap=bool(len(a)>1 and np.any(a[1:]<np.maximum.accumulate(b)[:-1]-1e-8))
    valid=selected&np.isfinite(values);coverage=float(lengths[valid].sum())
    estimate=float(np.dot(lengths[valid],values[valid])/coverage) if coverage>0 and not overlap else np.nan
    return estimate,coverage,overlap


def partition_interval(left,right,length=2.,boundaries=()):
    if not np.isfinite([left,right,length]).all() or right<=left or length<=0:raise ValueError('Invalid partition')
    edges=np.unique(np.r_[left,[x for x in boundaries if left<x<right],right])
    output=[]
    for a,b in zip(edges[:-1],edges[1:]):
        points=np.r_[np.arange(a,b,length),b]
        output.extend((float(x),float(y)) for x,y in zip(points[:-1],points[1:]) if y-x>1e-8)
    return output
