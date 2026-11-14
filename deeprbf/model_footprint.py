# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Explicit non-rectangular model limits from caller-selected drill traces."""
import numpy as np

def drilling_footprint(xy, *, ratio=.15, buffer_m=25., simplify_m=5.):
    from shapely import concave_hull,MultiPoint,Polygon,covers,points
    xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():raise ValueError('At least three finite XY points required')
    if not 0<=ratio<=1 or buffer_m<=0 or not 0<=simplify_m<buffer_m:raise ValueError('Invalid footprint controls')
    raw=concave_hull(MultiPoint(xy),ratio=ratio,allow_holes=False)
    if raw.geom_type!='Polygon':raise ValueError('Drilling must span a non-collinear footprint')
    polygon=raw.buffer(buffer_m).simplify(simplify_m,preserve_topology=True)
    polygon=Polygon(polygon.exterior)
    if not polygon.is_valid or not covers(polygon,points(xy)).all():raise ValueError('Footprint must be valid and contain every selected point')
    return polygon
