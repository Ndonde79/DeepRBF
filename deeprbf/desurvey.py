# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Minimum-curvature drillhole desurveying; positive dip down."""
import numpy as np
import pandas as pd


def minimum_curvature(collar_xyz,survey_depths,dips_positive_down,azimuths,query_depths):
    """Return world XYZ at measured depths using minimum-curvature integration.

    Partial survey segments use spherical tangent interpolation. Beyond the last
    station the final tangent is extended. A first station deeper than zero is
    extended back to the collar. Duplicate depths and invalid geometry are rejected.
    """
    collar=np.asarray(collar_xyz,float);depth=np.asarray(survey_depths,float)
    dips=np.asarray(dips_positive_down,float);azimuth=np.asarray(azimuths,float);query=np.asarray(query_depths,float)
    if collar.shape!=(3,) or not np.isfinite(collar).all():raise ValueError('Invalid collar XYZ')
    if depth.ndim!=1 or not len(depth) or dips.shape!=depth.shape or azimuth.shape!=depth.shape:raise ValueError('Survey arrays must have matching lengths')
    if not np.isfinite(depth).all() or not np.isfinite(dips).all() or not np.isfinite(azimuth).all() or (depth<0).any() or (abs(dips)>90).any():raise ValueError('Invalid survey values')
    if len(np.unique(depth))!=len(depth):raise ValueError('Duplicate survey depths')
    if query.ndim!=1 or not np.isfinite(query).all() or (query<0).any():raise ValueError('Invalid query depths')
    if not len(query):return np.empty((0,3))
    frame=pd.DataFrame(dict(depth=depth,dip_positive_down=dips,azimuth=azimuth))
    ordered=frame.sort_values('depth');ts=np.array([tangent(d,a) for d,a in zip(ordered.dip_positive_down,ordered.azimuth)])
    if len(ts)>1 and (np.sum(ts[1:]*ts[:-1],axis=1)<-1+1e-10).any():raise ValueError('Antipodal survey tangents are ambiguous')
    c=pd.Series(dict(easting=collar[0],northing=collar[1],elevation=collar[2]))
    return trajectory(c,frame,query)

def tangent(dip,azimuth):
    d,a=np.deg2rad([dip,azimuth]);return np.array([np.sin(a)*np.cos(d),np.cos(a)*np.cos(d),-np.sin(d)])

def arc(displacement,t1,t2):
    angle=np.arccos(np.clip(np.dot(t1,t2),-1.,1.))
    ratio=1. if angle<1e-10 else 2*np.tan(angle/2)/angle
    return displacement*(t1+t2)*ratio/2

def trajectory(collar,survey,depths):
    s=survey.sort_values('depth')
    md=s.depth.to_numpy(float); tangents=np.array([tangent(d,a) for d,a in zip(s.dip_positive_down,s.azimuth)])
    if md[0]>0: md=np.r_[0.,md];tangents=np.vstack([tangents[0],tangents])
    positions=[np.array([collar.easting,collar.northing,collar.elevation],float)]
    for i in range(1,len(md)): positions.append(positions[-1]+arc(md[i]-md[i-1],tangents[i-1],tangents[i]))
    result=[]
    for depth in depths:
        i=max(0,np.searchsorted(md,depth,side='right')-1)
        if i==len(md)-1: point=positions[i]+(depth-md[i])*tangents[i]
        else:
            fraction=(depth-md[i])/(md[i+1]-md[i]);angle=np.arccos(np.clip(tangents[i]@tangents[i+1],-1,1))
            if angle<1e-10: partial=tangents[i]
            else: partial=(np.sin((1-fraction)*angle)*tangents[i]+np.sin(fraction*angle)*tangents[i+1])/np.sin(angle)
            point=positions[i]+arc(depth-md[i],tangents[i],partial)
        result.append(point)
    return np.array(result)
