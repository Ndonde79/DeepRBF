# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Rank auditable alternative intersection correlations; no orebody truth claim."""
import numpy as np
from scipy.spatial import cKDTree

def propose_correlations(xyz, holes, families, formations, prospects, statuses,
                         before, after, normals=None, ranges=(150.,50.,100.)):
    x=np.asarray(xyz,float);n=len(x);r=np.asarray(ranges,float)
    arrays=[np.asarray(v).astype(str) for v in [holes,families,formations,prospects,statuses,before,after]]
    if x.shape!=(n,3) or not np.isfinite(x).all() or r.shape!=(3,) or not np.isfinite(r).all() or (r<=0).any() or any(v.shape!=(n,) for v in arrays):raise ValueError('Finite XYZ and aligned metadata required')
    h,f,form,pro,status,bc,ac=arrays
    normal=np.full_like(x,np.nan) if normals is None else np.asarray(normals,float)
    if normal.shape!=x.shape:raise ValueError('Normal shape mismatch')
    edges=[]
    for i,j in cKDTree(x/r).query_pairs(1,output_type='ndarray'):
        if h[i]==h[j] or f[i]!=f[j]:continue
        if form[i] in ['','UNKNOWN','nan'] or form[j] in ['','UNKNOWN','nan']:form_evidence='UNKNOWN'
        elif form[i]!=form[j]:continue
        else:form_evidence='MATCH'
        if pro[i]!=pro[j]:continue
        delta=x[j]-x[i];distance=float(np.linalg.norm(delta));scaled=float(np.linalg.norm(delta/r))
        orientation='UNRESOLVED';angle=np.nan;normal_distance=np.nan;alignment=.5
        if np.isfinite(normal[[i,j]]).all() and np.all(np.linalg.norm(normal[[i,j]],axis=1)>.5):
            na=normal[i]/np.linalg.norm(normal[i]);nb=normal[j]/np.linalg.norm(normal[j]);dot=abs(float(na@nb));angle=float(np.degrees(np.arccos(np.clip(dot,0,1))))
            normal_distance=float(max(abs(delta@na),abs(delta@nb)));alignment=dot*np.exp(-normal_distance/20);orientation='LOCAL_GEOMETRY'
        context_i={bc[i],ac[i]}-{'','None','nan','UNKNOWN'};context_j={bc[j],ac[j]}-{'','None','nan','UNKNOWN'}
        context=float(len(context_i&context_j)/len(context_i|context_j)) if context_i|context_j else .5
        complete=int(status[i]=='COMPLETE_LOGGED_CROSSING')+int(status[j]=='COMPLETE_LOGGED_CROSSING')
        ambiguous='AMBIGUOUS_OVERLAP' in [status[i],status[j]]
        score=.4*(1-scaled)+.3*alignment+.2*context+.1*(complete/2)
        if form_evidence=='UNKNOWN':score*=.7
        if ambiguous:score=0.
        edges.append(dict(a=int(i),b=int(j),distance_m=distance,scaled_distance=scaled,orientation_evidence=orientation,normal_angle_deg=angle,normal_separation_m=normal_distance,flank_code_similarity=context,complete_endpoints=complete,formation_evidence=form_evidence,ambiguous_overlap=ambiguous,score=float(score)))
    # Alternative matches to the SAME other hole remain visible; avoid hiding ambiguity.
    groups={}
    for index,e in enumerate(edges):
        for node,other in [(e['a'],e['b']),(e['b'],e['a'])]:groups.setdefault((node,h[other]),[]).append(index)
    ranks={};margins={}
    for key,ids in groups.items():
        ids=sorted(ids,key=lambda k:(-edges[k]['score'],k));margin=edges[ids[0]]['score']-edges[ids[1]]['score'] if len(ids)>1 else np.nan
        for rank,index in enumerate(ids,1):ranks[(key[0],index)]=rank;margins[(key[0],index)]=margin
    for i,e in enumerate(edges):
        e['rank_a']=ranks[(e['a'],i)];e['rank_b']=ranks[(e['b'],i)];e['best_alternative_margin_a']=margins[(e['a'],i)];e['best_alternative_margin_b']=margins[(e['b'],i)]
    return edges
