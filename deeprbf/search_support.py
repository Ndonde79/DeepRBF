# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Grade-blind support diagnostics for declared search radii."""
import numpy as np
from scipy.spatial import cKDTree


def search_support(train_xyz, train_holes, query_xyz, radii, *, min_holes, min_samples, max_per_hole, candidates=128):
    x=np.asarray(train_xyz,float)
    q=np.asarray(query_xyz,float)
    h=np.asarray(train_holes)
    r=np.asarray(radii,float)
    if (x.ndim!=2 or x.shape[1]!=3 or q.ndim!=2 or q.shape[1]!=3 or
        h.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError('Valid coordinates and training holes required')
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all() or (r<=0).any() or (np.diff(r)<=0).any():
        raise ValueError('Positive increasing search radii required')
    if any(type(k) is not int or k<1 for k in [min_holes,min_samples,max_per_hole,candidates]):
        raise ValueError('Positive search controls required')
    if candidates<max(min_samples,min_holes):
        raise ValueError('Candidate budget is smaller than the support requirement')
    result=np.zeros((len(q),len(r)),bool)
    clipped=np.zeros(len(q),bool)
    if not len(x) or not len(q):
        return result,clipped
    tree=cKDTree(x)
    _,hid=np.unique(h,return_inverse=True)
    k=min(candidates,len(x))
    for a in range(0,len(q),2048):
        v=q[a:a+2048]
        probe=min(k+1,len(x))
        d,i=tree.query(v,k=probe,distance_upper_bound=r[-1],workers=4)
        d,i=d.reshape(len(v),probe),i.reshape(len(v),probe)
        if probe>k:
            clipped[a:a+len(v)]=np.isfinite(d[:,k])
        d,i=d[:,:k],i[:,:k]
        holes=hid[np.minimum(i,len(hid)-1)]
        order=np.argsort(holes,axis=1,kind='stable')
        sorted_holes=np.take_along_axis(holes,order,axis=1)
        new=np.c_[np.ones(len(v),bool),np.diff(sorted_holes,axis=1)!=0]
        pos=np.broadcast_to(np.arange(k),(len(v),k))
        rank_sorted=pos-np.maximum.accumulate(np.where(new,pos,0),axis=1)+1
        rank=np.take_along_axis(rank_sorted,np.argsort(order,axis=1),axis=1)
        for j,radius in enumerate(r):
            ok=d<=radius
            n=(ok&(rank<=max_per_hole)).sum(1)
            nh=(ok&(rank==1)).sum(1)
            result[a:a+len(v),j]=(n>=min_samples)&(nh>=min_holes)
    return result,clipped
