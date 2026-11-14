# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Experimental contact-interpolating paired surfaces in a component frame."""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree,Delaunay

class PairedLens:
    def fit(self, starts, ends, holes, start_observed=None, end_observed=None):
        a=np.asarray(starts,float);b=np.asarray(ends,float);h=np.asarray(holes).astype(str)
        if a.ndim!=2 or a.shape[1]!=3 or b.shape!=a.shape or h.shape!=(len(a),) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite paired interval geometry required')
        sa=np.ones(len(a),bool) if start_observed is None else np.asarray(start_observed,bool);sb=np.ones(len(a),bool) if end_observed is None else np.asarray(end_observed,bool)
        if sa.shape!=h.shape or sb.shape!=h.shape:raise ValueError('Contact flags must match intervals')
        complete=sa&sb
        if len(set(h[complete]))<3:raise ValueError('Three complete-crossing holes required')
        center=(a[complete]+b[complete])/2;self.origin=center.mean(axis=0);_,singular,axes=np.linalg.svd(center-self.origin,full_matrices=False)
        if len(singular)<3 or singular[1]<1 or singular[1]<1.5*singular[2]:raise ValueError('Unresolved reference plane')
        self.axes=axes;aa=(a-self.origin)@axes.T;bb=(b-self.origin)@axes.T
        lower_a=aa[:,2]<bb[:,2];constraints=[[],[]];ids=[[],[]]
        for i in range(len(a)):
            if sa[i]:j=0 if lower_a[i] else 1;constraints[j].append(aa[i]);ids[j].append(h[i])
            if sb[i]:j=1 if lower_a[i] else 0;constraints[j].append(bb[i]);ids[j].append(h[i])
        self.models=[];self.constraints=[];self.removed_conflicts=0;errors=[]
        for points in constraints:
            p=np.array(points)
            if len(p)<3:raise ValueError('Insufficient wall constraints')
            uv=np.round(p[:,:2],6);_,inverse=np.unique(uv,axis=0,return_inverse=True);keep=[]
            for group in np.unique(inverse):
                ix=np.flatnonzero(inverse==group)
                if np.ptp(p[ix,2])>1e-4:self.removed_conflicts+=len(ix)
                else:keep.append(ix[0])
            p=p[keep]
            if len(p)<3:raise ValueError('Conflicting projected contacts')
            model=RBFInterpolator(p[:,:2],p[:,2],kernel='thin_plate_spline',degree=1,smoothing=0,neighbors=min(32,len(p)))
            error=float(np.max(abs(model(p[:,:2])-p[:,2])))
            if error>1e-5:raise ValueError('Contact interpolation tolerance failed')
            self.models.append(model);self.constraints.append(p);errors.append(error)
        self.max_contact_error=max(errors);allp=np.concatenate(self.constraints);self.hull=Delaunay(allp[:,:2]);self.tree=cKDTree(center);self.holes=h[complete]
        self.hole_trees=[cKDTree(center[self.holes==hole]) for hole in np.unique(self.holes)]
        self.center=center.mean(axis=0);self.radius=float(np.linalg.norm(center-self.center,axis=1).max()+150)
        return self

    def evaluate(self, xyz):
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite XYZ required')
        local=(x-self.origin)@self.axes.T;inside=self.hull.find_simplex(local[:,:2])>=0
        # Count distinct holes exactly: dense intervals from one hole must not
        # hide nearby supporting holes behind a fixed nearest-point cap.
        nh=np.zeros(len(x),int)
        for tree in self.hole_trees:nh+=(tree.query(x)[0]<=150)
        support=inside&(nh>=3);low=np.full(len(x),np.nan);high=low.copy()
        if support.any():low[support]=self.models[0](local[support,:2]);high[support]=self.models[1](local[support,:2])
        crossed=support&(high<=low);support&=~crossed
        member=support&(local[:,2]>=low)&(local[:,2]<=high)
        residual=np.minimum(abs(local[:,2]-low),abs(local[:,2]-high))
        return dict(member=member,supported=support,crossed=crossed,boundary_distance=residual,lower=low,upper=high,normal_coordinate=local[:,2])
