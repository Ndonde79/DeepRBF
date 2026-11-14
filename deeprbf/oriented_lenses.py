# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Experimental locally oriented paired-contact interpolation.

Inputs are interpreted interval endpoints, not economic ore labels. Local plane
and thickness interpolation are hypotheses; support diagnostics are mandatory.
"""
import numpy as np
from scipy.spatial import cKDTree

class OrientedLens:
    def __init__(self, ranges=(150.,50.,100.), neighbors=24, max_per_hole=2, min_holes=3):
        self.ranges=np.asarray(ranges,float)
        if self.ranges.shape!=(3,) or not np.isfinite(self.ranges).all() or (self.ranges<=0).any():raise ValueError('Positive ranges required')
        if any(not isinstance(v,int) or v<1 for v in [neighbors,max_per_hole,min_holes]) or neighbors<min_holes:raise ValueError('Invalid neighborhood')
        self.neighbors=neighbors;self.max_per_hole=max_per_hole;self.min_holes=min_holes

    def _select(self, xyz):
        # Every interval within one range is considered (2026-11-13): a fixed 128-nearest query let one densely
        # sampled hole fill the list and hide the other holes needed for min_holes support.
        q=np.atleast_2d(xyz)/self.ranges
        results=[]
        for p,ii in zip(q,self.tree.query_ball_point(q,1.0)):
            ii=np.asarray(ii,int)
            if not len(ii):results.append(ii);continue
            dd=np.linalg.norm(self.tree.data[ii]-p,axis=1);order=np.argsort(dd,kind='stable');dd=dd[order];ii=ii[order]
            counts={};used=[]
            for distance,j in zip(dd,ii):
                if distance>1:break
                h=self.holes[j]
                if counts.get(h,0)>=self.max_per_hole:continue
                counts[h]=counts.get(h,0)+1;used.append(j)
                if len(used)==self.neighbors:break
            results.append(np.asarray(used,int))
        return results

    def fit(self, starts, ends, hole_ids):
        a=np.asarray(starts,float);b=np.asarray(ends,float);h=np.asarray(hole_ids).astype(str)
        if a.ndim!=2 or a.shape[1]!=3 or b.shape!=a.shape or len(h)!=len(a) or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite paired contacts and matching hole IDs required')
        if np.any(np.linalg.norm(b-a,axis=1)<=0) or any(not x.strip() or x in ['None','nan'] for x in h):raise ValueError('Nonzero intervals and valid holes required')
        self.centers=(a+b)/2;self.starts=a;self.ends=b;self.holes=h;self.tree=cKDTree(self.centers/self.ranges)
        self.normals=np.full(a.shape,np.nan);self.halfwidth=np.full(len(a),np.nan);self.planarity=np.zeros(len(a));self.support_holes=np.zeros(len(a),int)
        for i,ids in enumerate(self._select(self.centers)):
            nh=len(set(h[ids]));self.support_holes[i]=nh
            if nh<self.min_holes or len(ids)<3:continue
            x=self.centers[ids];w=1/(1+np.linalg.norm((x-self.centers[i])/self.ranges,axis=1)**2)
            delta=x-np.average(x,axis=0,weights=w);cov=(delta*w[:,None]).T@delta/w.sum();values,vectors=np.linalg.eigh(cov)
            # Reject line-like supports; a line cannot define a unique plane.
            if values[1]<1 or values[1]<2*max(values[0],1e-8):continue
            normal=vectors[:,0]
            if normal[np.argmax(abs(normal))]<0:normal=-normal
            width=abs(np.dot(b[i]-a[i],normal))/2
            if width<.25:continue
            self.normals[i]=normal;self.halfwidth[i]=width;self.planarity[i]=values[1]/max(values[0],1e-8)
        return self

    def evaluate(self, xyz):
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite XYZ required')
        field=np.full(len(x),np.nan);half=np.full(len(x),np.nan);normal=np.full(x.shape,np.nan);nh=np.zeros(len(x),int)
        for i,ids in enumerate(self._select(x)):
            ids=ids[np.isfinite(self.halfwidth[ids])];nh[i]=len(set(self.holes[ids]))
            if nh[i]<self.min_holes:continue
            normals=self.normals[ids].copy();reference=normals[0];normals[np.einsum('ij,j->i',normals,reference)<0]*=-1
            w=1/(.02+np.linalg.norm((self.centers[ids]-x[i])/self.ranges,axis=1)**2)
            n=np.average(normals,axis=0,weights=w);norm=np.linalg.norm(n)
            if norm<.5:continue
            n/=norm;normal[i]=n
            signed=np.einsum('ij,ij->i',x[i]-self.centers[ids],normals)
            field[i]=np.average(signed,weights=w);half[i]=np.average(self.halfwidth[ids],weights=w)
        return dict(signed_distance=field,halfwidth=half,normals=normal,support_holes=nh,supported=np.isfinite(field))
