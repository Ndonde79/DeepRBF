# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Positive normalized radial-basis categorical probabilities with domain gates.

This is kernel interpolation, not an exact RBF linear-system solver or kriging.
Independent gate and within-domain bandwidths allow weathering to have a
different continuity from bedrock. Geometry and domains are explicit inputs.
"""
import numpy as np
from scipy.spatial import cKDTree


class DomainRBF:
    def __init__(self, categories, domains, *, ranges, rotation=None,
                 gate_scale=(1.,1.,1.), max_neighbors=24, max_per_hole=3,
                 min_holes=3, candidate_neighbors=128, radius=3.):
        self.categories=np.asarray(categories)
        self.domains=np.asarray(domains)
        self.ranges=np.asarray(ranges,float)
        self.rotation=np.eye(3) if rotation is None else np.asarray(rotation,float)
        self.gate_scale=np.asarray(gate_scale,float)
        if self.categories.ndim!=1 or len(self.categories)<2 or not np.issubdtype(self.categories.dtype,np.integer) or not np.array_equal(np.unique(self.categories),self.categories):
            raise ValueError('Sorted unique integer categories required')
        if self.domains.shape!=self.categories.shape:raise ValueError('One domain per category required')
        if self.ranges.shape!=(3,) or self.gate_scale.shape!=(3,) or not np.isfinite(np.r_[self.ranges,self.gate_scale]).all() or (self.ranges<=0).any() or (self.gate_scale<=0).any():raise ValueError('Positive finite range and gate triples required')
        if self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all() or not np.allclose(self.rotation@self.rotation.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(self.rotation),1):raise ValueError('Right-handed orthonormal rotation required')
        if any(not isinstance(v,int) or v<1 for v in [max_neighbors,max_per_hole,min_holes,candidate_neighbors]) or candidate_neighbors<max_neighbors or not np.isfinite(radius) or radius<=0:raise ValueError('Invalid neighborhood controls')
        self.max_neighbors=max_neighbors;self.max_per_hole=max_per_hole
        self.min_holes=min_holes;self.candidate_neighbors=candidate_neighbors;self.radius=radius
        self.group_values,self.group_index=np.unique(self.domains,return_inverse=True)

    def fit(self,xyz,labels,hole_ids,support=None):
        x=np.asarray(xyz,float);labels=np.asarray(labels);holes=np.asarray(hole_ids)
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all() or labels.shape!=(len(x),) or holes.shape!=(len(x),):raise ValueError('Finite sample coordinates, labels and holes required')
        if not np.isin(labels,self.categories).all():raise ValueError('Unknown sample category')
        w=np.ones(len(x)) if support is None else np.asarray(support,float)
        if w.shape!=(len(x),) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Positive support required')
        self.xyz=x.copy();self.center=x.mean(0);self.local=(x-self.center)@self.rotation.T
        self.encoded=np.searchsorted(self.categories,labels);_,self.holes=np.unique(holes,return_inverse=True)
        # Equal total prior weight per hole, weighted by interval support within it.
        total=np.bincount(self.holes,weights=w)
        self.prior=np.bincount(self.encoded,weights=w/total[self.holes],minlength=len(self.categories))+.001
        self.prior/=self.prior.sum();self.support=w.copy()
        self.tree=cKDTree(self.local/self.ranges)
        self.exact_tree=cKDTree(self.local)
        self._native_model=None;self._native_key=None
        return self

    def predict(self,xyz,*,hierarchical=True,prior_mix=0.,chunk_size=1024):
        if not hasattr(self,'tree'):raise ValueError('Fit before prediction')
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite N by 3 targets required')
        if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('Positive integer chunk size required')
        if not np.isfinite(prior_mix) or not 0<=prior_mix<=1:raise ValueError('Prior mixture must be between zero and one')
        local=(x-self.center)@self.rotation.T
        p=np.empty((len(x),len(self.categories)));nh=np.zeros(len(x),int);nearest=np.empty(len(x));exact=np.zeros(len(x),bool)
        ng=len(self.group_values);nc=len(self.categories)
        for start in range(0,len(x),chunk_size):
            q=local[start:start+chunk_size];n=len(q);k=min(self.candidate_neighbors,len(self.local))
            query=q/self.ranges;probe=min(k+1,len(self.local))
            dd,ii=self.tree.query(query,k=probe)
            dd=np.asarray(dd).reshape(n,probe);ii=np.asarray(ii).reshape(n,probe)
            # cKDTree does not promise index order for equidistant samples.
            # Match the native (distance, input index) contract, including ties
            # at the candidate budget boundary.
            distance=dd[:,:k].copy();idx=ii[:,:k].copy()
            for r in range(n):
                if probe>k and dd[r,k]==dd[r,k-1]:
                    tied=np.asarray(self.tree.query_ball_point(query[r],np.nextafter(dd[r,k-1],np.inf)),int)
                    td=np.linalg.norm(self.tree.data[tied]-query[r],axis=1)
                    order=np.lexsort((tied,td))[:k];idx[r]=tied[order];distance[r]=td[order]
            order=np.lexsort((idx,distance),axis=1)
            distance=np.take_along_axis(distance,order,axis=1);idx=np.take_along_axis(idx,order,axis=1)
            hid=self.holes[idx];order=np.argsort(hid,axis=1,kind='stable');sorted_h=np.take_along_axis(hid,order,axis=1)
            pos=np.broadcast_to(np.arange(k),(n,k));first=np.maximum.accumulate(np.where(np.c_[np.ones(n,bool),np.diff(sorted_h,axis=1)!=0],pos,0),axis=1)
            rank=np.take_along_axis(pos-first+1,np.argsort(order,axis=1),axis=1)
            keep=(rank<=self.max_per_hole)&(distance<=self.radius)
            keep&=np.cumsum(keep,axis=1)<=self.max_neighbors
            kept_h=np.sort(np.where(keep,hid,-1),axis=1)
            n_h=np.sum((kept_h>=0)&np.c_[np.ones(n,bool),np.diff(kept_h,axis=1)!=0],axis=1)
            w=np.exp(-.5*distance**2)*self.support[idx]*keep
            cat=self.encoded[idx];row=np.repeat(np.arange(n),k)
            raw=np.zeros((n,nc));np.add.at(raw,(row,cat.ravel()),w.ravel())
            if hierarchical:
                delta=(q[:,None,:]-self.local[idx])/(self.ranges*self.gate_scale)
                gate_w=np.exp(-.5*np.sum(delta**2,axis=2))*self.support[idx]*keep
                group=self.group_index[cat];gate=np.zeros((n,ng));np.add.at(gate,(row,group.ravel()),gate_w.ravel())
                gp=np.bincount(self.group_index,weights=self.prior,minlength=ng)
                gate+=1e-12*gp;gate/=gate.sum(1)[:,None]
                prob=np.empty_like(raw)
                for j in range(ng):
                    mask=self.group_index==j;conditional=raw[:,mask]+1e-12*self.prior[mask]
                    prob[:,mask]=gate[:,j,None]*conditional/conditional.sum(1)[:,None]
            else:
                prob=raw+1e-12*self.prior;prob/=prob.sum(1)[:,None]
            prob[n_h<self.min_holes]=self.prior
            prob=(1-prior_mix)*prob+prior_mix*self.prior
            # Honor all samples at a coincident node. Conflicting labels yield
            # explicit support fractions rather than an arbitrary hard category.
            physical_distance,_=self.exact_tree.query(q)
            for r in np.flatnonzero(physical_distance<=1e-7):
                matches=np.asarray(self.exact_tree.query_ball_point(q[r],1e-7),int)
                weights=self.support[matches];prob[r]=np.bincount(self.encoded[matches],weights=weights,minlength=nc)/weights.sum()
                exact[start+r]=True
            p[start:start+n]=prob;nh[start:start+n]=n_h;nearest[start:start+n]=physical_distance
        return p,dict(support_holes=nh,nearest_sample_m=nearest,exact_conditioned=exact,
                      supported=(nh>=self.min_holes)|exact)

    def predict_native(self,xyz,*,hierarchical=True,prior_mix=0.):
        """Use the tested C++ backend with the same conditioning and controls."""
        if not hasattr(self,'tree'):raise ValueError('Fit before prediction')
        from deeprbf_recovered import NeighborhoodSpec,NativeCategoricalModel
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite N by 3 targets required')
        s=NeighborhoodSpec();s.radii=self.ranges*self.radius;s.axes=self.rotation
        s.max_neighbors=self.max_neighbors;s.min_neighbors=1;s.min_holes=self.min_holes
        s.max_per_hole=self.max_per_hole;s.candidate_neighbors=self.candidate_neighbors
        key=(tuple(self.ranges),tuple(self.rotation.ravel()),tuple(self.gate_scale),self.radius,
             self.max_neighbors,self.min_holes,self.max_per_hole,self.candidate_neighbors)
        if self._native_model is None or key!=self._native_key:
            self._native_model=NativeCategoricalModel(self.xyz,self.categories[self.encoded],self.holes,self.support,
                self.categories,self.group_index,self.ranges,self.rotation,self.gate_scale,s)
            self._native_key=key
        r=self._native_model.predict(x,hierarchical,prior_mix)
        nearest=self.exact_tree.query((x-self.center)@self.rotation.T)[0]
        return np.asarray(r.probabilities).reshape(len(x),len(self.categories)),dict(
            support_holes=np.asarray(r.support_holes),nearest_sample_m=nearest,
            exact_conditioned=np.asarray(r.exact_conditioned,dtype=bool),supported=np.asarray(r.supported,dtype=bool))
