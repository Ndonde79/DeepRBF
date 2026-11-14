# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Local implicit interpolation of explicit signed/contact constraints.

Values and anisotropy are caller-supplied geological hypotheses. This component
does not infer geological truth or produce calibrated category probabilities.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


class ContactRBF:
    def __init__(self, ranges, rotation=None, neighbors=24, smoothing=1e-8):
        self.ranges=np.asarray(ranges,float)
        self.rotation=np.eye(3) if rotation is None else np.asarray(rotation,float)
        if self.ranges.shape!=(3,) or not np.isfinite(self.ranges).all() or (self.ranges<=0).any():
            raise ValueError('Positive finite ranges required')
        if self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all() or not np.allclose(self.rotation@self.rotation.T,np.eye(3),atol=1e-8):
            raise ValueError('Orthonormal rotation required')
        if not isinstance(neighbors,int) or neighbors<4 or not np.isfinite(smoothing) or smoothing<0:
            raise ValueError('Invalid interpolation controls')
        self.neighbors=neighbors;self.smoothing=smoothing

    def fit(self, xyz, signed_values):
        x=np.asarray(xyz,float);v=np.asarray(signed_values,float)
        if x.ndim!=2 or x.shape[1]!=3 or len(x)<4 or v.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('At least four finite XYZ/value constraints required')
        if len(np.unique(x,axis=0))!=len(x):raise ValueError('Duplicate XYZ constraints must be reconciled explicitly')
        self.center=x.mean(0)
        local=(x-self.center)@self.rotation.T/self.ranges
        self._tree=cKDTree(local);self._values=v.copy()
        self._positive_tree=cKDTree(x[v>0]) if (v>0).any() else None
        self.model=RBFInterpolator(local,v,
            neighbors=min(self.neighbors,len(x)),kernel='linear',degree=0,smoothing=self.smoothing)
        return self

    def neighborhood_diagnostics(self, xyz):
        """Expose signed support in the actual interpolation neighbourhood.

        Distances to positive constraints are physical metres if XYZ uses metres.
        Counts describe support, not geological confidence or probabilities.
        """
        if not hasattr(self,'model'):raise ValueError('Fit before diagnostics')
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite XYZ required')
        local=(x-self.center)@self.rotation.T/self.ranges
        k=min(self.neighbors,len(self._values))
        _,idx=self._tree.query(local,k=k)
        values=self._values[idx]
        return dict(positive_constraints=(values>0).sum(axis=1),
            negative_constraints=(values<0).sum(axis=1),contact_constraints=(values==0).sum(axis=1),
            nearest_positive_distance=(self._positive_tree.query(x)[0] if self._positive_tree is not None else np.full(len(x),np.inf)))

    def predict(self, xyz, chunk_size=4000):
        if not hasattr(self,'model'):raise ValueError('Fit before prediction')
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite XYZ targets required')
        if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('Positive chunk size required')
        result=np.empty(len(x))
        for start in range(0,len(x),chunk_size):
            q=(x[start:start+chunk_size]-self.center)@self.rotation.T/self.ranges
            result[start:start+len(q)]=self.model(q)
        return result


class BalancedContactRBF(ContactRBF):
    """Linear local RBF with explicit per-hole limits and support rejection.

    Predictions without min_holes and four constraints return NaN. Coincident
    constraints are honoured exactly and flagged separately. Search radius is
    dimensionless in the supplied rotated/range-scaled coordinates.
    """
    def __init__(self, ranges, rotation=None, neighbors=24, smoothing=1e-8,
                 max_per_hole=4, min_holes=2, candidate_neighbors=128, radius=3.):
        super().__init__(ranges,rotation,neighbors,smoothing)
        if any(not isinstance(v,int) or v<1 for v in [max_per_hole,min_holes,candidate_neighbors]):
            raise ValueError('Positive integer hole/search controls required')
        if candidate_neighbors<neighbors or min_holes>neighbors or not np.isfinite(radius) or radius<=0:
            raise ValueError('Invalid balanced search controls')
        self.max_per_hole=max_per_hole;self.min_holes=min_holes
        self.candidate_neighbors=candidate_neighbors;self.radius=radius

    def fit(self, xyz, signed_values, hole_ids):
        ids=np.asarray(hole_ids)
        if ids.ndim!=1 or len(ids)!=len(xyz) or any(v is None for v in ids):
            raise ValueError('One nonmissing hole ID per constraint required')
        if ids.dtype.kind in 'fc' and not np.isfinite(ids).all():raise ValueError('Finite hole IDs required')
        if any(not str(v).strip() for v in ids):raise ValueError('Nonempty hole IDs required')
        super().fit(xyz,signed_values)
        _,self._holes=np.unique(ids.astype(str),return_inverse=True)
        self._local=self._tree.data
        self._physical_tree=cKDTree(np.asarray(xyz,float))
        return self

    def _search(self, q):
        distance,idx=self._tree.query(q,k=min(self.candidate_neighbors,len(self._values)))
        order=np.lexsort((idx,distance),axis=1)
        idx=np.take_along_axis(idx,order,axis=1);distance=np.take_along_axis(distance,order,axis=1)
        holes=self._holes[idx];n,k=idx.shape
        order=np.argsort(holes,axis=1,kind='stable');sorted_h=np.take_along_axis(holes,order,axis=1)
        position=np.broadcast_to(np.arange(k),(n,k))
        first=np.maximum.accumulate(np.where(np.c_[np.ones(n,bool),np.diff(sorted_h,axis=1)!=0],position,0),axis=1)
        rank=np.take_along_axis(position-first+1,np.argsort(order,axis=1),axis=1)
        keep=(rank<=self.max_per_hole)&(distance<=self.radius)
        keep&=np.cumsum(keep,axis=1)<=self.neighbors
        h=np.sort(np.where(keep,holes,-1),axis=1)
        nh=np.sum((h>=0)&np.c_[np.ones(n,bool),np.diff(h,axis=1)!=0],axis=1)
        selected=np.argsort(~keep,axis=1,kind='stable')[:,:self.neighbors]
        ids=np.take_along_axis(idx,selected,axis=1)
        return ids,keep.sum(1),nh

    def predict_with_diagnostics(self, xyz, chunk_size=512):
        if not hasattr(self,'_holes'):raise ValueError('Fit with hole IDs before prediction')
        x=np.asarray(xyz,float)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite XYZ required')
        if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('Positive chunk size required')
        out=np.full(len(x),np.nan);counts=np.zeros(len(x),int);holes=counts.copy();positive=counts.copy()
        exact=np.zeros(len(x),bool);supported=np.zeros(len(x),bool)
        for start in range(0,len(x),chunk_size):
            sl=slice(start,start+chunk_size);q=(x[sl]-self.center)@self.rotation.T/self.ranges
            ids,n,nh=self._search(q);counts[sl]=n;holes[sl]=nh
            valid=(n>=4)&(nh>=self.min_holes)
            positive[sl]=np.sum((self._values[ids]>0)&(np.arange(ids.shape[1])[None,:]<n[:,None]),axis=1)
            result=np.full(len(q),np.nan)
            for size in np.unique(n[valid]):
                take=valid&(n==size);ix=ids[take,:size];p=self._local[ix]
                matrix=np.ones((len(p),size+1,size+1))
                matrix[:,:size,:size]=-np.linalg.norm(p[:,:,None,:]-p[:,None,:,:],axis=-1)
                diagonal=np.arange(size);matrix[:,diagonal,diagonal]+=self.smoothing
                matrix[:,-1,-1]=0
                rhs=np.zeros((len(p),size+1));rhs[:,:size]=self._values[ix]
                coefficients=np.linalg.solve(matrix,rhs[...,None])[...,0]
                basis=-np.linalg.norm(q[take,None,:]-p,axis=-1)
                result[take]=np.sum(basis*coefficients[:,:size],axis=1)+coefficients[:,-1]
            d,near=self._physical_tree.query(x[sl]);coincident=d<=1e-8
            result[coincident]=self._values[near[coincident]]
            exact[sl]=coincident;supported[sl]=valid|coincident;out[sl]=result
        return out,dict(constraints=counts,support_holes=holes,positive_constraints=positive,
                        supported=supported,exact_conditioned=exact)

    def predict(self, xyz, chunk_size=512):
        return self.predict_with_diagnostics(xyz,chunk_size)[0]

    def neighborhood_diagnostics(self, xyz):
        return self.predict_with_diagnostics(xyz)[1]
