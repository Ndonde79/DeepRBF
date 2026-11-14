# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Cross-hole experimental variogram with independently fitted total sill.

Pairs are capped by hole pair and lag to limit dense-hole domination. This is
an experimental covariance fit; diagnostics are retained, including bounds.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

def fit_crosshole_variogram(xyz, values, holes, *, seed=571, max_lag=400., pair_cap=12):
    x=np.asarray(xyz,float);y=np.asarray(values,float);h=np.asarray(holes)
    if x.shape!=(len(y),3) or len(h)!=len(y) or len(y)<30 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('At least 30 finite samples required')
    if len(np.unique(h))<4 or max_lag<25 or not isinstance(pair_cap,int) or pair_cap<1:raise ValueError('Four holes, max lag >=25 and positive integer pair cap required')
    rng=np.random.default_rng(seed);n=len(y);anchors=rng.choice(n,min(n,6000),replace=False)
    _,nb=cKDTree(x).query(x[anchors],k=min(n,40));a=np.repeat(anchors,nb.shape[1]);b=nb.ravel()
    a=np.r_[a,rng.integers(n,size=150000)];b=np.r_[b,rng.integers(n,size=150000)]
    pairs=np.unique(np.sort(np.c_[a,b],axis=1),axis=0);a,b=pairs.T;dist=np.linalg.norm(x[a]-x[b],axis=1);ok=(h[a]!=h[b])&(dist>1)&(dist<=max_lag);a=a[ok];b=b[ok];dist=dist[ok]
    if not len(a):return dict(range=60.,sill=max(float(np.var(y)),1e-8),nugget_fraction=.35,status='INSUFFICIENT_LAG_BINS',lags=[],semivariance=[],pairs=[])
    _,hid=np.unique(h,return_inverse=True);hp=np.sort(np.c_[hid[a],hid[b]],axis=1);lag=np.minimum((dist/25).astype(int),int(max_lag/25)-1)
    key=np.c_[lag,hp];order=np.lexsort((rng.random(len(a)),key[:,2],key[:,1],key[:,0]));sortedkey=key[order]
    start=np.r_[True,np.any(sortedkey[1:]!=sortedkey[:-1],axis=1)];runstart=np.maximum.accumulate(np.where(start,np.arange(len(order)),0));keep=order[np.arange(len(order))-runstart<pair_cap]
    a=a[keep];b=b[keep];dist=dist[keep];lag=lag[keep];gamma=.5*(y[a]-y[b])**2
    lags=[];sem=[];counts=[]
    for k in range(int(max_lag/25)):
        z=lag==k
        if z.sum()>=30:lags.append(float(dist[z].mean()));sem.append(float(gamma[z].mean()));counts.append(int(z.sum()))
    variance=max(float(np.var(y)),1e-8)
    if len(lags)<4:return dict(range=60.,sill=variance,nugget_fraction=.35,status='INSUFFICIENT_LAG_BINS',lags=lags,semivariance=sem,pairs=counts)
    d=np.asarray(lags);v=np.asarray(sem);weights=np.sqrt(np.minimum(counts,1000)/np.minimum(counts,1000).sum())
    def residual(p):return weights*(p[0]*(p[2]+(1-p[2])*(1-np.exp(-3*d/p[1])))-v)/variance
    fit=least_squares(residual,[variance,100.,.35],bounds=([variance*.1,10.,.01],[variance*5,600.,.95]),loss='soft_l1',f_scale=.5)
    sill,ran,nug=fit.x;bound=bool(sill<=variance*.101 or sill>=variance*4.99 or ran<=10.1 or ran>=599 or nug<=.011 or nug>=.949)
    return dict(range=float(ran),sill=float(sill),nugget_fraction=float(nug),status='FIT_BOUND' if bound else ('RANGE_WEAKLY_CONSTRAINED' if ran>max(lags) else 'FITTED'),lags=lags,semivariance=sem,pairs=counts,raw_variance=variance,pair_cap_per_hole_pair_per_lag=pair_cap,fit_success=bool(fit.success))
