# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Bounded directional experimental variograms and transparent first-pass fits."""
import numpy as np
from scipy.optimize import least_squares


def directional_fit(xyz,values,axes,*,family='exponential',seed=42,max_pairs=150000,hole_ids=None,isotropic=False):
    x=np.asarray(xyz,float);y=np.asarray(values,float);axes=np.asarray(axes,float)
    if x.shape!=(len(y),3) or len(y)<20 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite training data required')
    if family not in ('exponential','gaussian'):raise ValueError('Unknown family')
    if axes.shape!=(3,3) or not np.allclose(axes@axes.T,np.eye(3)):raise ValueError('Orthonormal axes required')
    sill=float(np.var(y));local=(x-x.mean(0))@axes.T
    if sill<=1e-15:raise ValueError('Constant population has no identifiable variogram')
    rng=np.random.default_rng(seed);n=min(max_pairs,len(y)*(len(y)-1)//2)
    i=rng.integers(0,len(y),n*2);j=rng.integers(0,len(y),n*2)
    pairs=np.unique(np.sort(np.c_[i,j],axis=1),axis=0);pairs=pairs[pairs[:,0]!=pairs[:,1]]
    # np.unique sorts by row index: taking its prefix would bias toward early
    # holes in source order. Uniformly subsample the deduplicated pair pool.
    if len(pairs)>n:pairs=pairs[rng.choice(len(pairs),n,replace=False)]
    hole_pairs=None
    if hole_ids is not None:
        holes=np.asarray(hole_ids).astype(str)
        if holes.shape!=y.shape:raise ValueError('One drillhole ID per variogram observation required')
        _,codes=np.unique(holes,return_inverse=True)
        hp=codes[pairs]
        between=hp[:,0]!=hp[:,1]
        pairs=pairs[between]
        hole_pairs=np.sort(hp[between],axis=1)
    delta=local[pairs[:,0]]-local[pairs[:,1]];distance=np.linalg.norm(delta,axis=1)
    semivariance=.5*(y[pairs[:,0]]-y[pairs[:,1]])**2
    rows=[];issues=[];span=np.ptp(local,axis=0);maximum=np.maximum(span*.5,10.)
    if isotropic:
        span=np.array([np.linalg.norm(span)])
        maximum=np.array([max(span[0]*.5,10.)])
    for axis in range(len(span)):
        along=distance if isotropic else abs(delta[:,axis])
        angle=along/np.maximum(distance,1e-12)
        for lag in range(12):
            low=maximum[axis]*lag/12;high=maximum[axis]*(lag+1)/12
            keep=(distance>0)&(angle>=np.cos(np.deg2rad(30)))&(along>=low)&(along<high)
            if keep.sum()>=30:
                distinct=None if hole_pairs is None else len(np.unique(hole_pairs[keep],axis=0))
                if distinct is not None and distinct<3:continue
                row=dict(axis=axis,lag=float(along[keep].mean()),gamma=float(semivariance[keep].mean()),pairs=int(keep.sum()))
                if distinct is not None:row['hole_pairs']=distinct
                rows.append(row)
        if sum(row['axis']==axis for row in rows)<3:issues.append('INSUFFICIENT_DIRECTION_'+str(axis))
    if issues:return dict(status='UNIDENTIFIABLE',issues=issues,sill=sill,experimental=rows)
    direction=np.array([r['axis'] for r in rows]);h=np.array([r['lag'] for r in rows]);observed=np.array([r['gamma'] for r in rows])/sill
    weights=np.sqrt(np.array([r['pairs'] for r in rows],float));weights/=weights.max()
    lower=np.r_[.001,np.full(len(span),2.)];upper=np.r_[.95,np.maximum(span,10.)]
    def residual(parameters):
        nugget=parameters[0];scaled=h/parameters[1:][direction]
        model=nugget+(1-nugget)*(1-np.exp(-3*scaled**(2 if family=='gaussian' else 1)))
        return weights*(model-observed)
    fitted=least_squares(residual,np.r_[.2,np.clip(maximum*.5,2.01,upper[1:]-.01)],bounds=(lower,upper))
    if not fitted.success:return dict(status='UNIDENTIFIABLE',issues=['OPTIMIZER_DID_NOT_CONVERGE'],sill=sill,experimental=rows)
    on_bound=bool(np.any(np.isclose(fitted.x,lower,rtol=.01)) or np.any(np.isclose(fitted.x,upper,rtol=.01)))
    ranges=np.repeat(fitted.x[1:],3) if isotropic else fitted.x[1:]
    return dict(status='FIT_AT_BOUND' if on_bound else 'FIT',sill=sill,nugget_fraction=float(fitted.x[0]),
        ranges=ranges.tolist(),family=family,range_convention='practical',experimental=rows,
        weighted_residual=float(np.linalg.norm(fitted.fun)),pairs_sampled=len(pairs),
        limitation='Training-only numerical fit; orientation supplied externally; sparse directions rejected')
