# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Training-only indicator calibration and auditable full simple IK.

Hole balancing is not spatial declustering. Models are fitted independently;
their joint indicator consistency still requires geological review.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class IndicatorCalibration:
    categories: np.ndarray
    prior: np.ndarray
    ranges: np.ndarray
    rotation: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        c=np.asarray(self.categories);p=np.asarray(self.prior);r=np.asarray(self.ranges);rot=np.asarray(self.rotation)
        if c.ndim!=1 or len(c)<2 or not np.issubdtype(c.dtype,np.integer) or not np.array_equal(c,np.unique(c)):
            raise ValueError('Calibration categories must be sorted unique integer codes')
        if p.shape!=c.shape or not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1):
            raise ValueError('Calibration prior must be positive and sum to one')
        if r.shape!=(len(c),3) or not np.isfinite(r).all() or (r<=0).any():
            raise ValueError('One positive range triple per category required')
        if rot.shape!=(3,3) or not np.isfinite(rot).all() or not np.allclose(rot@rot.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rot),1):
            raise ValueError('Calibration rotation must be right-handed orthonormal')
        for name,value in [('categories',c),('prior',p),('ranges',r),('rotation',rot)]:
            array=value.copy();array.flags.writeable=False;object.__setattr__(self,name,array)


def _coordinates(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Coordinates must be finite N by 3')
    return x


def fit_indicator_models(sample_xyz, sample_categories, hole_ids, *,
                         ranges=(160., 70., 30.), rotation=None, seed=42,
                         max_pairs=60000, min_positive_pairs=30):
    """Fit a range multiplier per lithology using binned cross-hole pairs.

    Anisotropy ratios/orientation are supplied explicitly. A seeded uniform
    selection of anchor holes reduces influence of densely sampled holes;
    nearest cross-hole and random pairs provide local and longer-lag support.
    Weak fits retain the supplied range and carry a REVIEW flag. This is an
    engineering calibration, not an automatic geological interpretation.
    """
    x = _coordinates(sample_xyz); labels = np.asarray(sample_categories)
    holes = np.asarray(hole_ids)
    if labels.shape != (len(x),) or holes.shape != (len(x),) or not len(x):
        raise ValueError('One category and hole ID per nonempty sample is required')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Categories must be integer codes')
    categories, encoded = np.unique(labels, return_inverse=True)
    if len(categories) < 2: raise ValueError('At least two categories required')
    base = np.asarray(ranges, float)
    rot = np.eye(3) if rotation is None else np.asarray(rotation, float)
    if base.shape != (3,) or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Three positive finite ranges required')
    if rot.shape != (3,3) or not np.isfinite(rot).all() or not np.allclose(rot@rot.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rot),1):
        raise ValueError('Rotation must be right-handed orthonormal')
    if not isinstance(max_pairs,int) or max_pairs < 100 or min_positive_pairs < 1:
        raise ValueError('Invalid pair limits')
    unique_holes, hi, counts = np.unique(holes, return_inverse=True, return_counts=True)
    if len(unique_holes) < 2: raise ValueError('At least two holes required')
    weights = 1. / counts[hi]; weights /= weights.sum()
    prior = np.bincount(encoded, weights=weights, minlength=len(categories))
    scaled = (x-x.mean(0))@rot.T/base
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(x), size=max_pairs, p=weights)
    other = rng.choice(len(x), size=max_pairs, p=weights)
    # Half the budget captures nearest cross-hole pairs without same-hole bias.
    nlocal = max_pairs//2
    _, near = cKDTree(scaled).query(scaled[anchors[:nlocal]], k=min(128,len(x)))
    near = np.asarray(near).reshape(nlocal,-1)
    cross = hi[near] != hi[anchors[:nlocal],None]
    has = cross.any(1)
    other[:nlocal][has] = near[np.flatnonzero(has),cross[has].argmax(1)]
    distance = np.linalg.norm(scaled[anchors]-scaled[other],axis=1)
    valid = (hi[anchors] != hi[other]) & (distance > 1e-9) & (distance <= 4)
    a,b,distance = anchors[valid],other[valid],distance[valid]
    edges = np.linspace(0,4,17); bins = np.minimum(np.searchsorted(edges,distance,side='right')-1,15)
    fitted=[]; reports=[]
    for k,c in enumerate(categories):
        positive_pairs = int(np.sum((encoded[a]==k)|(encoded[b]==k)))
        observed = .5*((encoded[a]==k)!=(encoded[b]==k))
        empirical=[];lags=[];npairs=[]
        for j in range(16):
            mask=bins==j
            if mask.sum()>=20:
                empirical.append(float(observed[mask].mean()));lags.append(float(distance[mask].mean()));npairs.append(int(mask.sum()))
        lag=np.array(lags); gamma=np.array(empirical); count=np.array(npairs)
        sill=prior[k]*(1-prior[k]); multiplier=1.; status='REVIEW_INSUFFICIENT_PAIRS'; objective=None
        if positive_pairs>=min_positive_pairs and len(lag)>=3 and sill>1e-8:
            def objective_fn(logscale):
                residual=sill*(1-np.exp(-3*lag/np.exp(logscale)))-gamma
                return float(np.average(residual**2,weights=np.sqrt(count)))
            fit=minimize_scalar(objective_fn,bounds=(np.log(.15),np.log(4.)),method='bounded')
            multiplier=float(np.exp(fit.x));objective=float(fit.fun)
            status='FITTED' if fit.success else 'REVIEW_OPTIMIZER'
            if multiplier<.151 or multiplier>3.99: status='REVIEW_RANGE_BOUND'
        fitted.append(base*multiplier)
        reports.append(dict(category=int(c),status=status,positive_pairs=positive_pairs,
                            range_multiplier=multiplier,objective=objective,
                            lag_scaled=lags,semivariance=empirical,pair_counts=npairs))
    return IndicatorCalibration(categories,prior,np.asarray(fitted),rot,
        dict(seed=seed,pair_count=len(a),model='exponential, zero geological nugget',
             proportions='equal-hole weighted sample proportions; not spatial declustering',
             anisotropy='fixed supplied ratios and orientation; fitted scalar per category',categories=reports))


def estimate_full_indicator(sample_xyz, sample_categories, hole_ids, target_xyz, calibration, *,
                            max_neighbors=20, max_per_hole=3, min_holes=3,
                            search_ranges=(320.,140.,60.)):
    """Category-specific simple IK, deterministic hole quotas and support audit.

    Unsupported nodes return explicit priors and UNSUPPORTED flags. Exact
    observed coordinates are honored, including where minimum support fails.
    Probabilities are clipped then normalized; correction size is reported.
    """
    from .lithology import LithologyResult
    x=_coordinates(sample_xyz);q=_coordinates(target_xyz)
    labels=np.asarray(sample_categories);holes=np.asarray(hole_ids)
    if not len(x) or labels.shape!=(len(x),) or holes.shape!=(len(x),):raise ValueError('Invalid sample arrays')
    if not np.issubdtype(labels.dtype,np.integer):raise ValueError('Categories must be integer codes')
    if any(not isinstance(v,int) or v<1 for v in (max_neighbors,max_per_hole,min_holes)):
        raise ValueError('Support controls must be positive integers')
    cats=np.asarray(calibration.categories);prior=np.asarray(calibration.prior)
    if not np.isin(labels,cats).all():raise ValueError('Unknown category')
    encoded=np.searchsorted(cats,labels)
    search=np.asarray(search_ranges,float)
    if search.shape!=(3,) or not np.isfinite(search).all() or (search<=0).any():raise ValueError('Invalid search ranges')
    # Reject contradictory collocated logs rather than choosing one silently.
    unique,inverse=np.unique(x,axis=0,return_inverse=True)
    lo=np.full(len(unique),len(cats));high=np.full(len(unique),-1)
    np.minimum.at(lo,inverse,encoded);np.maximum.at(high,inverse,encoded)
    if np.any(lo!=high):raise ValueError('Conflicting collocated sample categories')
    center=x.mean(0);sx=(x-center)@calibration.rotation.T; tq=(q-center)@calibration.rotation.T
    tree=cKDTree(sx/search)
    probabilities=np.tile(prior,(len(q),1));support=[];corrections=[]
    for i,target in enumerate(tq):
        candidates=np.asarray(tree.query_ball_point(target/search,1.),int)
        if len(candidates):
            distances=np.linalg.norm((sx[candidates]-target)/search,axis=1)
            candidates=candidates[np.lexsort((candidates,distances))]
            if distances.min()<1e-10:
                probabilities[i]=np.eye(len(cats))[encoded[candidates[0]]]
                support.append(dict(status='HARD_DATA',samples=1,holes=1));corrections.append(0.);continue
        selected=[];by_hole={}
        for j in candidates:
            h=holes[j]
            if by_hole.get(h,0)>=max_per_hole:continue
            selected.append(j);by_hole[h]=by_hole.get(h,0)+1
            if len(selected)>=max_neighbors:break
        status='ESTIMATED' if len(by_hole)>=min_holes else 'UNSUPPORTED'
        support.append(dict(status=status,samples=len(selected),holes=len(by_hole)))
        if status=='UNSUPPORTED':corrections.append(0.);continue
        sel=np.asarray(selected);delta=sx[sel]-target;pair=sx[sel,None]-sx[None,sel]
        raw=np.empty(len(cats))
        for k,ranges in enumerate(calibration.ranges):
            cov=np.exp(-3*np.linalg.norm(pair/ranges,axis=2));cov.flat[::len(sel)+1]+=1e-8
            cross=np.exp(-3*np.linalg.norm(delta/ranges,axis=1))
            w=np.linalg.solve(cov,cross)
            raw[k]=prior[k]+w@((encoded[sel]==k)-prior[k])
        clipped=np.maximum(raw,0.);p=clipped/clipped.sum() if clipped.sum()>1e-12 else prior.copy()
        probabilities[i]=p;corrections.append(float(np.abs(raw-p).sum()))
    return LithologyResult(cats,probabilities,None,dict(method='full_simple_indicator_kriging',
        backend='python_numpy_scipy',support=support,probability_correction_l1=corrections,
        calibration=calibration.diagnostics,search_ranges_m=search.tolist(),
        max_neighbors=max_neighbors,max_per_hole=max_per_hole,min_holes=min_holes))
