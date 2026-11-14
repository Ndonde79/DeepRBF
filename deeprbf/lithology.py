# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Auditable reference IK, sequential indicator simulation and rectangular PGS.

All distances are in the same geological coordinate system. IK uses a common
exponential indicator correlation; PGS uses finite spectral Gaussian fields and
Gibbs-constrained latent conditioning. These are comparison implementations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr,ndtri,log_ndtr,ndtri_exp
from scipy.linalg import cho_factor,cho_solve


@dataclass
class LithologyResult:
    """Node probabilities and simulations in original integer category codes."""
    categories: np.ndarray
    probabilities: np.ndarray
    realizations: np.ndarray | None
    diagnostics: dict

    @property
    def modal(self):
        return self.categories[self.probabilities.argmax(axis=1)]


def _prepare(sample_xyz, sample_categories, target_xyz, ranges, rotation, proportions, category_catalogue=None):
    samples=np.asarray(sample_xyz,dtype=float);targets=np.asarray(target_xyz,dtype=float)
    labels=np.asarray(sample_categories)
    if samples.ndim!=2 or samples.shape[1]!=3 or not len(samples):raise ValueError('sample_xyz must be nonempty N by 3')
    if targets.ndim!=2 or targets.shape[1]!=3 or not len(targets):raise ValueError('target_xyz must be nonempty N by 3')
    if labels.ndim!=1 or len(labels)!=len(samples):raise ValueError('One category per sample is required')
    if not np.issubdtype(labels.dtype,np.integer):raise ValueError('Categories must be integer codes')
    if not np.isfinite(samples).all() or not np.isfinite(targets).all():raise ValueError('Coordinates must be finite')
    ranges=np.asarray(ranges,float);rotation=np.eye(3) if rotation is None else np.asarray(rotation,float)
    if ranges.shape!=(3,) or not np.isfinite(ranges).all() or (ranges<=0).any():raise ValueError('ranges must contain three finite positive values')
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rotation),1.):
        raise ValueError('rotation rows must be right-handed orthonormal world axes')
    categories=np.unique(labels) if category_catalogue is None else np.asarray(category_catalogue)
    if categories.ndim!=1 or not np.issubdtype(categories.dtype,np.integer) or not np.array_equal(categories,np.unique(categories)) or not np.isin(labels,categories).all():
        raise ValueError('Category catalogue must be sorted unique integers containing every sample category')
    if category_catalogue is not None and proportions is None and len(categories)!=len(np.unique(labels)):
        raise ValueError('Explicit positive proportions required for categories absent from hard data')
    if len(categories)<2:raise ValueError('At least two lithology categories are required')
    all_xyz,inverse=np.unique(np.vstack([samples,targets]),axis=0,return_inverse=True)
    hard=np.full(len(all_xyz),-1,dtype=int);encoded=np.searchsorted(categories,labels)
    for index,value in zip(inverse[:len(samples)],encoded):
        if hard[index]>=0 and hard[index]!=value:raise ValueError('Conflicting collocated sample categories')
        hard[index]=value
    if proportions is None:
        prior=np.bincount(hard[hard>=0],minlength=len(categories)).astype(float);prior/=prior.sum()
    else:
        if set(proportions)!=set(categories.tolist()):raise ValueError('Proportions must specify every observed category exactly')
        prior=np.array([proportions[int(c)] for c in categories],float)
        if not np.isfinite(prior).all() or (prior<=0).any() or not np.isclose(prior.sum(),1.):raise ValueError('Proportions must be positive and sum to one')
    x=(all_xyz-all_xyz.mean(0))@rotation.T/ranges
    return x,hard,categories,prior,inverse[len(samples):]


def estimate_lithology_ik(sample_xyz,sample_categories,target_xyz,*,ranges=(160.,70.,30.),rotation=None,proportions=None,max_neighbors=20,backend='native',hole_ids=None,max_per_hole=0,min_holes=0):
    """Local simple IK with common exponential correlation and audited outputs.

    Search radius: two scaled range units. Probabilities are clipped/normalized;
    unsupported targets return the prior. Separate from native indicator_kriging.
    """
    if not isinstance(max_neighbors,int) or max_neighbors<1:raise ValueError('max_neighbors must be positive integer')
    x,hard,categories,prior,target=_prepare(sample_xyz,sample_categories,target_xyz,ranges,rotation,proportions)
    if backend not in ('native','python'):raise ValueError('backend must be native or python')
    native_diagnostics={}
    if backend=='native':
        from deeprbf_recovered import indicator_lithology_native
        result=indicator_lithology_native(x,hard,prior,max_neighbors,hole_ids=_node_holes(sample_xyz,target_xyz,hole_ids),max_per_hole=max_per_hole,min_holes=min_holes)
        p=np.asarray(result.probabilities).reshape(len(x),len(prior))
        native_diagnostics=dict(maximum_solve_residual=result.maximum_solve_residual,
            selected_hard_holes=np.asarray(result.direct_hard_holes)[target].tolist(),
            supported=np.asarray(result.supported,dtype=bool)[target].tolist(),min_holes=min_holes,max_per_hole=max_per_hole)
    else:
        if hole_ids is not None or max_per_hole or min_holes:raise ValueError('Drillhole support controls require backend=native')
        p=indicator_kriging(x,hard,prior,k=max_neighbors)
    diagnostics=_audit_lithology(x,hard,target,p,None,ranges,max_neighbors)
    return LithologyResult(categories,p[target],None,dict(method='simple_indicator_kriging',backend='native_cpp_sparse' if backend=='native' else 'python_numpy_scipy',prior=prior.tolist(),**diagnostics,**native_diagnostics))


def simulate_lithology_sis(sample_xyz,sample_categories,target_xyz,*,ranges=(160.,70.,30.),rotation=None,
                            proportions=None,n_realizations=10,seed=42,max_neighbors=20,backend='native',hole_ids=None,max_per_hole=0,min_holes=0):
    """Sequential local IK simulation; realizations shape (realization,target).

    Uses 96 grid candidates plus nearest hard samples. Independent target chunks
    are not equivalent to a joint simulation. Seed fixes the complete target set.
    """
    if not isinstance(n_realizations,int) or n_realizations<1:raise ValueError('n_realizations must be positive integer')
    if not isinstance(max_neighbors,int) or max_neighbors<1:raise ValueError('max_neighbors must be positive integer')
    x,hard,categories,prior,target=_prepare(sample_xyz,sample_categories,target_xyz,ranges,rotation,proportions)
    if backend not in ('native','python'):raise ValueError('backend must be native or python')
    native_diagnostics={}
    if backend=='native':
        from deeprbf_recovered import indicator_lithology_native
        result=indicator_lithology_native(x,hard,prior,max_neighbors,n_realizations,seed,hole_ids=_node_holes(sample_xyz,target_xyz,hole_ids),max_per_hole=max_per_hole,min_holes=min_holes)
        sim=np.asarray(result.simulations).reshape(n_realizations,len(x))
        native_diagnostics=dict(maximum_solve_residual=result.maximum_solve_residual,
            prior_only_draws=list(result.prior_only_draws),random_generator='mt19937_64; Fisher-Yates rejection; 53-bit categorical draws',
            selected_hard_holes=np.asarray(result.direct_hard_holes)[target].tolist(),
            supported=np.asarray(result.supported,dtype=bool)[target].tolist(),min_holes=min_holes,max_per_hole=max_per_hole)
    else:
        if hole_ids is not None or max_per_hole or min_holes:raise ValueError('Drillhole support controls require backend=native')
        sim=sequential_indicator(x,hard,prior,n_realizations,seed,k=max_neighbors)
    p=np.mean(sim[:,:,None]==np.arange(len(categories)),axis=0)
    diagnostics=_audit_lithology(x,hard,target,p,sim,ranges,max_neighbors)
    return LithologyResult(categories,p[target],categories[sim[:,target]],dict(method='sequential_indicator_simulation',backend='native_cpp_sparse' if backend=='native' else 'python_numpy_scipy',seed=seed,**diagnostics,**native_diagnostics))


def simulate_plurigaussian(sample_xyz,sample_categories,target_xyz,*,ranges=(160.,70.,30.),rotation=None,
                            proportions=None,n_realizations=10,seed=42,bands=768,gibbs_sweeps=30,
                            latent_nugget=.01,truncation_bounds=None,max_conditioning_nodes=3000,max_workspace_mb=512,gibbs_backend='native',category_catalogue=None):
    """Two-field finite-spectral PGS with Gibbs-constrained latent conditioning.

    Bounds shape (category,2,2), sorted category order, lower/upper per field.
    The default rule covers the example category codes 100-1400 only; supply a partition for other codes.
    Dense conditioning costs O(N^3); assess Gibbs and ensemble convergence.
    """
    for value,name in [(n_realizations,'n_realizations'),(bands,'bands'),(gibbs_sweeps,'gibbs_sweeps')]:
        if not isinstance(value,int) or value<1:raise ValueError(name+' must be positive integer')
    if not np.isfinite(latent_nugget) or not 0<latent_nugget<1:raise ValueError('latent_nugget must be between zero and one')
    x,hard,categories,prior,target=_prepare(sample_xyz,sample_categories,target_xyz,ranges,rotation,proportions,category_catalogue)
    if np.sum(hard>=0)>max_conditioning_nodes:raise ValueError('Dense PGS conditioning limit exceeded')
    sim,bounds,diagnostics=plurigaussian(x,hard,categories,prior,n_realizations,seed,bands,gibbs_sweeps,latent_nugget,truncation_bounds,max_workspace_mb=max_workspace_mb,gibbs_backend=gibbs_backend)
    p=np.mean(sim[:,:,None]==np.arange(len(categories)),axis=0)
    audit=_audit_lithology(x,hard,target,p,sim,ranges,None)
    return LithologyResult(categories,p[target],categories[sim[:,target]],dict(method='rectangular_plurigaussian',backend='hybrid_native_gibbs_numpy_scipy' if gibbs_backend=='native' else 'python_numpy_scipy',
                           seed=seed,truncation_bounds=bounds,realizations=diagnostics,**audit))


def _node_holes(samples,targets,hole_ids):
    if hole_ids is None:return []
    holes=np.asarray(hole_ids)
    if holes.shape!=(len(samples),):raise ValueError('One hole ID per sample required')
    _,encoded=np.unique(holes.astype(str),return_inverse=True)
    unique,inverse=np.unique(np.vstack([samples,targets]),axis=0,return_inverse=True)
    result=np.full(len(unique),-1,int)
    # Coincident records from different holes count conservatively as one hard
    # node; no artificial additional hole support is created by duplicates.
    for index,hole in zip(inverse[:len(samples)],encoded):
        if result[index]<0:result[index]=hole
    return result


def _audit_lithology(x,hard,target,probabilities,simulations,ranges,k):
    """Reject invalid results before publication; report direct hard-data support.

    SIS may propagate simulated information beyond direct hard support. PGS is
    globally conditioned, so a local support threshold is not asserted for PGS.
    """
    known=np.flatnonzero(hard>=0)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1.,rtol=0,atol=1e-10):
        raise RuntimeError('Invalid lithology probabilities')
    expected=np.eye(probabilities.shape[1])[hard[known]]
    if not np.array_equal(probabilities[known],expected):
        raise RuntimeError('Lithology hard-conditioning probability audit failed')
    if simulations is not None and np.any(simulations[:,known]!=hard[known]):
        raise RuntimeError('Lithology hard-conditioning simulation audit failed')
    out=dict(hard_conditioning_status='PASS',hard_mismatches=0,
             unique_hard_nodes=int(len(known)),range_convention='practical',ranges=list(map(float,ranges)))
    if k is not None:
        distances,_=cKDTree(x[known]).query(x[target],k=min(k,len(known)),distance_upper_bound=2.+1e-12)
        counts=np.isfinite(np.asarray(distances).reshape(len(target),-1)).sum(1)
        out.update(covariance='exp(-3*h)',search_radius_scaled=2.,
                   direct_hard_neighbor_count=counts.tolist(),
                   without_direct_hard_support_indices=np.flatnonzero(counts==0).tolist(),
                   support_definition='Hard samples within search radius, capped at max_neighbors; not a resource classification')
    else:
        out.update(covariance='finite spectral approximation to exp(-3*h^2), plus nugget',
                   support_definition='Global dense latent conditioning; local support not assessed')
    return out


def normalize(p):
    p=np.maximum(p,0.)
    return p/p.sum() if p.sum()>1e-12 else np.full(len(p),1/len(p))


def ik_probability(q,x,labels,prior):
    if not len(x):return prior.copy()
    d=np.linalg.norm(x-q,axis=1)
    if d.min()<1e-9:
        p=np.zeros(len(prior));p[labels[d.argmin()]]=1.;return p
    cov=np.exp(-3*np.linalg.norm(x[:,None]-x[None,:],axis=2))
    cov.flat[::len(x)+1]+=1e-7
    cross=np.exp(-3*d)
    weights=np.linalg.solve(cov,cross)
    # Simple indicator kriging, common normalized correlation, explicit prior.
    indicators=np.eye(len(prior))[labels]
    return normalize(prior+weights@(indicators-prior))


def indicator_kriging(x,hard,prior,k=20):
    idx=np.flatnonzero(hard>=0);neighbors=_canonical_neighbors(x[idx],x,k)
    probabilities=np.empty((len(x),len(prior)))
    for i in range(len(x)):
        select=idx[neighbors[i]]
        probabilities[i]=ik_probability(x[i],x[select],hard[select],prior)
    return probabilities


def sequential_indicator(x,hard,prior,n_realizations,seed,k=20):
    # Fixed candidate graph plus nearest hard nodes; selection depends on informed state.
    near=_canonical_neighbors(x,x,96)
    hard_idx=np.flatnonzero(hard>=0);nh=_canonical_neighbors(x[hard_idx],x,k)
    sims=[]
    for realization in range(n_realizations):
        rng=np.random.default_rng(seed+realization*104729);sim=hard.copy()
        for i in rng.permutation(np.flatnonzero(hard<0)):
            candidates=near[i];candidates=candidates[sim[candidates]>=0]
            fixed=hard_idx[nh[i]]
            candidates=np.unique(np.r_[candidates,fixed])
            if len(candidates):
                distance=np.linalg.norm(x[candidates]-x[i],axis=1)
                candidates=candidates[np.lexsort((candidates,np.rint(distance*1e12)))[:k]]
            p=ik_probability(x[i],x[candidates],sim[candidates],prior)
            sim[i]=rng.choice(len(prior),p=p)
        sims.append(sim)
    return np.array(sims)


def _canonical_neighbors(points,queries,k,radius=2.):
    """Nearest nodes with a 1e-12 scaled-distance tie/bin boundary convention."""
    tree=cKDTree(points);k=min(k,len(points));probe=min(k+1,len(points))
    distances,indices=tree.query(queries,k=probe,distance_upper_bound=radius+1e-12)
    distances=np.asarray(distances).reshape(len(queries),probe);indices=np.asarray(indices).reshape(len(queries),probe)
    result=[]
    for q,dd,ii in zip(queries,distances,indices):
        finite=np.isfinite(dd);dd=dd[finite];ii=ii[finite]
        if len(dd)>k and np.rint(dd[k-1]*1e12)==np.rint(dd[k]*1e12):
            ii=np.asarray(tree.query_ball_point(q,min(radius+1e-12,dd[k-1]+1e-12)),int)
            dd=np.linalg.norm(points[ii]-q,axis=1)
        result.append(ii[np.lexsort((ii,np.rint(dd*1e12)))[:k]])
    return result


def lithotype_rectangles(categories,prior):
    """Three X families; Y partitions within each. Observed masses reproduced."""
    families=[[1100],[100,200,300,400],[500,600,700,800,900,1000,1300,1400]]
    families=[[int(np.flatnonzero(categories==c)[0]) for c in family if c in categories] for family in families]
    families=[f for f in families if f]
    if sorted(i for f in families for i in f)!=list(range(len(categories))):
        raise ValueError('Default rule supports the example category codes only; supply truncation_bounds')
    x_edges=ndtri(np.r_[0.,np.cumsum([prior[f].sum() for f in families])].clip(0,1))
    x_edges[0]=-np.inf;x_edges[-1]=np.inf
    bounds=np.zeros((len(prior),2,2))
    for family_index,family in enumerate(families):
        y_edges=ndtri(np.r_[0.,np.cumsum(prior[family]/prior[family].sum())].clip(0,1))
        y_edges[0]=-np.inf;y_edges[-1]=np.inf
        for pos,c in enumerate(family):
            bounds[c,0]=x_edges[family_index:family_index+2]
            bounds[c,1]=y_edges[pos:pos+2]
    return bounds


def truncate(fields,bounds):
    result=np.full(fields.shape[0],-1,dtype=int)
    for c,b in enumerate(bounds):
        mask=((fields>=b[:,0])&(fields<b[:,1])).all(1);result[mask]=c
    if (result<0).any():raise RuntimeError('Incomplete truncation rule')
    return result


def sample_truncated(mean,sd,lower,upper,rng):
    a=(lower-mean)/sd;b=(upper-mean)/sd
    reflect=a>0
    lo=np.where(reflect,-b,a);hi=np.where(reflect,-a,b)
    u=np.clip(rng.random(np.shape(mean)),1e-15,1-1e-15)
    logp=np.logaddexp(log_ndtr(lo)+np.log1p(-u),log_ndtr(hi)+np.log(u))
    z=ndtri_exp(logp);value=mean+sd*np.where(reflect,-z,z)
    return np.maximum(np.nextafter(lower,np.inf),np.minimum(np.nextafter(upper,-np.inf),value))


def plurigaussian(x,hard,categories,prior,n_realizations,seed,bands=768,sweeps=30,latent_nugget=.01,truncation_bounds=None,*,max_workspace_mb=512,gibbs_backend='python'):
    idx=np.flatnonzero(hard>=0);bounds=lithotype_rectangles(categories,prior) if truncation_bounds is None else np.asarray(truncation_bounds,float)
    if gibbs_backend not in ('native','python'):raise ValueError('gibbs_backend must be native or python')
    # Conservative accounting for two retained bases/factors/precisions, solve
    # temporaries, output realizations and the largest field-evaluation chunk.
    effective_bands=max(bands,len(idx))
    workspace_bytes=8*(10*len(idx)**2+10*len(idx)*effective_bands+
        len(x)*(n_realizations+8)+6000*effective_bands)
    if not np.isfinite(max_workspace_mb) or max_workspace_mb<=0:raise ValueError('Positive finite PGS workspace limit required')
    if workspace_bytes>max_workspace_mb*1024**2:
        raise ValueError(f'PGS estimated workspace {workspace_bytes/1024**2:.1f} MiB exceeds {max_workspace_mb} MiB; reduce conditioning/bands or use another method')
    if bounds.shape!=(len(categories),2,2) or np.isnan(bounds).any() or (bounds[:,:,0]>=bounds[:,:,1]).any():raise ValueError('Invalid truncation bounds')
    cuts=[np.unique(bounds[:,dim,:]) for dim in range(2)]
    if any(v[0]!=-np.inf or v[-1]!=np.inf for v in cuts):raise ValueError('Rule must cover the latent plane')
    probes=[]
    for values in cuts:
        probes.append([0. if a==-np.inf and b==np.inf else b-1 if a==-np.inf else a+1 if b==np.inf else (a+b)/2 for a,b in zip(values[:-1],values[1:])])
    for a in probes[0]:
        for b in probes[1]:
            point=np.array([a,b]);matches=((point>=bounds[:,:,0])&(point<bounds[:,:,1])).all(1)
            if matches.sum()!=1:raise ValueError('Truncation rectangles have gaps or overlaps')
    bands=max(bands,len(idx))
    simulations=[];diagnostics=[]
    rng_spectrum=np.random.default_rng(seed+923)
    bases=[]
    for field in range(2):
        # Independent latent fields with distinct continuity; Gaussian covariance.
        frequencies=rng_spectrum.normal(size=(3,bands))*np.sqrt(6)/(1. if field==0 else .65)
        projection=x[idx]@frequencies
        basis=np.concatenate([np.cos(projection),np.sin(projection)],axis=1)*np.sqrt((1-latent_nugget)/bands)
        h=basis;cov=h@h.T
        cov.flat[::len(idx)+1]+=latent_nugget
        try: factor=cho_factor(cov,lower=True)
        except np.linalg.LinAlgError:raise RuntimeError('Spectral conditioning covariance is singular; refine bands/grid')
        precision=cho_solve(factor,np.eye(len(idx)));bases.append((frequencies,h,factor,precision))
    for r in range(n_realizations):
        rng=np.random.default_rng(seed+104729*r);fields=[];hard_errors=0
        for dim,(frequencies,h,factor,precision) in enumerate(bases):
            lower=bounds[hard[idx],dim,0];upper=bounds[hard[idx],dim,1]
            latent=sample_truncated(np.zeros(len(idx)),np.ones(len(idx)),lower,upper,rng)
            if gibbs_backend=='native':
                from deeprbf_recovered import gibbs_truncated_native
                latent=np.asarray(gibbs_truncated_native(precision.ravel(),lower,upper,sweeps,seed+104729*r+7919*(dim+1)))
            else:
                residual=precision@latent;diag=np.diag(precision)
                for sweep in range(sweeps):
                    for i in rng.permutation(len(idx)):
                        mean=latent[i]-residual[i]/diag[i];sd=1/np.sqrt(diag[i])
                        value=sample_truncated(mean,sd,lower[i],upper[i],rng)
                        delta=value-latent[i];latent[i]=value;residual+=precision[:,i]*delta
            coeff=rng.normal(size=h.shape[1]);noise=rng.normal(size=len(x))*np.sqrt(latent_nugget)
            correction=cho_solve(factor,latent-h@coeff-noise[idx]);coeff+=h.T@correction
            field=np.empty(len(x))
            for start in range(0,len(x),1000):
                projection=x[start:start+1000]@frequencies
                field[start:start+1000]=(np.cos(projection)@coeff[:bands]+np.sin(projection)@coeff[bands:])*np.sqrt((1-latent_nugget)/bands)
            field+=noise;field[idx]+=latent_nugget*correction
            if not np.allclose(field[idx],latent,atol=1e-6):raise RuntimeError('Latent conditioning residual too large')
            # Known Gaussian values are exact constraints; remove solve roundoff.
            field[idx]=latent
            fields.append(field)
        sim=truncate(np.array(fields).T,bounds)
        hard_errors=int(np.sum(sim[idx]!=hard[idx]))
        if hard_errors:raise RuntimeError(f'PGS conditioning failed at {hard_errors} nodes')
        simulations.append(sim);diagnostics.append(dict(hard_mismatches=hard_errors,gibbs_sweeps=sweeps,bands=bands,latent_nugget=latent_nugget,estimated_workspace_bytes=workspace_bytes,gibbs_backend=gibbs_backend))
    return np.array(simulations),bounds,diagnostics
