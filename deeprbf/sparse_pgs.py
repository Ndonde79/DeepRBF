# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Native sparse Vecchia latent PGS; an explicit approximation to dense PGS."""
import numpy as np
from .lithology import _prepare,lithotype_rectangles,truncate,LithologyResult
from deeprbf_recovered import vecchia_truncated_native


def validate_rule(bounds,count):
    bounds=np.asarray(bounds,float)
    if bounds.shape!=(count,2,2) or np.isnan(bounds).any() or (bounds[:,:,0]>=bounds[:,:,1]).any():raise ValueError('Invalid truncation bounds')
    probes=[]
    for dimension in range(2):
        cuts=np.unique(bounds[:,dimension,:])
        if cuts[0]!=-np.inf or cuts[-1]!=np.inf:raise ValueError('Rule must cover latent plane')
        probes.append([0. if a==-np.inf and b==np.inf else b-1 if a==-np.inf else a+1 if b==np.inf else (a+b)/2 for a,b in zip(cuts[:-1],cuts[1:])])
    for a in probes[0]:
        for b in probes[1]:
            if ((([a,b]>=bounds[:,:,0])&([a,b]<bounds[:,:,1])).all(1)).sum()!=1:raise ValueError('Rule has gaps or overlaps')
    return bounds


def simulate_plurigaussian_sparse(sample_xyz,sample_categories,target_xyz,*,ranges=(160.,70.,30.),rotation=None,
        proportions=None,category_catalogue=None,truncation_bounds=None,n_realizations=1,seed=42,
        neighbors=12,gibbs_sweeps=60,latent_nugget=.01,max_workspace_mb=512):
    if type(n_realizations) is not int or n_realizations<1 or type(seed) is not int or seed<0:raise ValueError('Invalid realization count/seed')
    x,hard,cats,prior,target=_prepare(sample_xyz,sample_categories,target_xyz,ranges,rotation,proportions,category_catalogue)
    if 8*len(x)*(n_realizations+8)>max_workspace_mb*1024**2:raise ValueError('Sparse output workspace exceeded')
    bounds=validate_rule(lithotype_rectangles(cats,prior) if truncation_bounds is None else truncation_bounds,len(cats))
    known=hard>=0;draws=[];diagnostics=[]
    reserved=8*len(x)*(n_realizations+8)/1024**2
    for draw in range(n_realizations):
        fields=[]
        for dimension in range(2):
            lower=np.full(len(x),-np.inf);upper=np.full(len(x),np.inf)
            lower[known]=bounds[hard[known],dimension,0];upper[known]=bounds[hard[known],dimension,1]
            result=vecchia_truncated_native(x/(1. if dimension==0 else .65),lower,upper,neighbors,gibbs_sweeps,
                seed+104729*draw+7919*(dimension+1),latent_nugget,max_workspace_mb-reserved)
            fields.append(np.asarray(result.values));diagnostics.append(dict(realization=draw,field=dimension,
                precision_nonzeros=result.precision_nonzeros,estimated_workspace_bytes=result.estimated_workspace_bytes+int(reserved*1024**2),
                maximum_local_residual=result.maximum_local_residual))
        encoded=truncate(np.asarray(fields).T,bounds)
        if not np.array_equal(encoded[known],hard[known]):raise RuntimeError('Sparse PGS hard conditioning failed')
        draws.append(cats[encoded[target]])
    sims=np.asarray(draws);p=np.mean(sims[:,:,None]==cats,axis=0)
    return LithologyResult(cats,p,sims,dict(backend='native_cpp_vecchia_sparse',neighbors=neighbors,gibbs_sweeps=gibbs_sweeps,
        hard_mismatches=0,fields=diagnostics,approximation='Fixed random ordering, nearest preceding subset within max(256,8*k) candidates',
        limitation='Approximate latent covariance; assess neighbor sensitivity, Gibbs mixing and geological continuity before production'))
