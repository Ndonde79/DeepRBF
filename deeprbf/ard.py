# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Continuous ARD prediction and explicit conservative routing policies.

Dense Gaussian conditioning uses SciPy; this is not a native C++ GP kernel.
Point/new-assay uncertainty is retained, not represented as block fractions.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import beta

class LogNPRModel:
    def __init__(self,xyz,npr,*,ranges,nugget_fraction):
        self.xyz=np.asarray(xyz,float);values=np.asarray(npr,float);self.ranges=np.asarray(ranges,float)
        if self.xyz.shape!=(len(values),3) or len(values)<3 or not np.isfinite(self.xyz).all() or not np.isfinite(values).all() or (values<=0).any():raise ValueError('Finite XYZ and positive NPR required')
        if self.ranges.shape!=(3,) or not np.isfinite(self.ranges).all() or (self.ranges<=0).any():raise ValueError('Positive finite ranges required')
        if not np.isfinite(nugget_fraction) or not 0<nugget_fraction<1:raise ValueError('Nugget fraction must be between zero and one')
        self.nugget=float(nugget_fraction);y=np.log(values);self.mean=float(y.mean());self.sill=max(float(y.var()),1e-6)
        k=(1-self.nugget)*np.exp(-3*cdist(self.xyz/self.ranges,self.xyz/self.ranges));k[np.diag_indices_from(k)]+=self.nugget+1e-8
        self.factor=cho_factor(k);self.alpha=cho_solve(self.factor,y-self.mean)
        self.solve_residual=float(np.linalg.norm(k@self.alpha-(y-self.mean))/max(np.linalg.norm(y-self.mean),1e-15))

    def predict(self,xyz):
        q=np.asarray(xyz,float)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('Finite target XYZ required')
        cross=(1-self.nugget)*np.exp(-3*cdist(q/self.ranges,self.xyz/self.ranges))
        mean=self.mean+cross@self.alpha
        variance=self.sill*np.maximum(1-np.sum(cross*cho_solve(self.factor,cross.T).T,axis=1),1e-9)
        sd=np.sqrt(variance)
        return dict(log_mean=mean,log_variance=variance,npr_p10=np.exp(mean-1.2815515655446004*sd),npr_p50=np.exp(mean),npr_p90=np.exp(mean+1.2815515655446004*sd),p_npr_le_2=ndtr((np.log(2)-mean)/sd))

def select_nag_threshold(probability,adverse,holes,*,min_records=30,min_holes=10,max_upper_error=.10):
    """Choose largest passing cutoff on training OOF predictions only.

    Zero observed adverse decisions; one-sided 95% binomial bound <=10%.
    Binomial bound is a screening diagnostic, not a hole-cluster guarantee.
    Returns None when evidence is insufficient, never an invented cutoff.
    """
    p=np.asarray(probability,float);y=np.asarray(adverse,bool);h=np.asarray(holes)
    if p.shape!=y.shape or h.shape!=p.shape or p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Aligned finite probabilities, labels and holes required')
    passing=[]
    for cutoff in np.arange(.01,.501,.01):
        chosen=p<=cutoff;n=int(chosen.sum());errors=int(y[chosen].sum());nh=len(np.unique(h[chosen]))
        upper=1. if n==0 or errors==n else float(beta.ppf(.95,errors+1,n-errors))
        if n>=min_records and nh>=min_holes and errors==0 and upper<=max_upper_error:
            passing.append(dict(cutoff=float(cutoff),records=n,holes=nh,false_nag=errors,error_upper_95=upper))
    return passing[-1] if passing else None

def material_routes(ore,weathering,supported,p_npr_le_2,nag_threshold):
    ore=np.asarray(ore,bool);w=np.asarray(weathering);support=np.asarray(supported,bool);p=np.asarray(p_npr_le_2,float)
    if not (ore.shape==w.shape==support.shape==p.shape):raise ValueError('One policy value per block required')
    result=np.full(len(ore),'PAG_CONTROLLED',object)
    if nag_threshold is not None:
        if not 0<=nag_threshold<=1:raise ValueError('Invalid NAG threshold')
        result[(w=='PRIMARY')&support&np.isfinite(p)&(p<=nag_threshold)]='NAG'
    result[w=='OXIDE']='NAG';result[ore]='ORE'
    return result
