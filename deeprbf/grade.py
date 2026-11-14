# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Controlled native continuous-grade estimation with explicit support geometry."""
import numpy as np
from deeprbf_recovered import RotatedSupport,audited_ordinary_kriging


class GradeEstimator:
    """Native OK or constant-augmented Gaussian RBF under identical searches.

    Gaussian RBF uses the same augmented kernel equations as Gaussian OK; its
    variance is returned only as a kernel diagnostic, not calibrated uncertainty.
    Covariance/search specifications are required, never silently fitted here.
    """
    def __init__(self,xyz,grades,hole_ids,domains,*,covariance,search,method='ordinary_kriging',regularization=0.,observation_support=None):
        self.xyz=np.asarray(xyz,float);self.grades=np.asarray(grades,float)
        if self.xyz.shape!=(len(self.grades),3) or not len(self.grades) or not np.isfinite(self.xyz).all() or not np.isfinite(self.grades).all() or (self.grades<0).any():raise ValueError('Finite nonnegative audited grades and XYZ required')
        if np.asarray(hole_ids).shape!=self.grades.shape or np.asarray(domains).shape!=self.grades.shape:raise ValueError('One hole/domain ID per observation required')
        if method not in ('ordinary_kriging','gaussian_rbf'):raise ValueError('Unknown grade estimator')
        if method=='gaussian_rbf' and covariance.family!='gaussian':raise ValueError('Gaussian RBF requires an explicit Gaussian covariance specification')
        covariance.validate();search.validate()
        if not np.isfinite(regularization) or regularization<0:raise ValueError('Nonnegative explicit regularization required')
        domain_array=np.asarray(domains)
        if not np.issubdtype(domain_array.dtype,np.integer) or (domain_array<np.iinfo(np.int32).min).any() or (domain_array>np.iinfo(np.int32).max).any():
            raise ValueError('Sample domains must be exact int32 identifiers; fractional codes cannot be truncated')
        self.holes=np.unique(np.asarray(hole_ids).astype(str),return_inverse=True)[1]
        self.domains=domain_array.astype(np.int32);self.covariance=covariance;self.search=search
        self.regularization=regularization;self.method=method
        self.observation_support=[] if observation_support is None else observation_support

    def predict(self,centers,*,target_domains=None,block_sizes=None,block_axes=None,discretization=(1,1,1)):
        centers=np.asarray(centers,float)
        if centers.ndim!=2 or centers.shape[1]!=3 or not np.isfinite(centers).all():raise ValueError('Finite target XYZ required')
        sizes=np.zeros_like(centers) if block_sizes is None else np.broadcast_to(np.asarray(block_sizes,float),centers.shape)
        if not np.isfinite(sizes).all() or (sizes<0).any():raise ValueError('Finite nonnegative support dimensions required')
        if self.search.restrict_domain and target_domains is None:raise ValueError('Explicit target domains required by domain-restricted search')
        domains=np.zeros(len(centers),int) if target_domains is None else np.asarray(target_domains)
        if domains.shape!=(len(centers),) or not np.issubdtype(domains.dtype,np.integer):raise ValueError('Integer target domain per target required')
        targets=[]
        for center,size,domain in zip(centers,sizes,domains):
            block=RotatedSupport();block.center=center;block.size=size;block.domain=int(domain);block.discretization=discretization
            if block_axes is not None:block.axes=block_axes
            targets.append(block)
        result=audited_ordinary_kriging(self.xyz,self.grades,self.holes,self.domains,targets,self.covariance,self.search,
            self.regularization,observation_support=self.observation_support)
        raw=np.asarray(result.estimate);estimate=raw.copy();variance=np.asarray(result.variance)
        status=np.array([d.status for d in result.diagnostics],object)
        negative=np.isfinite(raw)&(raw<0);status[negative]='NEGATIVE_GRADE';estimate[negative]=np.nan
        return dict(estimate=estimate,raw_estimate=raw,variance=variance,status=status,
            n_samples=np.array([d.n_samples for d in result.diagnostics]),n_holes=np.array([d.n_holes for d in result.diagnostics]),
            residual=np.array([d.relative_residual for d in result.diagnostics]),
            minimum_weight=np.array([d.minimum_weight if d.success else np.nan for d in result.diagnostics]),
            maximum_weight=np.array([d.maximum_weight if d.success else np.nan for d in result.diagnostics]),
            weight_sum=np.array([d.weight_sum if d.success else np.nan for d in result.diagnostics]),
            weight_l1=np.array([d.weight_l1 if d.success else np.nan for d in result.diagnostics]),
            negative_weights=np.array([d.negative_weights if d.success else -1 for d in result.diagnostics]),
            supported=np.isfinite(estimate),method=self.method,regularization=self.regularization,
            variance_interpretation='Model-based kriging variance' if self.method=='ordinary_kriging' else 'Uncalibrated Gaussian kernel diagnostic')
