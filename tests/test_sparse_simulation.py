# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pytest
from deeprbf import vecchia_truncated_native,simulate_plurigaussian_sparse
from deeprbf.ensemble import simulate_pgs_ensemble,simulate_snesim_ensemble
from deeprbf.snesim import simulate_snesim,SNESIMConfig,UNKNOWN


def test_sparse_unconstrained_joint_covariance_matches_dense_gaussian():
    x=np.array([[0.,0,0],[.3,0,0],[0,.5,0]])
    draws=np.array([vecchia_truncated_native(x,[-np.inf]*3,[np.inf]*3,2,2,int(seed),.1).values for seed in range(600)])
    expected=.9*np.exp(-3*np.sum((x[:,None]-x)**2,axis=2))+.1*np.eye(3)
    np.testing.assert_allclose(draws.mean(0),0,atol=.13)
    np.testing.assert_allclose(np.cov(draws.T),expected,atol=.15)


def test_sparse_truncated_marginal_and_memory_guard():
    samples=[vecchia_truncated_native([[0,0,0]],[0],[np.inf],1,3,seed).values[0] for seed in range(600)]
    assert abs(np.mean(samples)-np.sqrt(2/np.pi))<.09
    with pytest.raises(ValueError,match='workspace'):
        vecchia_truncated_native(np.zeros((3001,3)),[-np.inf]*3001,[np.inf]*3001,12,2,42,.01,1)


def test_sparse_pgs_supports_more_than_dense_hard_limit():
    x=np.c_[np.arange(3100)*3.,np.zeros((3100,2))];labels=np.resize([10,20],3100)
    bounds=np.array([[[-np.inf,0],[-np.inf,np.inf]],[[0,np.inf],[-np.inf,np.inf]]])
    result=simulate_plurigaussian_sparse(x,labels,x,ranges=(1,1,1),neighbors=4,gibbs_sweeps=3,truncation_bounds=bounds)
    np.testing.assert_array_equal(result.realizations[0],labels)
    assert all(d['precision_nonzeros']<3100*25 for d in result.diagnostics['fields'])


def test_native_soft_mps_respects_hard_data_and_soft_trend():
    ti=np.indices((10,10)).sum(0)%2;hard=np.full((8,8),UNKNOWN,int);hard[0,0]=0
    soft=np.zeros((8,8,2));soft[:,:,1]=1
    # Force prior backoff so both categories have nonzero TI probability;
    # Bayesian soft data should not invent a TI-forbidden pattern.
    cfg=SNESIMConfig(search_radii=(1,1,1),n_multigrids=1,min_replicates=100000,soft_tau=5)
    values,_=simulate_snesim(ti,hard.shape,hard_data=hard,soft_probabilities=soft,config=cfg,backend='native')
    assert values[0,0,0]==0 and (values[0,1:]==1).mean()>.9


def test_retained_geometry_resume_and_corruption(tmp_path):
    ti=np.indices((6,6)).sum(0)%2;path=tmp_path/'mps.npz'
    kwargs=dict(checkpoint_path=path,retain_realizations=True,backend='native',config=SNESIMConfig(search_radii=(1,1,1),n_multigrids=1))
    simulate_snesim_ensemble(ti,(4,4),n_realizations=2,**kwargs)
    result=simulate_snesim_ensemble(ti,(4,4),n_realizations=4,**kwargs)
    draws=np.stack([np.load(path.with_name(path.name+'.draws')/f'{i:08d}.npy') for i in range(4)])
    np.testing.assert_array_equal(np.mean(draws[...,None]==result['categories'],axis=0),result['probabilities'])
    (path.with_name(path.name+'.draws')/'00000000.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):simulate_snesim_ensemble(ti,(4,4),n_realizations=5,**kwargs)
