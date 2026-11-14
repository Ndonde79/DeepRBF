# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Resume parity, crash recovery, masks and input identity on real simulators."""
import numpy as np
import pytest
from deeprbf import simulate_pgs_ensemble, simulate_snesim_ensemble
from deeprbf.snesim import SNESIMConfig, UNKNOWN, simulate_snesim
from deeprbf.lithology import simulate_plurigaussian
import deeprbf.ensemble as ensemble


def pgs(path, n, batch=2, **extra):
    return simulate_pgs_ensemble([[0.,0,0],[1.,0,0]],[10,20],[[0.,0,0],[.5,0,0],[1.,0,0]],
        n_realizations=n,checkpoint_path=path,batch_size=batch,seed=71,bands=32,gibbs_sweeps=5,
        truncation_bounds=np.array([[[-np.inf,0],[-np.inf,np.inf]],[[0,np.inf],[-np.inf,np.inf]]]),**extra)


def mps(path, n, batch=2, backend='native', **extra):
    ti=np.indices((7,7)).sum(0)%2
    mask=np.ones((5,5),bool);mask[-1,-1]=False
    hard=np.full((5,5),UNKNOWN,int);hard[0,0]=1
    return simulate_snesim_ensemble(ti,(5,5),n_realizations=n,checkpoint_path=path,
        batch_size=batch,seed=71,backend=backend,simulation_mask=mask,hard_data=hard,
        config=SNESIMConfig(search_radii=(1,1,1),n_multigrids=1,min_replicates=1),**extra)


@pytest.mark.parametrize('runner',[pgs,mps])
def test_restart_batch_independence_and_hard_data(tmp_path,runner):
    runner(tmp_path/'resume.npz',2)
    resumed=runner(tmp_path/'resume.npz',5,3)
    fresh=runner(tmp_path/'fresh.npz',5,1)
    np.testing.assert_array_equal(resumed['probabilities'],fresh['probabilities'])
    assert resumed['maximum_wilson_95_halfwidth']>0
    if runner is pgs:
        np.testing.assert_array_equal(resumed['probabilities'][[0,2]],np.eye(2))
    else:
        np.testing.assert_array_equal(resumed['probabilities'][0,0],[0,1])
        assert np.isnan(resumed['probabilities'][-1,-1]).all()
    with pytest.raises(ValueError,match='smaller'):runner(tmp_path/'resume.npz',4)


def test_python_mps_soft_maps_resume(tmp_path):
    soft=np.full((5,5,2),.5)
    mps(tmp_path/'python.npz',1,backend='python',soft_probabilities=soft)
    actual=mps(tmp_path/'python.npz',3,backend='python',soft_probabilities=soft)
    expected=mps(tmp_path/'fresh.npz',3,1,backend='python',soft_probabilities=soft)
    np.testing.assert_array_equal(actual['probabilities'],expected['probabilities'])
    soft[2,2]=[.7,.3]
    with pytest.raises(ValueError,match='fingerprint'):
        mps(tmp_path/'python.npz',4,backend='python',soft_probabilities=soft)


@pytest.mark.parametrize('runner',[pgs,mps])
def test_failed_atomic_commit_recovers(tmp_path,monkeypatch,runner):
    path=tmp_path/'resume.npz';runner(path,2)
    replace=ensemble.os.replace
    def fail(*args):raise OSError('injected interruption')
    monkeypatch.setattr(ensemble.os,'replace',fail)
    with pytest.raises(OSError,match='injected'):runner(path,4)
    with np.load(path) as state:
        assert ensemble.json.loads(str(state['metadata']))['completed']==2
    monkeypatch.setattr(ensemble.os,'replace',replace)
    resumed=runner(path,4)
    expected=runner(tmp_path/'fresh.npz',4,1)
    np.testing.assert_array_equal(resumed['probabilities'],expected['probabilities'])


@pytest.mark.parametrize('runner',[pgs,mps])
def test_corruption_and_implementation_change_rejected(tmp_path,monkeypatch,runner):
    path=tmp_path/'state.npz';runner(path,2)
    original=ensemble._implementation_identity
    monkeypatch.setattr(ensemble,'_implementation_identity',lambda:{'changed':True})
    with pytest.raises(ValueError,match='fingerprint'):runner(path,3)
    monkeypatch.setattr(ensemble,'_implementation_identity',original)
    with np.load(path) as state:metadata=str(state['metadata']);counts=state['counts'].copy()
    counts.flat[0]+=1
    np.savez(path,counts=counts,metadata=metadata)
    with pytest.raises(ValueError,match='invariant'):runner(path,3)


def test_pgs_matches_documented_single_draw_seed_schedule(tmp_path):
    result=pgs(tmp_path/'pgs.npz',3)
    draws=[]
    for i in range(3):
        draws.append(simulate_plurigaussian([[0.,0,0],[1.,0,0]],[10,20],[[0.,0,0],[.5,0,0],[1.,0,0]],
            seed=71+104729*i,n_realizations=1,bands=32,gibbs_sweeps=5,
            truncation_bounds=np.array([[[-np.inf,0],[-np.inf,np.inf]],[[0,np.inf],[-np.inf,np.inf]]])).realizations[0])
    expected=np.mean(np.asarray(draws)[:,:,None]==np.array([10,20]),axis=0)
    np.testing.assert_array_equal(result['probabilities'],expected)


def test_pgs_explicit_catalogue_retains_unobserved_categories(tmp_path):
    bounds=np.array([[[-np.inf,-.5],[-np.inf,np.inf]],[[-.5,.5],[-np.inf,np.inf]],[[.5,np.inf],[-np.inf,np.inf]]])
    settings=dict(category_catalogue=np.array([10,20,30]),proportions={10:.3,20:.4,30:.3},
                  truncation_bounds=bounds,bands=32,gibbs_sweeps=5)
    result=simulate_pgs_ensemble([[0.,0,0],[1,0,0]],[10,30],[[0.,0,0],[1,0,0],[20,20,20]],
        n_realizations=3,checkpoint_path=tmp_path/'catalogue.npz',**settings)
    np.testing.assert_array_equal(result['categories'],[10,20,30])
    np.testing.assert_array_equal(result['probabilities'][:2],[[1,0,0],[0,0,1]])
    settings.pop('proportions')
    with pytest.raises(ValueError,match='proportions'):
        simulate_plurigaussian([[0.,0,0],[1,0,0]],[10,30],[[2,0,0]],n_realizations=1,**settings)
