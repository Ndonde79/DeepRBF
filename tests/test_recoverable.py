# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pytest
from deeprbf.recoverable import recoverable_moments, aggregate_joint_realizations

def calculate(g=None,w=None,c=None,a=None):
    return recoverable_moments([[[.1,0],[.5,.2]]] if g is None else g,
        [[1,1]] if w is None else w, [0,.24,.4,1] if c is None else c,
        [1,.537] if a is None else a, support='20 x 20 x 16 m panel')

def test_analytic_metal_and_inclusive_selection():
    r=calculate()
    assert np.allclose(r.probability,[[1,.5,.5,0]])
    assert np.allclose(r.truncated_grade[0,1],[.25,.1])
    tonnes,metal=r.totals([100])
    assert tonnes.tolist()==[100,50,50,0]
    assert np.allclose(metal[1],[25,10])
    assert calculate(c=[.1]).probability[0,0]==1

def test_joint_metal_dependence_matters():
    same=recoverable_moments([[[0,0],[1,1]]],[[1,1]],[1.5],[1,1],support='block')
    opposite=recoverable_moments([[[0,1],[1,0]]],[[1,1]],[1.5],[1,1],support='block')
    assert same.probability[0,0]==.5 and opposite.probability[0,0]==0

def test_aggregate_before_selection_conserves_each_realization_metal():
    g=np.array([[[0.,0.],[1.,2.]],[[2.,4.],[0.,0.]]])
    labels,t,a=aggregate_joint_realizations(g,[1,3],[7,7])
    assert labels.tolist()==[7] and t.tolist()==[4]
    assert np.allclose(a[:,0],[[.75,1.5],[.5,1.]])
    assert np.allclose(np.einsum('n,rnm->rm',[1,3],g),np.einsum('n,rnm->rm',t,a))
    r=recoverable_moments(a.transpose(1,0,2),[[1,1]],[.6],[1,0],support='aggregated block')
    assert r.probability[0,0]==.5

def test_monotone_and_zero_cutoff_identity():
    rng=np.random.default_rng(234)
    g=rng.random((9,31,3));w=rng.random((9,31))
    r=recoverable_moments(g,w,[0,.1,.5,1,2],[1,.2,.3],support='block')
    assert (np.diff(r.probability,axis=1)<=1e-14).all()
    assert (np.diff(r.truncated_grade,axis=1)<=1e-14).all()
    assert np.allclose(r.truncated_grade[:,0],np.einsum('nk,nkm->nm',w/w.sum(1)[:,None],g))

@pytest.mark.parametrize('kwargs',[{'g':[[[np.nan,0]]]}, {'g':[[[-1,0]]]},
    {'w':[[0,0]]},{'w':[[1,-1]]},{'w':[[1]]},{'c':[.4,.2]}, {'c':[]},
    {'a':[1]}, {'a':[0,0]}, {'c':[.2,.2]}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError): calculate(**kwargs)

def test_weight_scaling_and_support_required():
    assert np.allclose(calculate(w=[[1e308,1e308]]).probability,calculate().probability)
    with pytest.raises(ValueError): recoverable_moments([[[1]]],[[1]],[0],[1],support='')
    with pytest.raises(ValueError): calculate().totals([-1])
    with pytest.raises(ValueError): calculate().totals([1,2])

@pytest.mark.parametrize('t,ids',[([0,1],[1,1]),([1,1],[1.,1.]),([1],[1,1])])
def test_bad_aggregation(t,ids):
    with pytest.raises(ValueError): aggregate_joint_realizations(np.ones((2,2,1)),t,ids)
