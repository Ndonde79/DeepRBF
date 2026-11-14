# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pytest
from deeprbf.contact_rbf import ContactRBF, BalancedContactRBF

def test_contact_and_sign_constraints():
    x=np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[1,0,1],[1,0,2]],float)
    y=np.array([-1,0,1,-1,0,1.])
    m=ContactRBF([1,1,1],smoothing=0).fit(x,y)
    assert np.allclose(m.predict(x),y,atol=1e-10)
    assert np.allclose(m.predict(x,chunk_size=1),m.predict(x))

def test_translation_and_constant_field():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    a=ContactRBF([2,3,4]).fit(x,np.ones(4))
    b=ContactRBF([2,3,4]).fit(x+1e6,np.ones(4))
    assert np.allclose(a.predict(x+.2),1)
    assert np.allclose(a.predict(x+.2),b.predict(x+.2+1e6))

def test_reject_conflicting_locations():
    with pytest.raises(ValueError):ContactRBF([1,1,1]).fit(np.zeros((4,3)),np.arange(4))
    with pytest.raises(ValueError):ContactRBF([1,0,1])
    with pytest.raises(ValueError):ContactRBF([1,1,1]).predict([[0,0,0]])

def test_signed_support_diagnostics():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    m=ContactRBF([1,1,1]).fit(x,[-1,0,1,-1])
    d=m.neighborhood_diagnostics([[0,1,0]])
    assert d['positive_constraints'].tolist()==[1]
    assert d['negative_constraints'].tolist()==[2]
    assert d['contact_constraints'].tolist()==[1]
    assert d['nearest_positive_distance'].tolist()==[0]
    m.fit(x,[-1,-1,-1,-1])
    assert np.isinf(m.neighborhood_diagnostics([[0,0,0]])['nearest_positive_distance'][0])

def test_balanced_matches_scipy_when_limits_inactive():
    rng=np.random.default_rng(9);x=rng.normal(size=(20,3));v=rng.normal(size=20);q=rng.normal(size=(7,3))
    a=ContactRBF([2,3,4],neighbors=12).fit(x,v)
    b=BalancedContactRBF([2,3,4],neighbors=12,max_per_hole=20,min_holes=1,radius=100).fit(x,v,np.arange(20))
    assert np.allclose(a.predict(q),b.predict(q),atol=1e-10)
    assert np.allclose(b.predict(q,chunk_size=1),b.predict(q))

def test_balanced_caps_holes_and_preserves_contacts():
    x=np.array([[j*.01,h,0] for h in range(3) for j in range(10)])
    ids=np.repeat(np.arange(3),10);values=np.sin(x[:,0])+x[:,1]
    m=BalancedContactRBF([1,1,1],max_per_hole=2,neighbors=8,candidate_neighbors=30).fit(x,values,ids)
    result,d=m.predict_with_diagnostics([[.035,.01,0]])
    assert d['constraints'][0]==6 and d['support_holes'][0]==3 and d['supported'][0]
    assert np.allclose(m.predict(x),values,atol=1e-12)

def test_balanced_unsupported_never_silently_predicts_host():
    x=np.array([[j,0,0] for j in range(5)],float)
    m=BalancedContactRBF([1,1,1]).fit(x,np.ones(5),['a']*5)
    p,d=m.predict_with_diagnostics([[.5,0,0],[0,0,0],[100,0,0]])
    assert np.isnan(p[0]) and np.isnan(p[2]) and p[1]==1
    assert d['supported'].tolist()==[False,True,False]
    assert d['exact_conditioned'].tolist()==[False,True,False]
