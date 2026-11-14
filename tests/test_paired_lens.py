# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
from deeprbf.paired_lens import PairedLens

def test_exact_contacts_and_membership():
    x,z=np.meshgrid([-20.,0,20],[-20.,0,20]);c=np.c_[x.ravel(),.2*x.ravel()+.1*z.ravel(),z.ravel()];n=np.array([-.2,1.,-.1]);n/=np.linalg.norm(n)
    m=PairedLens().fit(c-2*n,c+2*n,np.arange(len(c)).astype(str))
    assert m.max_contact_error<1e-8
    r=m.evaluate(np.vstack([np.zeros(3),5*n,[1000.,1000,1000]]))
    assert list(r['member'])==[True,False,False] and not r['supported'][-1]

def test_partial_endpoint_not_imposed():
    x,z=np.meshgrid([-20.,0,20],[-20.,0,20]);c=np.c_[x.ravel(),np.zeros(9),z.ravel()];a=c-[0,2,0];b=c+[0,2,0]
    b[-1,1]=100;flags=np.ones(9,bool);flags[-1]=False
    m=PairedLens().fit(a,b,np.arange(9).astype(str),end_observed=flags)
    assert m.max_contact_error<1e-8 and max(abs(p[:,2]).max() for p in m.constraints)<3

def test_dense_hole_does_not_hide_other_supporting_holes():
    x,z=np.meshgrid(np.arange(-3,4)*.1,np.arange(-3,4)*.1)
    c=np.vstack([np.c_[x.ravel(),np.zeros(49),z.ravel()],[20,0,0],[0,0,20]])
    holes=['dense']*49+['second','third']
    m=PairedLens().fit(c-[0,2,0],c+[0,2,0],holes)
    assert m.evaluate(np.array([[0.,0,0]]))['supported'][0]
