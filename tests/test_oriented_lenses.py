# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
from deeprbf.oriented_lenses import OrientedLens

def test_tilted_paired_contacts_and_rejection():
    x,z=np.meshgrid(np.arange(-40,41,20),np.arange(-40,41,20));centers=np.c_[x.ravel(),.3*x.ravel()+.2*z.ravel(),z.ravel()]
    n=np.array([-.3,1.,-.2]);n/=np.linalg.norm(n)
    m=OrientedLens().fit(centers-3*n,centers+3*n,np.arange(len(centers)).astype(str))
    q=m.evaluate(np.array([[0.,0,0],*(n[None,:]*[3,6][:1])]))
    assert np.allclose(q['halfwidth'],3) and np.allclose(abs(q['signed_distance']),[0,3],atol=1e-8)
    assert not m.evaluate(np.array([[1000.,1000,1000]]))['supported'][0]

def test_single_hole_does_not_define_plane():
    centers=np.c_[np.arange(10)*5,np.zeros(10),np.zeros(10)]
    m=OrientedLens().fit(centers-[0,2,0],centers+[0,2,0],['one']*10)
    assert not m.evaluate(np.array([[20.,0,0]]))['supported'][0]

def test_collinear_many_holes_rejected():
    centers=np.c_[np.arange(10)*5,np.zeros(10),np.zeros(10)]
    m=OrientedLens().fit(centers-[0,2,0],centers+[0,2,0],np.arange(10).astype(str))
    assert not np.isfinite(m.normals).any()


def test_dense_hole_does_not_hide_supporting_holes():
    """2026-11-13: 300 intervals of one hole at the target used to fill the neighbour list and hide four holes 40-60 m away."""
    st, en, h = [], [], []
    for k in range(300):
        z = -k * 0.2; st.append([0, 0, z]); en.append([0, 0, z - 2]); h.append('D')
    for name, (x, y) in {'A': (40, 0), 'B': (0, 45), 'C': (-50, -30), 'E': (30, -50)}.items():
        for z in (0, -20, -40):
            st.append([x, y, z]); en.append([x, y, z - 2]); h.append(name)
    lens = OrientedLens(ranges=(150, 150, 100), neighbors=24, max_per_hole=2, min_holes=3).fit(np.array(st, float), np.array(en, float), h)
    r = lens.evaluate(np.array([[5., 5., -20.]]))
    assert r['support_holes'][0] >= 3 and r['supported'][0]
