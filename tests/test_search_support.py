# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
from deeprbf.search_support import search_support


def test_hole_quota_and_candidate_budget_diagnostics():
    x=np.c_[[1,2,3,4,6,8],np.zeros((6,2))]
    q=np.zeros((1,3))
    kwargs=dict(min_holes=3,min_samples=4,max_per_hole=2)
    support,clipped=search_support(x,['A','A','A','B','C','D'],q,[4.5,6.5,10.],candidates=4,**kwargs)
    assert support.tolist()==[[False,False,False]]
    assert clipped.tolist()==[True]
    support,clipped=search_support(x,['A','A','A','B','C','D'],q,[4.5,6.5,10.],candidates=8,**kwargs)
    assert support.tolist()==[[False,True,True]]
    assert clipped.tolist()==[False]
