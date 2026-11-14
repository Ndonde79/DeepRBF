# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
from deeprbf.lens_correlations import propose_correlations

def test_incompatible_connections_not_created():
    x=np.array([[0.,0,0],[20.,0,0],[30.,0,0],[40.,0,0]])
    e=propose_correlations(x,['A','A','B','C'],['S']*4,['F','F','OTHER','F'],['P','P','P','Q'],['COMPLETE_LOGGED_CROSSING']*4,['R']*4,['R']*4)
    assert e==[]

def test_competing_intersections_remain_ranked():
    x=np.array([[0.,0,0],[20.,0,0],[25.,0,0]])
    e=propose_correlations(x,['A','B','B'],['S']*3,['F']*3,['P']*3,['COMPLETE_LOGGED_CROSSING']*3,['R']*3,['R']*3,np.tile([0.,1,0],(3,1)))
    assert len(e)==2 and sorted(q['rank_a'] for q in e)==[1,2]
    assert e[0]['best_alternative_margin_a']>0
