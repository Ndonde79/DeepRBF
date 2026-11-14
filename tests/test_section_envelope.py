# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
from deeprbf.section_envelope import SectionEnvelope

def test_sections_reproduced_and_bounds_honoured():
    m=SectionEnvelope([dict(x=0,z=[0,10],lower_y=[0,1],upper_y=[2,3]),dict(x=10,z=[2,12],lower_y=[2,3],upper_y=[4,5])])
    lo,hi,s=m.bounds(np.array([0.,10,5]),np.array([10.,2,6]));assert np.allclose(lo,[1,2,1.5]) and np.allclose(hi,[3,4,3.5]) and s.all()
    code,s=m.classify(np.array([[5,2,6],[5,-1,6],[5,10,6],[11,2,6]]));assert list(code)==[3,2,1,0]

def test_crossing_controls_rejected():
    try:SectionEnvelope([dict(x=0,z=[0,1],lower_y=[0,2],upper_y=[1,1]),dict(x=1,z=[0,1],lower_y=[0,0],upper_y=[1,1])])
    except ValueError:return
    raise AssertionError('Crossing controls accepted')
