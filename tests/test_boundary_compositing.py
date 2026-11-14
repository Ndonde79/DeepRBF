# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pytest
from deeprbf.compositing import interval_average,partition_interval


def test_boundary_partition_preserves_length_and_contacts():
    pieces=partition_interval(0,7.3,2,[1.7,5.1])
    assert sum(b-a for a,b in pieces)==pytest.approx(7.3)
    assert max(b-a for a,b in pieces)<=2+1e-9
    assert all(not a<c<b for a,b in pieces for c in (1.7,5.1))


def test_length_metal_conservation_and_missing_coverage():
    a=[0,1,3];b=[1,3,5];v=[2,4,8]
    total=0
    for left,right in partition_interval(0,5,2,[2.3]):
        grade,coverage,overlap=interval_average(a,b,v,left,right)
        assert not overlap and coverage==pytest.approx(right-left)
        total+=grade*coverage
    assert total==pytest.approx(26)
    grade,coverage,overlap=interval_average([0,2],[1,3],[0,np.nan],0,3)
    assert grade==0 and coverage==1 and not overlap


def test_overlap_cannot_hide_behind_a_gap():
    grade,coverage,overlap=interval_average([0,0,2],[1,1,3],[1,1,1],0,3)
    assert coverage==3 and overlap and np.isnan(grade)
