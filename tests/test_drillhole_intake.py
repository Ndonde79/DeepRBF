# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
from deeprbf.drillhole_intake import prepare_grade_supports


def test_intake_retains_hole_splits_contacts_and_flags_bad_source_rows():
    c=pd.DataFrame([('A',100,200,50,4)],columns=['hole','x','y','z','depth'])
    s=pd.DataFrame([('A',4,-90,0)],columns=['hole','depth','dip','azimuth'])
    a=pd.DataFrame([('A',0,2,5,5),('A',0,2,5,5),('A',2,5,9,9)],
                   columns=['hole','from_m','to_m','lower','upper'])
    g=pd.DataFrame([('A',0,1.5,'L'),('A',1.5,4,'M')],columns=['hole','from_m','to_m','domain'])
    r,issues=prepare_grade_supports(c,s,a,g,dip_convention='negative_down',single_station_straight=True)
    assert r.domain.tolist()==['L','L','M']
    assert set(issues.reason)=={'EXACT_DUPLICATE','BEYOND_COLLAR'}
    assert r.assay_source_row.unique().tolist()==[2]
    np.testing.assert_allclose(r.Z,[49.5,48.75,48.25])
    r,issues=prepare_grade_supports(c,s,a,g,dip_convention='negative_down')
    assert r.empty and 'NO_USABLE_TRAJECTORY' in set(issues.reason)
