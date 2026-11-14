# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
import pytest

from deeprbf.project_parameters import (CompositeSchema, read_grade_project, propose_grade_parameters,
    estimator_from_parameters, validate_grade_parameters)

SCHEMA = CompositeSchema('BH','ROCK',('E','N','RL'),'START','END','AU','g/t')


def project(scale=1,holes=9):
    rows=[]
    for h in range(holes):
        for k in range(6):
            rows.append((f'H{h}', 'D1',scale*(h%3)*45,scale*(h//3)*45,-k*2,k*2,k*2+2,1+h/9+k/12))
    return pd.DataFrame(rows,columns=['BH','ROCK','E','N','RL','START','END','AU'])


def test_deposit_geometry_changes_spacing_without_name_presets():
    candidates=[]
    for scale in (1,3):
        frame=read_grade_project(project(scale),SCHEMA)
        r=propose_grade_parameters(frame,units='g/t')['domains']['D1']
        candidates.append(r['candidates'][-1])
        assert r['median_nearest_other_hole_m']==45*scale
        assert all(row['hole_pairs']>=3 for fit in r['variograms'] for row in fit['experimental'])
    assert candidates[1]['search_radii'][0]==3*candidates[0]['search_radii'][0]
    with pytest.raises(ValueError,match='opt-in'):
        estimator_from_parameters(frame,candidates[-1],domain='D1',censor_fraction=.5)
    est=estimator_from_parameters(frame,candidates[-1],domain='D1',censor_fraction=.5,allow_spacing_control=True)
    assert est.predict([[100,100,-4]],target_domains=[1])['supported'][0]


def test_duplicate_support_is_rejected_and_one_hole_is_not_fitted():
    f=project(holes=1)
    with pytest.raises(ValueError,match='Overlapping'):
        read_grade_project(pd.concat([f,f.iloc[:1]]),SCHEMA)
    result=propose_grade_parameters(read_grade_project(f,SCHEMA),units='g/t')['domains']['D1']
    assert result['status']=='INSUFFICIENT_INDEPENDENT_SUPPORT' and not result['candidates']


def test_spacing_diagnostic_is_never_promoted_even_with_permissive_gates():
    frame=read_grade_project(project(),SCHEMA)
    r=validate_grade_parameters(frame,units='g/t',gates=dict(coverage=0,scored=1,abs_relative_bias=10,rmse_ratio=100),
        buffer_m=25,censor_fraction=.5,allow_spacing_control=True)['D1']
    assert len(r['folds'])==3 and all(f['nearest_training_m']>=25 for f in r['folds'])
    assert all(f['candidate']['source']=='SPACING_CONTROL_NOT_FITTED' for f in r['folds'])
    assert r['status']=='NOT_VALIDATED'
