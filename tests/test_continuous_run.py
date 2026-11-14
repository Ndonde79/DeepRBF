# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
import pytest
from deeprbf.continuous_run import GradeRule,estimator_digest,validate_grade,export_grade_blocks


def rule(config):
    return GradeRule('Au','g/t','test host',2,.5,0.,.1,25.,estimator_digest(config))


def test_frozen_grade_run_blinds_assays_and_uses_paired_baseline(tmp_path):
    cfg={'method':'test'}
    r=rule(cfg)
    x=np.array([0.,100.,200.])
    d=pd.DataFrame(dict(source_row=[1,2,3],hole_id=['a','b','c'],X=x,Y=0.,Z=0.,
                        domain='host',support_m=1.,grade=x/100+1))
    def predict(train,target):
        assert set(target)=={'X','Y','Z','domain'}
        assert len(train)==2
        return dict(estimate=target.X.to_numpy()/100+1,supported=np.array([True]),n_holes=np.array([2]))
    rows,v=validate_grade(r,d,predictor=predict,config=cfg,ledger_path=tmp_path/'ledger.json')
    assert v['status']=='PASS' and v['mae_skill']==1 and v['relative_bias']==0
    assert rows.nearest_training_m.min()>=25
    with pytest.raises(ValueError,match='settings'):
        validate_grade(r,d,predictor=predict,config={'method':'different'},ledger_path=tmp_path/'ledger.json')


def test_export_keeps_unsupported_blank_and_refuses_unapproved_release(tmp_path):
    r=rule({'method':'test'})
    blocks=pd.DataFrame(dict(block_id=[1,2],X=[0,5],Y=0,Z=0,DX=5,DY=5,DZ=5,domain='host'))
    raw=dict(estimate=[3.,9.],supported=[True,True],n_holes=[2,1])
    qa=dict(status='PASS',rule_sha256=r.digest())
    path=tmp_path/'blocks.csv'
    export_grade_blocks(r,blocks,raw,output=path,validation=qa)
    d=pd.read_csv(path)
    assert d.grade.iloc[0]==3 and np.isnan(d.grade.iloc[1])
    assert list(d.grade_status)==['ESTIMATED','UNSUPPORTED']
    denied=tmp_path/'release.csv'
    with pytest.raises(ValueError,match='Release blocked'):
        export_grade_blocks(r,blocks,raw,output=denied,validation=qa,release=True)
    assert not denied.exists()
