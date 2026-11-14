# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
import pytest
from deeprbf.model_rules import IndicatorRule
from deeprbf.model_run import freeze_run,run_indicator


def test_frozen_folds_are_buffered_and_reject_changed_inputs(tmp_path):
    rule=IndicatorRule('Au','host','Au','g/t',(.5,),2,.5,.0,20.)
    x=np.array([[0,0,0],[0,0,1],[100,0,0],[100,0,1],[200,0,0],[200,0,1.]])
    ids=np.arange(len(x))
    holes=np.repeat(['A','B','C'],2)
    kw=dict(payload_sha256='a'*64,ledger_path=tmp_path/'run.json')
    ledger=freeze_run(rule,ids,x,holes,**kw)
    assert ledger==freeze_run(rule,ids,x,holes,**kw)
    for f in ledger['folds']:
        tr,te=([int(i) for i in f[k]] for k in ['training','testing'])
        assert not set(holes[tr])&set(holes[te])
        assert f['min_test_to_training_m']>=20
    with pytest.raises(ValueError,match='Existing ledger differs'):
        freeze_run(rule,ids,x,holes,**dict(kw,payload_sha256='b'*64))


def test_duplicate_ids_rejected_before_ledger_is_written(tmp_path):
    rule=IndicatorRule('Cu','vein','Cu','pct',(.5,),2,.5,.0,20.)
    path=tmp_path/'run.json'
    with pytest.raises(ValueError,match='Duplicate'):
        freeze_run(rule,[1,1],[[0,0,0],[100,0,0]],['A','B'],payload_sha256='a'*64,ledger_path=path)
    assert not path.exists()


def test_runner_blinds_assays_and_preserves_indeterminate_labels(tmp_path):
    rule=IndicatorRule('Au','host','Au','g/t',(.5,),2,.5,0.,20.)
    d=pd.DataFrame(dict(source_row=[1,2,3],hole_id=['A','B','C'],
         X=[0.,100.,200.],Y=[0.]*3,Z=[0.]*3,domain=['host']*3,support_m=[1.]*3,
         lower=[0.,.2,1.],upper=[0.,.7,1.]))
    def estimator(train,targets,cut):
        assert len(train)==2 and cut==.5
        assert set(targets)=={'X','Y','Z','domain'}
        assert not set(train.X)&set(targets.X)
        return [0.5],[True],[0.5]
    result,_=run_indicator(rule,d,predictor=estimator,ledger_path=tmp_path/'fit.json')
    y=result.sort_values('source_row').observed_indicator.to_numpy()
    assert y[0]==0 and np.isnan(y[1]) and y[2]==1
