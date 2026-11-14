# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import pandas as pd
import pytest
from deeprbf.model_rules import IndicatorRule,evaluate_indicator


def example():
    rule=IndicatorRule('Au','declared host','assay Au','g/t',(0.5,),2,.5,.1,25.)
    data=pd.DataFrame(dict(source_row=[1,2,3],hole_id=['a','b','c'],fold=[0,1,1],
         threshold=[.5]*3,observed_indicator=[0.,1.,float('nan')],
         probability_exceedance=[.1,.9,.7],training_prior=[.5]*3,
         threshold_direct_support=[True,False,True],nearest_training_m=[25.,40.,30.]))
    return rule,data


def test_baseline_and_censoring_scored_on_identical_rows():
    r,d=example()
    got=evaluate_indicator(r,d,frozen_rule_sha256=r.digest(),verified_training_ledger=True,
                           verified_units=True,verified_domains=True)
    assert got['status']=='INDICATOR_EVIDENCE_PASS'
    t=got['thresholds'][0]
    assert t['ambiguous']==1 and t['direct_support_fraction']==.5
    assert t['brier_skill']==pytest.approx(.96)


def test_numeric_pass_cannot_override_missing_provenance_and_buffer():
    r,d=example()
    d.loc[0,'nearest_training_m']=24.
    got=evaluate_indicator(r,d)
    assert got['status']=='HOLD'
    assert 'SPATIAL_BUFFER_BREACH' in got['blockers']
    assert 'RULE_FREEZE_NOT_VERIFIED' in got['blockers']
    assert 'TRAINING_LEDGER_NOT_VERIFIED' in got['blockers']


def test_reject_hole_fold_leakage_and_duplicate_rows():
    r,d=example()
    duplicate=pd.concat([d,d.iloc[:1]])
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate_indicator(r,duplicate)
    row=d.iloc[:1].assign(source_row=99,fold=1)
    with pytest.raises(ValueError,match='multiple test folds'):
        evaluate_indicator(r,pd.concat([d,row]))


def test_threshold_failure_not_averaged_away():
    r,d=example()
    d['threshold_direct_support']=False
    got=evaluate_indicator(r,d)
    assert 'LOW_DIRECT_SUPPORT' in got['thresholds'][0]['failed_rules']
    assert 'TARGET_THRESHOLDS_FAILED' in got['blockers']
