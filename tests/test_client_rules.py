# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
from deeprbf.client_rules import AssayChannel,audit_channel,audit_intervals,release_gate


def test_unverified_units_preserve_raw_without_numeric_grade():
    p,r=audit_channel(['10','< 5','>2',None,'-999','foo'],
        AssayChannel('Cu','Cu ppm','ppm','HEADER_ONLY','total'))
    assert p.kind.tolist()==['EXACT','LEFT_CENSORED','RIGHT_CENSORED','MISSING','INVALID','INVALID']
    assert p.grade_ppm.isna().all()
    assert p.limit_in_source_unit.iloc[1:3].tolist()==[5,2]
    assert set(r['blockers'])=={'UNIT_NOT_VERIFIED','INVALID_ASSAY_TOKENS','CENSORING_POLICY_REQUIRED'}


def test_explicit_conversion_never_replaces_censored_limit():
    p,r=audit_channel(['1e-2','<=.01','0'],AssayChannel(
        'Cu','column','pct','LAB_OR_DATA_DICTIONARY_VERIFIED','total',.2))
    assert p.grade_ppm.iloc[[0,2]].tolist()==[100.,0.]
    assert np.isnan(p.grade_ppm.iloc[1])
    assert r['blockers']==['CENSORING_EXCEEDS_APPROVED_LIMIT']


def test_nested_overlap_and_duplicate_detection():
    d=pd.DataFrame({'h':['a','a','a','a','b','b',''],'a':[0,1,3,3,0,2,2],'b':[10,2,4,4,2,4,3]})
    r=audit_intervals(d,hole='h',start='a',end='b')
    assert (r['invalid'],r['duplicate_rows'],r['later_intervals_overlapping_previous'])==(1,2,2)


def test_release_requires_independent_evidence():
    r=release_gate({})
    assert r['status']=='HOLD'
    evidence=dict.fromkeys(r['unmet_rules'],True)
    assert release_gate(evidence)['status']=='RELEASE_ELIGIBLE'
    evidence['resource_review_signed']='yes'
    assert release_gate(evidence)['unmet_rules']==['resource_review_signed']
