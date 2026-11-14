# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
from deeprbf.intersection_semantics import classify_intersections

def row(a,b,code,known=True,form='A'):return dict(start=a,end=b,code=code,known=known,formation=form)
def classify(rows):return classify_intersections(rows,{'MS':'S','SM':'S','SG':'SG'})[0]

def test_merge_and_complete_crossing():
    a=classify([row(0,2,'RH'),row(2,4,'MS'),row(4,6,'SM'),row(6,8,'RH')])
    assert len(a)==1 and a[0]['status']=='COMPLETE_LOGGED_CROSSING' and a[0]['end']==6

def test_missing_is_not_negative_evidence():
    a=classify([row(0,2,'',False),row(2,4,'MS'),row(5,8,'RH')])[0]
    assert a['status']=='NO_OBSERVED_FAMILY_BOUNDARY' and a['start_evidence']=='UNKNOWN_LOG' and a['end_evidence']=='LOGGING_GAP'

def test_hole_end_is_partial():
    a=classify([row(0,2,'RH'),row(2,4,'MS')])[0]
    assert a['status']=='PARTIAL_ONE_CONTACT' and a['end_evidence']=='UNOBSERVED'

def test_formation_change_is_not_family_termination():
    a=classify([row(0,2,'RH'),row(2,4,'MS'),row(4,6,'MS',form='B'),row(6,8,'RH')])
    assert len(a)==2 and all(q['status']=='PARTIAL_ONE_CONTACT' for q in a)
    assert a[0]['end_evidence']=='FAMILY_CONTINUES'

def test_nested_overlap_is_ambiguous():
    a,issues=classify_intersections([row(0,10,'RH'),row(2,3,'MS'),row(4,5,'MS')],{'MS':'S'})
    assert len(issues)==3 and all(q['status']=='AMBIGUOUS_OVERLAP' for q in a)

def test_mixed_holes_rejected():
    try:classify_intersections([dict(hole='A',**row(0,1,'MS')),dict(hole='B',**row(1,2,'MS'))],{'MS':'S'})
    except ValueError:return
    raise AssertionError('Mixed holes must not merge')

def test_explicit_unknown_not_selected():
    assert classify([row(0,1,'MS',known=False)])==[]
