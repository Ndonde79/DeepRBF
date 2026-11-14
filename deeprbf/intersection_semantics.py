# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Evidence-preserving classification of logged indicator intersections.

Family membership is caller supplied. A contrary lithology is NOT a barren assay.
Missing logs and gaps never establish a terminating contact.
"""
import math

def classify_intersections(records, family_map, tolerance=0.01):
    """Classify one hole's records; reject explicitly mixed hole identifiers."""
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid depth tolerance')
    records=list(records)
    holes={str(r['hole']).strip() for r in records if 'hole' in r}
    if len(holes)>1:raise ValueError('Classify one hole at a time')
    rows=[];issues=[]
    for source,r in enumerate(records):
        q=dict(r);q['source_id']=q.get('source_id',source)
        try:a,b=float(q['start']),float(q['end'])
        except (ValueError,TypeError,KeyError):issues.append(dict(source_id=q['source_id'],issue='INVALID_DEPTH'));continue
        if not math.isfinite(a+b) or a<0 or b<=a:issues.append(dict(source_id=q['source_id'],issue='INVALID_DEPTH'));continue
        q.update(start=a,end=b,family=family_map.get(q.get('code')) if q.get('known',True) else None,overlap=False)
        rows.append(q)
    rows.sort(key=lambda q:(q['start'],q['end']))
    active=[]
    for i,q in enumerate(rows):
        active=[j for j in active if rows[j]['end']>q['start']+tolerance]
        if active:
            q['overlap']=True
            for j in active:rows[j]['overlap']=True
        active.append(i)
    for q in rows:
        if q['overlap']:issues.append(dict(source_id=q['source_id'],issue='OVERLAPPING_LOG'))
    out=[];i=0
    while i<len(rows):
        first=rows[i]
        if first['family'] is None:i+=1;continue
        j=i
        while j+1<len(rows):
            a,b=rows[j],rows[j+1]
            if a['overlap'] or b['overlap'] or abs(a['end']-b['start'])>tolerance or a['family']!=b['family'] or a.get('formation')!=b.get('formation'):break
            j+=1
        last=rows[j]
        def flank(adjacent,at_start):
            current=first if at_start else last
            if current['overlap']:return 'AMBIGUOUS_OVERLAP',None
            if adjacent is None:return 'UNOBSERVED',None
            if adjacent['overlap']:return 'AMBIGUOUS_OVERLAP',adjacent.get('code')
            gap=(current['start']-adjacent['end']) if at_start else (adjacent['start']-current['end'])
            if abs(gap)>tolerance:return 'LOGGING_GAP',adjacent.get('code')
            if not adjacent.get('known',True):return 'UNKNOWN_LOG',adjacent.get('code')
            if adjacent['family']==first['family']:return 'FAMILY_CONTINUES',adjacent.get('code')
            return 'OBSERVED_DIFFERENT_LOG',adjacent.get('code')
        left,lc=flank(rows[i-1] if i else None,True);right,rc=flank(rows[j+1] if j+1<len(rows) else None,False)
        n=(left=='OBSERVED_DIFFERENT_LOG')+(right=='OBSERVED_DIFFERENT_LOG')
        status='AMBIGUOUS_OVERLAP' if 'AMBIGUOUS_OVERLAP' in [left,right] else 'COMPLETE_LOGGED_CROSSING' if n==2 else 'PARTIAL_ONE_CONTACT' if n==1 else 'NO_OBSERVED_FAMILY_BOUNDARY'
        out.append(dict(start=first['start'],end=last['end'],family=first['family'],formation=first.get('formation'),status=status,start_evidence=left,end_evidence=right,before_code=lc,after_code=rc,source_ids=[r['source_id'] for r in rows[i:j+1]]))
        i=j+1
    return out,issues
