# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Deposit-independent intake and release rules. No unit inference or assay substitution."""
from dataclasses import dataclass
import math
import re
import numpy as np
import pandas as pd

_MASS = {'pct': 10000., 'ppm': 1., 'ppb': .001, 'g/t': 1.}
_ASSAY = re.compile(r'^([<>]=?)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)$')


@dataclass(frozen=True)
class AssayChannel:
    element: str
    column: str
    unit: str | None
    unit_evidence: str
    fraction: str
    max_censored_fraction: float | None = None

    def __post_init__(self):
        if not self.element.strip() or not self.column.strip() or not self.fraction.strip():
            raise ValueError('Element, column and analytical fraction required')
        if self.unit is not None and self.unit not in _MASS:
            raise ValueError('Unsupported mass concentration unit')
        if self.unit_evidence not in {'UNRESOLVED','HEADER_ONLY','LAB_OR_DATA_DICTIONARY_VERIFIED'}:
            raise ValueError('Unknown unit evidence class')
        if self.unit_evidence != 'UNRESOLVED' and self.unit is None:
            raise ValueError('A declared unit is required for this evidence class')
        c=self.max_censored_fraction
        if c is not None and (not math.isfinite(c) or not 0<=c<=1):
            raise ValueError('Invalid censoring limit')


def audit_channel(values, spec):
    s=pd.Series(values,dtype='string').str.strip()
    empty=s.isna()|s.eq('').fillna(False)
    m=s.str.extract(_ASSAY)
    qualifier=m[0].fillna('')
    val=pd.to_numeric(m[1],errors='coerce').astype(float)
    finite=np.isfinite(val)
    observed=(~empty)&finite&val.ge(0)&qualifier.eq('')
    left=(~empty)&finite&val.gt(0)&qualifier.str.startswith('<')
    right=(~empty)&finite&val.gt(0)&qualifier.str.startswith('>')
    invalid=~(empty|observed|left|right)
    n=int((observed|left|right).sum())
    cf=float((left|right).sum()/n) if n else None
    blockers=[]
    if spec.unit_evidence!='LAB_OR_DATA_DICTIONARY_VERIFIED':
        blockers.append('UNIT_NOT_VERIFIED')
    if invalid.any():
        blockers.append('INVALID_ASSAY_TOKENS')
    if not observed.any():
        blockers.append('NO_EXACT_ASSAYS')
    if (left|right).any():
        if spec.max_censored_fraction is None:
            blockers.append('CENSORING_POLICY_REQUIRED')
        elif cf>spec.max_censored_fraction:
            blockers.append('CENSORING_EXCEEDS_APPROVED_LIMIT')
    trusted=spec.unit_evidence=='LAB_OR_DATA_DICTIONARY_VERIFIED'
    # Even accepted censoring limits never make a detection limit a grade.
    output=pd.DataFrame({'raw':s,'kind':np.select(
        [x.to_numpy(dtype=bool) for x in (observed,left,right,empty)],
        ['EXACT','LEFT_CENSORED','RIGHT_CENSORED','MISSING'],default='INVALID'),
        'grade_ppm':val.where(observed)*_MASS[spec.unit] if trusted else np.nan,
        'limit_in_source_unit':val.where(left|right)})
    report=dict(element=spec.element,column=spec.column,fraction=spec.fraction,
                declared_unit=spec.unit,unit_evidence=spec.unit_evidence,
                counts={k:int(v) for k,v in output.kind.value_counts().items()},
                censor_fraction_nonmissing_valid=cf,blockers=blockers,
                channel_ready_for_review=not blockers)
    return output,report


def audit_intervals(data, *, hole, start, end):
    ids=data[hole].astype('string').str.strip()
    lo=pd.to_numeric(data[start],errors='coerce').to_numpy(float)
    hi=pd.to_numeric(data[end],errors='coerce').to_numpy(float)
    valid=np.isfinite(lo)&np.isfinite(hi)&(lo>=0)&(hi>lo)&ids.notna().to_numpy(dtype=bool)&ids.ne('').fillna(False).to_numpy(dtype=bool)
    v=pd.DataFrame({'hole':ids.to_numpy()[valid],'from':lo[valid],'to':hi[valid]})
    duplicate=v.duplicated(['hole','from','to'],keep=False)
    distinct=v.drop_duplicates().sort_values(['hole','from','to'])
    previous=distinct.groupby('hole')['to'].transform(lambda a:a.cummax().shift())
    overlap=int((distinct['from']<previous-1e-7).sum())
    invalid=int((~valid).sum())
    blockers=[name for count,name in [(invalid,'INVALID_INTERVALS'),(int(duplicate.sum()),'DUPLICATE_INTERVALS'),
                                      (overlap,'OVERLAPPING_INTERVALS')] if count]
    return dict(rows=len(data),holes=int(v.hole.nunique()),invalid=invalid,duplicate_rows=int(duplicate.sum()),
                later_intervals_overlapping_previous=overlap,blockers=blockers)


def release_gate(evidence):
    """Default-deny release; rule IDs describe evidence, not deposit choices."""
    required=('verified_units_and_assay_qaqc','reconciled_collar_survey_intervals',
              'explicit_geological_domains_and_contacts','held_out_geometry_and_grade_pass',
              'block_support_and_density_verified','resource_review_signed')
    missing=[k for k in required if evidence.get(k) is not True]
    return {'status':'RELEASE_ELIGIBLE' if not missing else 'HOLD','unmet_rules':missing}
