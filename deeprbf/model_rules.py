# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Model evidence gates for client-defined indicator targets."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorRule:
    target: str
    geological_population: str
    assay_channel: str
    unit: str
    thresholds: tuple
    minimum_holes: int
    minimum_support_fraction: float
    minimum_brier_skill: float
    buffer_m: float

    def __post_init__(self):
        if not all(str(s).strip() for s in (self.target,self.geological_population,self.assay_channel,self.unit)):
            raise ValueError('Explicit target, geological population, channel and unit required')
        if not self.thresholds or not all(math.isfinite(v) and v>=0 for v in self.thresholds):
            raise ValueError('Finite nonnegative thresholds required')
        if tuple(sorted(set(self.thresholds)))!=tuple(self.thresholds):
            raise ValueError('Thresholds must be strictly increasing')
        if type(self.minimum_holes) is not int or self.minimum_holes<2:
            raise ValueError('Minimum hole count must be at least two')
        if not 0<self.minimum_support_fraction<=1 or not 0<=self.minimum_brier_skill<1:
            raise ValueError('Invalid acceptance thresholds')
        if not math.isfinite(self.buffer_m) or self.buffer_m<=0:
            raise ValueError('A positive spatial exclusion buffer is required')

    def digest(self):
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def evaluate_indicator(rule, rows, *, frozen_rule_sha256=None, verified_training_ledger=False,
                       verified_units=False, verified_domains=False):
    """Evaluate held-out evidence; a numeric pass alone is never release."""
    required={'source_row','hole_id','fold','threshold','observed_indicator','probability_exceedance',
              'training_prior','threshold_direct_support','nearest_training_m'}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError('Incomplete held-out evidence')
    if rows[['source_row','hole_id','fold','threshold']].isna().any().any():
        raise ValueError('Null held-out identifiers')
    if rows.duplicated(['hole_id','source_row','threshold']).any():
        raise ValueError('Duplicate held-out observations')
    if rows.groupby('hole_id').fold.nunique().gt(1).any():
        raise ValueError('A hole appears in multiple test folds')
    support=rows.threshold_direct_support
    if not pd.api.types.is_bool_dtype(support) or support.isna().any():
        raise ValueError('Direct support must be an explicit Boolean')
    y=rows.observed_indicator.to_numpy(float)
    if not np.isin(y[~np.isnan(y)],[0.,1.]).all():
        raise ValueError('Observed indicator must be 0, 1 or unknown')
    for k in ['probability_exceedance','training_prior']:
        values=rows[k].to_numpy(float)
        if not (np.isfinite(values)&(values>=0)&(values<=1)).all():
            raise ValueError('Invalid probabilities')
    t=rows.threshold.to_numpy(float)
    if not np.isfinite(t).all():
        raise ValueError('Invalid observed threshold')
    blockers=[]
    if frozen_rule_sha256!=rule.digest():
        blockers.append('RULE_FREEZE_NOT_VERIFIED')
    if verified_training_ledger is not True:
        blockers.append('TRAINING_LEDGER_NOT_VERIFIED')
    distance=rows.nearest_training_m.to_numpy(float)
    if (np.isfinite(distance)&(distance<rule.buffer_m-1e-7)).any():
        blockers.append('SPATIAL_BUFFER_BREACH')
    if (~np.isfinite(distance)).any():
        blockers.append('SPATIAL_DISTANCE_MISSING')
    if verified_units is not True:
        blockers.append('ASSAY_UNITS_NOT_VERIFIED')
    if verified_domains is not True:
        blockers.append('GEOLOGICAL_DOMAINS_NOT_VERIFIED')
    output=[]
    for threshold in rule.thresholds:
        match=np.isclose(t,threshold,atol=1e-10,rtol=1e-9)
        scored=rows[match&np.isfinite(y)]
        hits=0; fraction=None; skill=None; brier=None; baseline=None
        fail=[]
        if len(scored):
            hits=int(scored.hole_id.nunique())
            fraction=float(scored.threshold_direct_support.mean())
            actual=scored.observed_indicator.to_numpy(float)
            brier=float(np.mean((scored.probability_exceedance-actual)**2))
            baseline=float(np.mean((scored.training_prior-actual)**2))
            skill=1-brier/baseline if baseline>0 else None
        if hits<rule.minimum_holes: fail.append('TOO_FEW_HOLES')
        if fraction is None or fraction<rule.minimum_support_fraction: fail.append('LOW_DIRECT_SUPPORT')
        if skill is None or skill<rule.minimum_brier_skill: fail.append('BASELINE_SKILL_NOT_MET')
        output.append(dict(threshold=float(threshold),scored=len(scored),holes=hits,
                           ambiguous=int(match.sum()-len(scored)),direct_support_fraction=fraction,
                           brier=brier,baseline_brier=baseline,brier_skill=skill,failed_rules=fail))
    if any(r['failed_rules'] for r in output):
        blockers.append('TARGET_THRESHOLDS_FAILED')
    return dict(target=rule.target,rule_sha256=rule.digest(),
                status='INDICATOR_EVIDENCE_PASS' if not blockers else 'HOLD',
                blockers=blockers,thresholds=output)
