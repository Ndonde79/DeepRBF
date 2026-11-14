# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Controlled continuous-grade validation and gated block CSV export."""
from dataclasses import dataclass,asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .model_run import freeze_run
from .client_rules import release_gate


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class GradeRule:
    target: str
    unit: str
    population: str
    minimum_holes: int
    minimum_support_fraction: float
    minimum_mae_skill: float
    maximum_relative_bias: float
    buffer_m: float
    estimator_sha256: str

    def __post_init__(self):
        if not all(isinstance(s,str) and s.strip() for s in (self.target,self.unit,self.population)):
            raise ValueError('Target, unit and population required')
        if type(self.minimum_holes) is not int or self.minimum_holes<2:
            raise ValueError('At least two contributing holes required')
        if not (0<self.minimum_support_fraction<=1 and 0<=self.minimum_mae_skill<1 and
                math.isfinite(self.maximum_relative_bias) and self.maximum_relative_bias>=0 and
                math.isfinite(self.buffer_m) and self.buffer_m>0):
            raise ValueError('Invalid frozen limits')
        if len(self.estimator_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.estimator_sha256):
            raise ValueError('Estimator settings hash required')

    def digest(self):
        return _hash(asdict(self))


def estimator_digest(config):
    return _hash(config)


def _checked_result(raw,n,minimum_holes):
    grades=np.asarray(raw['estimate'],float)
    holes=np.asarray(raw['n_holes'])
    flags=np.asarray(raw['supported'])
    if any(a.shape!=(n,) for a in (grades,holes,flags)) or flags.dtype!=bool or not np.issubdtype(holes.dtype,np.integer) or (holes<0).any():
        raise ValueError('Invalid estimator outputs')
    good=flags & np.isfinite(grades) & (grades>=0) & (holes>=minimum_holes)
    return np.where(good,grades,np.nan),holes,good


def validate_grade(rule, observations, *, predictor, config, ledger_path, n_folds=3):
    """Validate exact assays only; the caller must account for censored rows upstream."""
    if estimator_digest(config)!=rule.estimator_sha256:
        raise ValueError('Estimator settings differ from frozen rule')
    keys=['source_row','hole_id','X','Y','Z','domain','support_m','grade']
    d=observations[keys].reset_index(drop=True).copy()
    x=d[['X','Y','Z']].to_numpy(float)
    g=d.grade.to_numpy(float)
    w=d.support_m.to_numpy(float)
    if not np.isfinite(g).all() or (g<0).any() or not np.isfinite(w).all() or (w<=0).any() or d.domain.isna().any():
        raise ValueError('Exact audited grades, domains and positive support required')
    payload=hashlib.sha256(d.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()).hexdigest()
    ledger=freeze_run(rule,d.source_row,x,d.hole_id,payload_sha256=payload,ledger_path=ledger_path,n_folds=n_folds)
    idx={str(a):i for i,a in enumerate(d.source_row)}
    pred=np.full(len(d),np.nan)
    base=pred.copy()
    dist=pred.copy()
    nh=np.zeros(len(d),int)
    fold_id=np.zeros(len(d),int)
    for f in ledger['folds']:
        tr=np.array([idx[k] for k in f['training']]); te=np.array([idx[k] for k in f['testing']])
        train=d.iloc[tr].copy()
        target=d.iloc[te][['X','Y','Z','domain']].copy()
        pred[te],nh[te],_=_checked_result(predictor(train,target),len(te),rule.minimum_holes)
        for domain,t in target.groupby('domain').groups.items():
            r=train[train.domain.eq(domain)]
            if len(r):
                base[np.asarray(t,int)]=np.average(r.grade,weights=r.support_m)
        dist[te]=cKDTree(x[tr]).query(x[te])[0]
        fold_id[te]=f['fold']
    paired=np.isfinite(pred)&np.isfinite(base)
    support=float(w[paired].sum()/w.sum())
    mae=baseline=skill=bias=None
    if paired.any():
        mae=float(np.average(abs(pred[paired]-g[paired]),weights=w[paired]))
        baseline=float(np.average(abs(base[paired]-g[paired]),weights=w[paired]))
        skill=1-mae/baseline if baseline>0 else None
        mean=float(np.average(g[paired],weights=w[paired]))
        bias=float(np.average(pred[paired]-g[paired],weights=w[paired])/mean) if mean>0 else None
    blockers=[]
    if support<rule.minimum_support_fraction: blockers.append('LOW_PAIRED_SUPPORT')
    if skill is None or skill<rule.minimum_mae_skill: blockers.append('BASELINE_SKILL_NOT_MET')
    if bias is None or abs(bias)>rule.maximum_relative_bias: blockers.append('BIAS_LIMIT_NOT_MET')
    report=dict(rule_sha256=rule.digest(),ledger_sha256=_hash(ledger),status='PASS' if not blockers else 'HOLD',
                blockers=blockers,paired_support_fraction=support,mae=mae,baseline_mae=baseline,mae_skill=skill,
                relative_bias=bias)
    d=d.assign(fold=fold_id,estimate=pred,baseline=base,n_holes=nh,nearest_training_m=dist,scored=paired)
    return d,report


def export_grade_blocks(rule, targets, raw, *, output, validation, release_evidence=None, release=False):
    """Review exports mask unsupported grades; release exports require all gates."""
    evidence=dict(release_evidence or {})
    passed=validation.get('status')=='PASS' and validation.get('rule_sha256')==rule.digest()
    evidence['held_out_geometry_and_grade_pass']=passed and evidence.get('held_out_geometry_and_grade_pass') is True
    gate=release_gate(evidence)
    if release and gate['status']!='RELEASE_ELIGIBLE':
        raise ValueError('Release blocked: '+', '.join(gate['unmet_rules']))
    keys=['block_id','X','Y','Z','DX','DY','DZ','domain']
    d=targets[keys].copy()
    geom=d[['X','Y','Z','DX','DY','DZ']].to_numpy(float)
    if (d.empty or d.block_id.isna().any() or d.block_id.duplicated().any() or
        d.duplicated(['X','Y','Z']).any() or d.domain.isna().any() or
        not np.isfinite(geom).all() or (geom[:,3:]<=0).any()):
        raise ValueError('Invalid block identifiers, geometry or domains')
    d['grade'],d['n_holes'],ok=_checked_result(raw,len(d),rule.minimum_holes)
    d['grade_status']=np.where(ok,'ESTIMATED','UNSUPPORTED')
    d['grade_target']=rule.target
    d['grade_unit']=rule.unit
    d['resource_class']='UNCLASSIFIED'
    d['export_mode']='GATED_GRADE_EXPORT' if release else 'REVIEW_ONLY'
    with Path(output).open('x',encoding='utf8',newline='') as stream:
        d.to_csv(stream,index=False)
    return {'blocks':len(d),'supported':int(ok.sum()),'release_gate':gate,'mode':d.export_mode.iloc[0]}
