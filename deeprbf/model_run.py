# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Deterministic pre-fit fold ledger for the modelling runner."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .spatial_validation import buffered_hole_folds


def _digest(ids, xyz, holes, payload_sha256):
    h=hashlib.sha256()
    h.update(json.dumps([ids.tolist(),holes.tolist(),payload_sha256],
                       ensure_ascii=False,separators=(',',':')).encode())
    h.update(np.asarray(xyz,dtype='<f8',order='C').tobytes())
    return h.hexdigest()


def _inputs(source_ids, xyz, hole_ids, payload_sha256):
    ids=np.asarray(source_ids)
    x=np.asarray(xyz,dtype=float)
    holes=np.asarray(hole_ids)
    if ids.ndim!=1 or holes.shape!=ids.shape or x.shape!=(len(ids),3) or not len(ids):
        raise ValueError('One unique row ID, coordinate and drillhole per observation')
    if not np.isfinite(x).all() or any(not isinstance(h,str) or not h.strip() for h in holes):
        raise ValueError('Finite coordinates and nonempty drillhole IDs required')
    if any(not isinstance(i,(str,int,np.integer)) or isinstance(i,(bool,np.bool_)) for i in ids):
        raise ValueError('Row IDs must be strings or integers')
    ids=ids.astype(str)
    if len(np.unique(ids))!=len(ids) or any(not i.strip() for i in ids):
        raise ValueError('Duplicate or empty row ID')
    if not isinstance(payload_sha256,str) or len(payload_sha256)!=64 or any(c not in '0123456789abcdef' for c in payload_sha256):
        raise ValueError('SHA-256 of the actual modelling input payload required')
    return ids,x,holes.astype(str)


def freeze_run(rule, source_ids, xyz, hole_ids, *, payload_sha256, ledger_path, n_folds=3):
    """Create or verify an immutable split file before exposing training rows."""
    ids,x,h=_inputs(source_ids,xyz,hole_ids,payload_sha256)
    splits=[]
    for i,tr,te in buffered_hole_folds(x,h,n_folds=n_folds,buffer_m=rule.buffer_m):
        if np.intersect1d(h[tr],h[te]).size:
            raise RuntimeError('Drillhole leaked between training and test')
        distances=cKDTree(x[tr]).query(x[te])[0]
        if (distances<rule.buffer_m-1e-7).any():
            raise RuntimeError('Spatial buffer breach')
        splits.append(dict(fold=i,training=ids[tr].tolist(),testing=ids[te].tolist(),
                           min_test_to_training_m=float(distances.min())))
    if sorted(i for fold in splits for i in fold['testing'])!=sorted(ids.tolist()):
        raise RuntimeError('Each input observation must occur exactly once as a test')
    ledger=dict(schema=1,rule_sha256=rule.digest(),input_sha256=_digest(ids,x,h,payload_sha256),
                payload_sha256=payload_sha256,buffer_m=rule.buffer_m,folds=splits)
    path=Path(ledger_path)
    body=json.dumps(ledger,indent=2,allow_nan=False)
    try:
        with path.open('x',encoding='utf8') as f:
            f.write(body)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf8'))!=ledger:
            raise ValueError('Existing ledger differs; start a new run for changed rules or inputs')
    return ledger


def run_indicator(rule, observations, *, predictor, ledger_path, n_folds=3):
    """Run a development estimator with the frozen splits and blinded test rows."""
    keys=['source_row','hole_id','X','Y','Z','domain','support_m','lower','upper']
    if not set(keys).issubset(observations.columns):
        raise ValueError('Canonical interval observations required')
    data=observations[keys].reset_index(drop=True).copy()
    bounds=data[['lower','upper']].to_numpy(float)
    if (np.isnan(bounds).any() or (bounds[:,0]<0).any() or (bounds[:,1]<bounds[:,0]).any()
            or not np.isfinite(bounds[:,0]).all() or data.domain.isna().any()
            or not np.isfinite(data.support_m).all() or not data.support_m.gt(0).all()):
        raise ValueError('Invalid interval bounds, support or geological domain')
    payload=hashlib.sha256(data.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()).hexdigest()
    xyz=data[['X','Y','Z']].to_numpy(float)
    ledger=freeze_run(rule,data.source_row,xyz,data.hole_id,payload_sha256=payload,
                      ledger_path=ledger_path,n_folds=n_folds)
    key={str(i):k for k,i in enumerate(data.source_row)}
    result=[]
    for fold in ledger['folds']:
        ti=[key[i] for i in fold['training']]
        vi=[key[i] for i in fold['testing']]
        tr=data.iloc[ti].reset_index(drop=True)
        # Do not expose held-out assay bounds, support length, row IDs or hole IDs.
        targets=data.iloc[vi][['X','Y','Z','domain']].reset_index(drop=True)
        z=data.iloc[vi]
        dist=cKDTree(xyz[ti]).query(xyz[vi])[0]
        for cut in rule.thresholds:
            pred,ok,base=predictor(tr.copy(),targets.copy(),cut)
            if any(np.asarray(v).shape!=(len(z),) for v in [pred,ok,base]) or np.asarray(ok).dtype!=bool:
                raise ValueError('Predictor must return one probability, Boolean support and prior per row')
            y=np.select([z.lower.gt(cut),z.upper.le(cut)],[1.,0.],default=np.nan)
            result.append(pd.DataFrame(dict(source_row=z.source_row.to_numpy(),hole_id=z.hole_id.to_numpy(),
                fold=fold['fold'],threshold=cut,observed_indicator=y,probability_exceedance=pred,
                training_prior=base,threshold_direct_support=ok,nearest_training_m=dist)))
    return pd.concat(result,ignore_index=True),ledger
