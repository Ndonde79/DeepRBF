# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Whole-hole spatial folds, exclusion buffers and category-balanced scoring."""
import numpy as np
from scipy.spatial import cKDTree


def buffered_hole_folds(xyz,hole_ids,*,n_folds=3,buffer_m=25.):
    xyz=np.asarray(xyz,float);holes=np.asarray(hole_ids).astype(str)
    if xyz.shape!=(len(holes),3) or not np.isfinite(xyz).all():raise ValueError('Finite sample coordinates required')
    if not isinstance(n_folds,int) or n_folds<2 or not np.isfinite(buffer_m) or buffer_m<0:raise ValueError('Invalid fold/buffer controls')
    unique=np.unique(holes)
    if len(unique)<n_folds:raise ValueError('Too few holes for spatial folds')
    centers=np.array([xyz[holes==hole].mean(0) for hole in unique])
    _,_,axes=np.linalg.svd(centers-centers.mean(0),full_matrices=False)
    axis=axes[0];axis*=1 if axis[np.abs(axis).argmax()]>=0 else -1
    order=np.lexsort((unique,centers@axis))
    for fold,ids in enumerate(np.array_split(order,n_folds)):
        test=np.flatnonzero(np.isin(holes,unique[ids]));train=np.flatnonzero(~np.isin(holes,unique[ids]))
        distance=cKDTree(xyz[test]).query(xyz[train])[0]
        train=train[distance>=buffer_m]
        if not len(train):raise ValueError('Buffer removed every training sample')
        yield fold,train,test


def categorical_scores(probabilities,labels,categories,*,weights=None,supported=None):
    p=np.asarray(probabilities,float);categories=np.asarray(categories);labels=np.asarray(labels)
    if p.shape!=(len(labels),len(categories)) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-9):raise ValueError('Invalid probability matrix')
    if not np.isin(labels,categories).all():raise ValueError('Scoring labels are outside the declared catalogue')
    weights=np.ones(len(labels)) if weights is None else np.asarray(weights,float)
    if weights.shape!=(len(labels),) or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Positive scoring weights required')
    truth=labels[:,None]==categories;prediction=categories[p.argmax(1)]
    rows=[]
    for category in categories:
        mask=labels==category
        rows.append(dict(category=int(category),test_samples=int(mask.sum()),
            recall=float(np.average(prediction[mask]==category,weights=weights[mask])) if mask.any() else None))
    result=dict(samples=len(labels),accuracy=float(np.average(prediction==labels,weights=weights)),
        multiclass_brier=float(np.average(np.sum((p-truth)**2,axis=1),weights=weights)),
        log_loss=float(np.average(-np.log(np.maximum(np.sum(p*truth,axis=1),1e-12)),weights=weights)),
        macro_recall=float(np.mean([row['recall'] for row in rows if row['recall'] is not None])),per_category=rows)
    if supported is not None:
        supported=np.asarray(supported,bool)
        if supported.shape!=(len(labels),):raise ValueError('Support mask shape mismatch')
        result['supported_samples']=int(supported.sum())
        result['supported_fraction']=float(np.average(supported,weights=weights))
        for name,mask in [('supported',supported),('unsupported',~supported)]:
            result[name]=categorical_scores(p[mask],labels[mask],categories,weights=weights[mask]) if mask.any() else None
    return result
