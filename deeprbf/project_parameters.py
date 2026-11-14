# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Deposit-neutral grade parameter discovery from audited composite supports."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from deeprbf_recovered import CovarianceSpec, NeighborhoodSpec
from .grade import GradeEstimator
from .grade_benchmark import compare_idw2, paired_grade_scores
from .spatial_validation import buffered_hole_folds
from .variography import directional_fit


@dataclass(frozen=True)
class CompositeSchema:
    """Explicit field mapping; XYZ, measured depth and support must be in metres."""
    hole: str
    domain: str
    xyz: tuple[str,str,str]
    depth_from: str
    depth_to: str
    lower: str
    units: str
    upper: str | None = None


def read_grade_project(data, schema):
    """Read one analyte; reject ambiguous or overlapping supports rather than guessing."""
    fields = [schema.hole,schema.domain,*schema.xyz,schema.depth_from,schema.depth_to,schema.lower]
    if schema.upper is not None:
        fields.append(schema.upper)
    if len(fields)!=len(set(fields)) or not schema.units.strip():
        raise ValueError('Distinct input fields and declared grade units required')
    raw = (pd.read_csv(Path(data),usecols=fields,dtype={schema.hole:'string',schema.domain:'string'})
           if not isinstance(data,pd.DataFrame) else data[fields].copy())
    out = raw[fields].copy()
    out.columns = ['hole','domain','X','Y','Z','from_m','to_m','lower']+(['upper'] if schema.upper is not None else [])
    if schema.upper is None:
        out['upper'] = out.lower
    for k in ('hole','domain'):
        out[k] = out[k].astype('string').str.strip()
    nums = ['X','Y','Z','from_m','to_m','lower','upper']
    out[nums] = out[nums].apply(pd.to_numeric,errors='coerce')
    bad = (~np.isfinite(out[nums]).all(axis=1) | out[['hole','domain']].fillna('').eq('').any(axis=1) |
           out.from_m.lt(0) | out.to_m.le(out.from_m) | out.lower.lt(0) | out.upper.lt(out.lower))
    if bad.any():
        raise ValueError(f'{int(bad.sum())} invalid composite rows; first zero-based positions: {np.flatnonzero(bad)[:10].tolist()}')
    if out.empty:
        raise ValueError('No composites supplied')
    out = out.sort_values(['hole','from_m','to_m'],kind='stable').reset_index(drop=True)
    previous = out.groupby('hole').to_m.transform(lambda g:g.cummax().shift())
    if out.from_m.lt(previous-1e-8).any():
        raise ValueError('Overlapping composite supports within a hole; reconcile before parameter discovery')
    out['support_m'] = out.to_m-out.from_m
    return out


def _weights(frame):
    w = frame.support_m / frame.groupby('hole').support_m.transform('sum')
    return w.to_numpy()/w.sum()


def _geometry(frame,x):
    w = _weights(frame)
    centered = x-w@x
    _, spread, axes = np.linalg.svd(centered*np.sqrt(w[:,None]),full_matrices=False)
    if axes.shape==(3,3):
        for row in axes:
            if row[np.argmax(abs(row))]<0:
                row *= -1
        if np.linalg.det(axes)<0:
            axes[-1] *= -1
    # Nearest independent hole measured from observed supports, not collar spacing.
    nearest = np.full(len(x),np.inf)
    holes = frame.hole.to_numpy()
    unique = np.unique(holes)
    if len(unique)>1:
        # Adaptive k avoids building one near-identical tree per hole.
        tree = cKDTree(x)
        pending = np.arange(len(x))
        k = 2
        while len(pending):
            # Keep k × query size bounded even for an unusually dense single hole.
            unresolved = []
            for chunk in np.array_split(pending, max(1,int(np.ceil(len(pending)*k/500000)))):
                d, ix = tree.query(x[chunk],k=k)
                cross = holes[ix]!=holes[chunk,None]
                found = cross.any(axis=1)
                nearest[chunk[found]] = d[np.arange(len(chunk))[found],cross.argmax(axis=1)[found]]
                unresolved.extend(chunk[~found])
            pending = np.asarray(unresolved,int)
            if k==len(x):
                break
            k = min(k*2,len(x))
    return spread,axes,nearest


def propose_grade_parameters(frame, *, units, geological_axes=None, max_pairs=100000):
    """Return ranked, unvalidated candidates per declared domain; never infer domain labels."""
    if not units.strip() or not isinstance(max_pairs,int) or max_pairs<100:
        raise ValueError('Declare grade units and a valid pair limit')
    override = None if geological_axes is None else np.asarray(geological_axes,float)
    if override is not None and (override.shape!=(3,3) or not np.isfinite(override).all() or
            not np.allclose(override@override.T,np.eye(3)) or np.linalg.det(override)<0):
        raise ValueError('Geological axes must be right-handed orthonormal rows')
    result = dict(status='PARAMETER_DISCOVERY_ONLY',units=units,coordinate_units='m',domains={})
    for key,g in frame.groupby('domain',sort=True):
        x=g[['X','Y','Z']].to_numpy()
        exact=g.lower.eq(g.upper).to_numpy()
        spread,pca,nearest=_geometry(g,x)
        valid=nearest[np.isfinite(nearest)]
        spacing=float(np.median(valid)) if len(valid) else None
        length=float(g.support_m.median())
        distinct=int(g.hole.nunique())
        report=dict(samples=len(g),holes=distinct,exact_samples=int(exact.sum()),
            censored_samples=int((~exact).sum()),median_support_m=length,median_nearest_other_hole_m=spacing,
            geometry_spread_m=spread.tolist(),variograms=[],candidates=[],
            status='INSUFFICIENT_INDEPENDENT_SUPPORT')
        result['domains'][str(key)]=report
        if distinct<3 or len(g)<8 or spacing is None:
            continue
        base=max(spacing,length)
        search=dict(min_neighbors=8,max_neighbors=24,min_holes=3,max_per_hole=3)
        w=_weights(g)
        midpoint=g.lower.to_numpy()/2+g.upper.to_numpy()/2
        report['hole_balanced_midpoint_mean']=float(w@midpoint)
        sill=float(w@(midpoint-w@midpoint)**2)
        if not np.isfinite(sill) or sill<=1e-15:
            report['status']='NO_IDENTIFIABLE_GRADE_VARIANCE'
            continue
        geometries=[('world_axes',np.eye(3))]
        if override is not None:
            geometries.insert(0,('declared_geology',override))
        elif pca.shape==(3,3) and spread[-1]>1e-6 and spread[0]/spread[-1]>1.15:
            geometries.insert(0,('sample_geometry_pca',pca))
        exact_count=int(exact.sum())
        if exact_count>=20 and g.loc[exact,'hole'].nunique()>=4 and np.var(g.lower[exact])>1e-15:
            for origin,axes in [*geometries,('isotropic_between_holes',np.eye(3))]:
                for family in ('exponential','gaussian'):
                    fit=directional_fit(x[exact],g.lower.to_numpy()[exact],axes,family=family,
                        max_pairs=max_pairs,hole_ids=g.hole.to_numpy()[exact],isotropic=origin=='isotropic_between_holes')
                    report['variograms'].append(dict(orientation=origin,family=family,**{k:v for k,v in fit.items() if k!='family'}))
                    if fit['status']!='FIT':
                        continue
                    ranges=np.asarray(fit['ranges'])
                    radii=np.maximum(ranges*1.5,base)
                    report['candidates'].append(dict(id=f'{origin}_{family}',source='BETWEEN_HOLE_VARIANCE_FIT',
                        axes=axes.tolist(),family=family,ranges=ranges.tolist(),search_radii=radii.tolist(),
                        sill=fit['sill'],nugget_fraction=fit['nugget_fraction'],search=search,
                        fit_residual=fit['weighted_residual']))
        # Radial and directional residuals use different experimental bins.
        # Preserve supported directional fits ahead of the radial fallback.
        report['candidates'].sort(key=lambda c:(c['id'].startswith('isotropic'),c['fit_residual'],c['id']))
        report['status']='FITTED_CANDIDATES_REQUIRE_VALIDATION' if report['candidates'] else 'NO_RELIABLE_CONTINUITY_FIT'
        report['candidates'].append(dict(id='spacing_control',source='SPACING_CONTROL_NOT_FITTED',
            axes=np.eye(3).tolist(),family='exponential',ranges=[2*base]*3,search_radii=[3*base]*3,
            sill=sill,nugget_fraction=.50,search=search))
    return result


def estimator_from_parameters(frame, candidate, *, domain, censor_fraction, allow_spacing_control=False):
    """Build the native estimator for a chosen domain; pass integer target domain 1."""
    if not np.isfinite(censor_fraction) or not 0<=censor_fraction<=1:
        raise ValueError('Specify a censoring fraction from zero to one')
    if candidate['source']=='SPACING_CONTROL_NOT_FITTED' and not allow_spacing_control:
        raise ValueError('Spacing control requires explicit diagnostic opt-in')
    g=frame[frame.domain.eq(domain)]
    cov=CovarianceSpec()
    cov.family=candidate['family'];cov.range_convention='practical';cov.axes=candidate['axes']
    cov.ranges=candidate['ranges'];cov.sill=candidate['sill'];cov.nugget=cov.sill*candidate['nugget_fraction']
    search=NeighborhoodSpec()
    search.axes=candidate['axes'];search.radii=candidate['search_radii'];search.restrict_domain=True
    for name,value in candidate['search'].items():
        setattr(search,name,value)
    y=g.lower+censor_fraction*(g.upper-g.lower)
    return GradeEstimator(g[['X','Y','Z']],y,g.hole,np.ones(len(g),np.int32),
                          covariance=cov,search=search)


def validate_grade_parameters(frame, *, units, gates, buffer_m, censor_fraction,
                              geological_axes=None, max_pairs=100000, n_folds=3,
                              allow_spacing_control=False):
    """Refit discovery inside each spatial training fold; score untouched exact targets."""
    reports={}
    for domain,g in frame.groupby('domain',sort=True):
        rows=[]
        report=dict(folds=rows,status='INSUFFICIENT_HOLES')
        reports[str(domain)]=report
        if g.hole.nunique()<max(n_folds,4):
            continue
        xyz=g[['X','Y','Z']].to_numpy()
        try:
            folds=list(buffered_hole_folds(xyz,g.hole,n_folds=n_folds,buffer_m=buffer_m))
        except ValueError as e:
            if str(e)!='Buffer removed every training sample':
                raise
            report['status']='BUFFER_REMOVED_ALL_TRAINING_IN_A_FOLD'
            continue
        for fold,tr,te in folds:
            a,b=g.iloc[tr],g.iloc[te]
            nearest=float(cKDTree(xyz[tr]).query(xyz[te])[0].min())
            if nearest<buffer_m-1e-8 or not set(a.hole).isdisjoint(b.hole):
                raise RuntimeError('Fold violates the holdout contract')
            proposal=propose_grade_parameters(a,units=units,geological_axes=geological_axes,max_pairs=max_pairs)
            detail=proposal['domains'][str(domain)]
            exact=b[b.lower.eq(b.upper)]
            row=dict(fold=fold,train_holes=int(a.hole.nunique()),test_holes=int(b.hole.nunique()),
                     exact_test_samples=len(exact),nearest_training_m=nearest,discovery_status=detail['status'],
                     candidate=None,status='NO_ELIGIBLE_CANDIDATE')
            rows.append(row)
            options=[c for c in detail['candidates'] if allow_spacing_control or c['source']!='SPACING_CONTROL_NOT_FITTED']
            if not options or exact.empty:
                if exact.empty:
                    row['status']='NO_EXACT_HOLDOUT_GRADES'
                continue
            chosen=options[0]
            model=estimator_from_parameters(a,chosen,domain=domain,censor_fraction=censor_fraction,
                                            allow_spacing_control=allow_spacing_control)
            pred, ref=compare_idw2(model,exact[['X','Y','Z']],np.ones(len(exact),np.int32))
            row.update(candidate=chosen,status='SCORED',**paired_grade_scores(
                exact.lower,pred['estimate'],ref,_weights(exact),float(_weights(a)@model.grades),gates))
        report['status']='ALL_FOLDS_PASS' if all(
            r.get('deeprbf',{}).get('pass_all',False) and r['candidate']['source']!='SPACING_CONTROL_NOT_FITTED'
            for r in rows) else 'NOT_VALIDATED'
    return reports


def main():
    import argparse
    import hashlib
    import json
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('csv',type=Path)
    cli.add_argument('config',type=Path,help='JSON with schema, and optional geological_axes and validation')
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    with args.csv.open('rb') as source:
        before=hashlib.file_digest(source,'sha256').hexdigest()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    schema=CompositeSchema(**cfg['schema'])
    frame=read_grade_project(args.csv,schema)
    result=propose_grade_parameters(frame,units=schema.units,geological_axes=cfg.get('geological_axes'))
    if 'validation' in cfg:
        result['validation']=validate_grade_parameters(frame,units=schema.units,
            geological_axes=cfg.get('geological_axes'),**cfg['validation'])
    with args.csv.open('rb') as source:
        if before!=hashlib.file_digest(source,'sha256').hexdigest():
            raise RuntimeError('Input changed during parameter discovery')
    result['source_sha256']=before
    result['config']=cfg
    if args.output.resolve() in (args.csv.resolve(),args.config.resolve()):
        raise ValueError('Report path must differ from input files')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+'.tmp')
    if tmp.resolve() in (args.csv.resolve(),args.config.resolve()):
        raise ValueError('Temporary report path conflicts with inputs')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(args.output)


if __name__=='__main__':
    main()
