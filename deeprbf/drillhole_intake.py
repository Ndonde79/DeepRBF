# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Mapped drillhole intake and boundary-respecting supports for grade discovery."""
import numpy as np
import pandas as pd
from .compositing import partition_interval
from .desurvey import minimum_curvature


def prepare_grade_supports(collar, survey, assay, geology, *, dip_convention,
                           single_station_straight=False, support_m=1.):
    """Standard columns: collar hole/x/y/z/depth; survey hole/depth/dip/azimuth;
    assay hole/from_m/to_m/lower/upper; geology hole/from_m/to_m/domain.
    Return supports and issue ledger without mutating source tables.
    """
    if dip_convention not in ('negative_down','positive_down') or not np.isfinite(support_m) or support_m<=0:
        raise ValueError('Declare dip convention and positive support length')
    issues=[]
    def flag(table,i,h,reason):
        issues.append(dict(table=table,source_row=int(i)+2,hole=h,reason=reason))
    def read(frame, name, fields):
        d=frame.reset_index(drop=True)[['hole',*fields]].copy()
        d.hole=d.hole.astype('string').str.strip()
        d[fields]=d[fields].apply(pd.to_numeric,errors='coerce')
        invalid=~np.isfinite(d[fields]).all(axis=1)|d.hole.fillna('').eq('')
        for i in d.index[invalid]:
            flag(name,i,d.at[i,'hole'],'INVALID_REQUIRED_VALUE')
        return d[~invalid].copy()
    c=read(collar,'collar',['x','y','z','depth'])
    invalid=c.hole.duplicated(keep=False)|c.depth.le(0)
    for i in c.index[invalid]:
        flag('collar',i,c.at[i,'hole'],'AMBIGUOUS_OR_INVALID_COLLAR')
    c=c[~invalid].set_index('hole')
    s=read(survey,'survey',['depth','dip','azimuth'])
    invalid=~s.hole.isin(c.index)|s.depth.lt(0)|s.depth.gt(s.hole.map(c.depth)+1e-6)|s.dip.abs().gt(90)|~s.azimuth.between(0,360)
    for i in s.index[invalid]:
        flag('survey',i,s.at[i,'hole'],'INVALID_STATION')
    invalid_survey_holes=set(s.loc[invalid,'hole'])
    invalid_survey_holes.update(row['hole'] for row in issues if row['table']=='survey' and pd.notna(row['hole']))
    s=s[~invalid].drop_duplicates()
    bad_stations=set(s.loc[s.duplicated(['hole','depth'],keep=False),'hole'])
    station={h:g for h,g in s.groupby('hole') if h not in bad_stations|invalid_survey_holes}

    def intervals(raw, name, fields):
        d=read(raw,name,['from_m','to_m',*fields])
        if name=='geology':
            d['domain']=raw.reset_index(drop=True).domain.astype('string').str.strip()
            invalid=d.domain.fillna('').eq('')
        else:
            invalid=d.lower.lt(0)|d.upper.lt(d.lower)
        for i in d.index[invalid]:
            flag(name,i,d.at[i,'hole'],'INVALID_VALUE_OR_DOMAIN')
        d=d[~invalid].copy()
        duplicate=d.duplicated()
        for i in d.index[duplicate]:
            flag(name,i,d.at[i,'hole'],'EXACT_DUPLICATE')
        d=d[~duplicate]
        bad=~d.hole.isin(c.index)|d.from_m.lt(0)|d.to_m.le(d.from_m)|d.to_m.gt(d.hole.map(c.depth)+1e-6)
        for i in d.index[bad]:
            h=d.at[i,'hole']
            flag(name,i,h,'MISSING_COLLAR' if h not in c.index else
                 'INVALID_INTERVAL' if d.at[i,'from_m']<0 or d.at[i,'to_m']<=d.at[i,'from_m'] else 'BEYOND_COLLAR')
        d=d[~bad].sort_values(['hole','from_m','to_m'])
        conflicts=set()
        for _,g in d.groupby('hole'):
            starts,ends=g.from_m.to_numpy(),g.to_m.to_numpy()
            indices=g.index.to_numpy()
            for j,end in enumerate(ends):
                k=np.searchsorted(starts,end-1e-8)
                if k>j+1:
                    conflicts.update(indices[j:k])
        for i in sorted(conflicts):
            flag(name,i,d.at[i,'hole'],'OVERLAPPING_INTERVAL')
        return d[~d.index.isin(conflicts)]

    a=intervals(assay,'assay',['lower','upper'])
    g=intervals(geology,'geology',[])
    logs={h:r for h,r in g.groupby('hole')}
    output=[]
    for h,samples in a.groupby('hole'):
        st=station.get(h)
        if st is None or (len(st)==1 and not single_station_straight):
            for i in samples.index:
                flag('assay',i,h,'NO_USABLE_TRAJECTORY')
            continue
        geo=logs.get(h,g.iloc[:0])
        pending=[]
        for r in samples.itertuples():
            if len(st)>1 and (r.from_m<st.depth.min()-1e-6 or r.to_m>st.depth.max()+1e-6):
                flag('assay',r.Index,h,'OUTSIDE_MEASURED_SURVEY');continue
            q=geo[geo.from_m.lt(r.to_m)&geo.to_m.gt(r.from_m)]
            length=0.
            for j in q.itertuples():
                left,right=max(r.from_m,j.from_m),min(r.to_m,j.to_m)
                if right-left<=1e-8:
                    continue
                length+=right-left
                for x,y in partition_interval(left,right,support_m):
                    pending.append((h,str(j.domain),x,y,r.lower,r.upper,r.Index+2,j.Index+2,
                                    'DECLARED_STRAIGHT_SINGLE_STATION' if len(st)==1 else 'WITHIN_MEASURED_STATIONS'))
            if length<r.to_m-r.from_m-1e-6:
                flag('assay',r.Index,h,'PARTIAL_OR_MISSING_GEOLOGY')
        if pending:
            co=c.loc[h]
            xyz=minimum_curvature(co[['x','y','z']],st.depth,st.dip*(-1 if dip_convention=='negative_down' else 1),
                                  st.azimuth,np.array([(r[2]+r[3])/2 for r in pending]))
            output.extend((*row,*p) for row,p in zip(pending,xyz))
    return (pd.DataFrame(output,columns=['hole','domain','from_m','to_m','lower','upper','assay_source_row',
                'geology_source_row','survey_status','X','Y','Z']),
            pd.DataFrame(issues,columns=['table','source_row','hole','reason']))
