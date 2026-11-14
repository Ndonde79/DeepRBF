# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Conceptual mineral-system geometry, not deposit classification or grade prediction.

Arrays use local (across, down-dip, strike) indices. Codes describe geological
roles; a site-specific mapping is required before conditioning real lithology.
Dimensions and geometry ratios below are explicit scenario assumptions.
"""
import numpy as np

GEOLOGICAL_PROFILES={
    'vms':dict(roles={0:'country_rock',1:'prospective_horizon',2:'massive_sulphide_lens',3:'footwall_stringer',4:'alteration_halo'},
        source='https://www.usgs.gov/publications/volcanogenic-massive-sulfide-occurrence-model'),
    'structural_cu_ag_au':dict(roles={0:'country_rock',1:'shear_domain',2:'mineralized_shoot_scenario',3:'vein_breccia_splays',4:'alteration_halo'},
        source='https://pubs.usgs.gov/bul/b1693/html/bull7kdv.htm'),
    'orogenic_gold':dict(roles={0:'country_rock',1:'shear_domain',2:'oreshoots_scenario',3:'quartz_vein_splays',4:'alteration_halo'},
        source='https://pubs.usgs.gov/publication/dr1198/full'),
    'sedex':dict(roles={0:'country_rock',1:'prospective_sedimentary_interval',2:'stratiform_sulphide_scenario',3:'fault_feeder_scenario',4:'optional_barite_horizon'},
        source='https://pubs.usgs.gov/bul/b1693/html/bull5nmq.htm')}


def conceptual_training_image(profile,shape=(40,40,48),seed=42):
    """Generate one complete, seeded conceptual geometry for sensitivity studies.

    No evidence that an ore body exists is created by this function. Local
    geometry ratios are illustrative; the returned array carries no CRS.
    """
    if profile not in GEOLOGICAL_PROFILES:raise ValueError('Unknown geological profile')
    if len(shape)!=3 or any(not isinstance(v,(int,np.integer)) or v<8 for v in shape):raise ValueError('Three dimensions >=8 required')
    rng=np.random.default_rng(seed);z,y,x=np.indices(shape,dtype=float)
    x=(x+.5)/shape[2];y=(y+.5)/shape[1];z=(z+.5)/shape[0]
    image=np.zeros(shape,dtype=np.int64)
    if profile=='vms':
        horizon=z-.54-.035*np.sin(2*np.pi*x)
        image[abs(horizon)<.13]=1
        lens=((x-.5)/.38)**2+((y-.5)/.30)**2+(horizon/.065)**2
        alteration=((x-.5)/.43)**2+((y-.5)/.34)**2+((horizon+.13)/.25)**2
        image[(alteration<1)&(horizon<0)]=4
        for offset in rng.uniform(-.20,.20,8):
            vein=abs(x-.5-offset*(.6-z)-.025*np.sin(15*z+offset))<.012
            image[vein&(z<.54)&(z>.16)&(abs(y-.5)<.18)]=3
        image[lens<1]=2
    elif profile=='sedex':
        bedding=z+.065*np.sin(2*np.pi*x)+.035*np.sin(2*np.pi*y)-.08*(x>.56)
        image[(bedding>.32)&(bedding<.70)]=1
        image[(abs(bedding-.65)<.018)&(x<.83)]=4
        for level,width in ((.43,.014),(.52,.023),(.59,.012)):
            ore=(abs(bedding-level)<width)&(((x-.5)/.43)**2+((y-.5)/.45)**2<1)
            image[ore]=2
        image[(abs(x-.56)<.013)&(z<.52)&(z>.12)&(abs(y-.5)<.16)]=3
    else:
        shear=x-.50-.055*np.sin(3*np.pi*z)
        image[abs(shear)<.14]=4;image[abs(shear)<.075]=1
        angles=(.24,-.20) if profile=='structural_cu_ag_au' else (.34,-.28,.06)
        for slope in angles:
            splays=abs(shear-slope*(z-.5))<.012
            image[splays&(abs(y-.5)<.30)]=3
        shoot=(abs(shear)<.027)&(abs(y-.5-.08*np.sin(8*z))<.13)&((abs(z-.34)<.09)|(abs(z-.70)<.10))
        image[shoot]=2
    return image
