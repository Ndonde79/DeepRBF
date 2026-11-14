# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Explicit paired section traces lofted in normalized section-height coordinates."""
import numpy as np

class SectionEnvelope:
    def __init__(self, sections):
        self.sections=sorted(sections,key=lambda s:s['x'])
        self.x=np.array([s['x'] for s in self.sections],float)
        if len(self.x)<2 or not np.isfinite(self.x).all() or (np.diff(self.x)<=0).any():raise ValueError('At least two distinct ordered sections required')
        for s in self.sections:
            z=np.asarray(s['z'],float);a=np.asarray(s['lower_y'],float);b=np.asarray(s['upper_y'],float)
            if z.ndim!=1 or len(z)<2 or a.shape!=z.shape or b.shape!=z.shape or not np.isfinite(np.r_[z,a,b]).all() or (np.diff(z)<=0).any() or (b<=a).any():raise ValueError('Ordered, noncrossing paired section controls required')
            s.update(z=z,lower_y=a,upper_y=b)

    def bounds(self,x,z):
        x=np.asarray(x,float);z=np.asarray(z,float)
        if x.shape!=z.shape or not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Matching finite coordinates required')
        k=np.clip(np.searchsorted(self.x,x,side='right')-1,0,len(self.x)-2);lower=np.full(x.shape,np.nan);upper=lower.copy();support=np.zeros(x.shape,bool)
        for i in np.unique(k):
            take=k==i;a,b=self.sections[i:i+2];t=(x[take]-a['x'])/(b['x']-a['x']);lo=(1-t)*a['z'][0]+t*b['z'][0];hi=(1-t)*a['z'][-1]+t*b['z'][-1];v=(z[take]-lo)/(hi-lo)
            za=a['z'][0]+v*(a['z'][-1]-a['z'][0]);zb=b['z'][0]+v*(b['z'][-1]-b['z'][0])
            lower[take]=(1-t)*np.interp(za,a['z'],a['lower_y'])+t*np.interp(zb,b['z'],b['lower_y'])
            upper[take]=(1-t)*np.interp(za,a['z'],a['upper_y'])+t*np.interp(zb,b['z'],b['upper_y'])
            support[take]=(t>=0)&(t<=1)&(v>=0)&(v<=1)
        return lower,upper,support

    def classify(self,xyz):
        """Codes: 3 between the paired traces, 2 below the lower trace, 1 above the upper trace, 0 unsupported."""
        xyz=np.asarray(xyz,float)
        if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('XYZ required')
        lo,hi,support=self.bounds(xyz[:,0],xyz[:,2]);code=np.where(xyz[:,1]<lo,2,np.where(xyz[:,1]>hi,1,3));code[~support]=0
        return code,support
