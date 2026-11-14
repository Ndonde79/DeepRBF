# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Paired geological contacts represented in a data-derived local frame.

This interpolates observed mid-surfaces and log thickness, with explicit
footprint and distance limits. It does not assign resource confidence.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree, Delaunay
from scipy.optimize import least_squares


class PairedContactSurface:
    def __init__(self, *, smoothing=.1, neighbors=24, max_distance=150., edge_buffer=25.):
        if not np.isfinite(smoothing) or smoothing<0 or not isinstance(neighbors,int) or neighbors<3:
            raise ValueError('Nonnegative smoothing and at least three neighbours required')
        if not np.isfinite(max_distance) or max_distance<=0 or not np.isfinite(edge_buffer) or edge_buffer<0:
            raise ValueError('Positive search distance and nonnegative edge buffer required')
        self.smoothing=smoothing;self.neighbors=neighbors
        self.max_distance=max_distance;self.edge_buffer=edge_buffer

    def fit(self, xy, midpoint, thickness):
        xy=np.asarray(xy,float);midpoint=np.asarray(midpoint,float);thickness=np.asarray(thickness,float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<4 or len(midpoint)!=len(xy) or len(thickness)!=len(xy):
            raise ValueError('At least four two-dimensional contact locations required')
        if not np.isfinite(xy).all() or not np.isfinite(midpoint).all() or not np.isfinite(thickness).all() or (thickness<=0).any():
            raise ValueError('Contacts and positive thickness must be finite')
        if len(np.unique(xy,axis=0))!=len(xy):raise ValueError('Duplicate contact locations must be reconciled')
        self.center=xy.mean(0);self.xy=xy.copy();self.tree=cKDTree(xy);self.hull=Delaunay(xy)
        scale=100.;self.scale=scale
        values=np.c_[midpoint,np.log(thickness)]
        self.model=RBFInterpolator((xy-self.center)/scale,values,kernel='thin_plate_spline',
            smoothing=self.smoothing,neighbors=min(self.neighbors,len(xy)),degree=1)
        self.thickness_bounds=(float(thickness.min()),float(thickness.max()))
        return self

    def predict(self,xy):
        xy=np.asarray(xy,float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Finite Nx2 targets required')
        d=self.tree.query(xy,k=2)[0]
        # Outside the hull, permit only a short distance to an observed contact.
        supported=(d[:,1]<=self.max_distance)&((self.hull.find_simplex(xy)>=0)|(d[:,0]<=self.edge_buffer))
        out=np.full((len(xy),2),np.nan)
        if supported.any():
            r=self.model((xy[supported]-self.center)/self.scale)
            out[supported,0]=r[:,0]
            out[supported,1]=np.clip(np.exp(np.clip(r[:,1],-20,20)),*self.thickness_bounds)
        return out[:,0],out[:,1],supported


class HarmonicContactTrend:
    """Two smooth spatial waves plus a plane, fitted to observed contacts.

    A hypothesis for structured curvature, not a universal geological prior.
    Wave directions, periods and amplitudes are inferred from training contacts.
    """
    def fit(self,xy,z):
        self.center=np.asarray(xy).mean(0);x=(np.asarray(xy)-self.center)/100
        y=np.asarray(z);best=None
        def design(k,xx=x):
            phase=xx@np.asarray(k).reshape(2,2).T
            return np.c_[np.ones(len(xx)),xx,np.sin(phase),np.cos(phase)]
        def residual(k):
            A=design(k);coef=np.linalg.lstsq(A,y,rcond=None)[0]
            return (A@coef-y)/20
        for a in [1.,2.,3.,4.]:
            for b in [1.,2.,3.,4.]:
                fit=least_squares(residual,[a,0,0,b],bounds=([.4,-2,-2,.4],[6,2,2,6]),max_nfev=150)
                score=float(np.mean(residual(fit.x)**2))
                if best is None or score<best[0]:best=(score,fit.x)
        self.frequency=best[1];self.coef=np.linalg.lstsq(design(self.frequency),y,rcond=None)[0]
        self.training_rmse=float(np.sqrt(best[0])*20)
        return self

    def predict(self,xy):
        x=(np.asarray(xy)-self.center)/100;phase=x@self.frequency.reshape(2,2).T
        return np.c_[np.ones(len(x)),x,np.sin(phase),np.cos(phase)]@self.coef


class DatumOffsetSurface(PairedContactSurface):
    """Local thickness/footprint with a shared curved datum and measured offset."""
    def fit_with_datum(self,xy,midpoint,thickness,datum):
        super().fit(xy,midpoint,thickness)
        self.datum=datum;self.offset=float(np.median(np.asarray(midpoint)-datum.predict(xy)))
        return self

    def predict(self,xy):
        _,thickness,supported=super().predict(xy)
        mid=self.datum.predict(xy)+self.offset;mid[~supported]=np.nan
        return mid,thickness,supported
