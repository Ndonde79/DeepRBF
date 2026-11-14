# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Conservative anisotropic IDW reference estimator with explicit support gates.

Block estimates integrate point estimates at tensor Gauss-Legendre nodes.
This reference produces no kriging variance or resource classification.
"""
import numpy as np
from scipy.spatial import cKDTree
from numpy.polynomial.legendre import leggauss

AXES = np.array([[-.2588190451025207,.9659258262890683,0.],
                 [.40821789367673483,.10938165494661499,-.9063077870366499],
                 [-.875426098065593,-.23456971600980442,-.42261826174069944]])


class LocalEstimator:
    def __init__(self, xyz, grades, holes, ranges=(160.,70.,30.), power=2.,
                 min_holes=3, min_samples=8, max_samples=24, max_per_hole=3):
        self.xyz = np.asarray(xyz, dtype=float)
        self.grades = np.asarray(grades, dtype=float)
        self.holes = np.asarray(holes).astype(str)
        self.ranges = np.asarray(ranges, dtype=float)
        if self.xyz.shape != (len(self.grades),3) or len(self.holes)!=len(self.grades):
            raise ValueError('Sample array sizes differ')
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.grades).all() or (self.grades<0).any():
            raise ValueError('Only audited finite nonnegative samples are accepted')
        if self.ranges.shape!=(3,) or not np.isfinite(self.ranges).all() or (self.ranges<=0).any() or not np.isfinite(power) or power<=0: raise ValueError('Invalid distance parameters')
        if any(type(v) is not int or v<1 for v in (min_holes,min_samples,max_samples,max_per_hole)) or min_samples>max_samples or min_holes>max_samples:
            raise ValueError('Invalid support limits')
        self.power=power; self.min_holes=min_holes; self.min_samples=min_samples
        self.max_samples=max_samples; self.max_per_hole=max_per_hole
        self.tree=cKDTree(self.transform(self.xyz))

    def transform(self, xyz):
        return np.asarray(xyz) @ AXES.T / self.ranges

    def points(self, targets):
        targets=np.asarray(targets,dtype=float)
        if targets.ndim!=2 or targets.shape[1]!=3 or not np.isfinite(targets).all():raise ValueError('Finite N by 3 targets required')
        neighborhoods=self.tree.query_ball_point(self.transform(targets),1.)
        result=[]
        for q, candidates in zip(targets,neighborhoods):
            candidates=np.asarray(candidates,dtype=int)
            distance=np.linalg.norm(self.transform(self.xyz[candidates]-q),axis=1)
            order=np.lexsort((candidates,distance))
            selected=[]; counts={}
            for pos in order:
                idx=candidates[pos]; hole=self.holes[idx]
                if counts.get(hole,0)>=self.max_per_hole: continue
                selected.append((idx,distance[pos]));counts[hole]=counts.get(hole,0)+1
                if len(selected)==self.max_samples: break
            n=len(selected); nh=len(counts)
            row=dict(estimate=np.nan,n_samples=n,n_holes=nh,status='INSUFFICIENT_SUPPORT')
            if n>=self.min_samples and nh>=self.min_holes:
                idx=np.array([p[0] for p in selected]);d=np.array([p[1] for p in selected])
                exact=d<1e-10
                if exact.any():
                    values=self.grades[idx[exact]]
                    if np.ptp(values)>1e-8:
                        row['status']='CONFLICTING_EXACT_SAMPLES'; result.append(row);continue
                    grade=float(values.mean())
                else:
                    weights=d**(-self.power);grade=float(np.dot(weights,self.grades[idx])/weights.sum())
                row.update(estimate=grade,status='SUPPORTED',nearest_scaled_distance=float(d.min()),
                           farthest_scaled_distance=float(d.max()))
            result.append(row)
        return result

    def blocks(self, centers, sizes, order=2):
        centers=np.asarray(centers,float); sizes=np.asarray(sizes,float)
        if centers.shape!=sizes.shape or centers.ndim!=2 or centers.shape[1]!=3:
            raise ValueError('Centers and sizes must be N by 3')
        if not np.isfinite(centers).all() or not np.isfinite(sizes).all() or (sizes<=0).any():
            raise ValueError('Invalid block geometry')
        if order not in (2,3): raise ValueError('Supported quadrature orders: 2,3')
        nodes,weights=leggauss(order)
        grid=np.array(np.meshgrid(nodes,nodes,nodes,indexing='ij')).reshape(3,-1).T
        w=np.prod(np.array(np.meshgrid(weights,weights,weights,indexing='ij')),axis=0).ravel()/8.
        rows=[]
        for center,size in zip(centers,sizes):
            points=self.points(center+grid*size/2)
            supported=all(p['status']=='SUPPORTED' for p in points)
            rows.append(dict(estimate=float(np.dot(w,[p['estimate'] for p in points])) if supported else np.nan,
                             status='SUPPORTED' if supported else 'INSUFFICIENT_QUADRATURE_SUPPORT',
                             min_samples=min(p['n_samples'] for p in points),
                             min_holes=min(p['n_holes'] for p in points),quadrature_points=len(grid)))
        return rows
