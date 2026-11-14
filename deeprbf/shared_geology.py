# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Joint voxel partition: exhaustive domains, shared faces, separate support.

Harmonic relaxation is a geometric interpolation hypothesis, not evidence of
stratigraphic order or faults. Caller supplies the modelling envelope and seeds.
"""
import numpy as np
from scipy.ndimage import convolve

def joint_partition(mask, scores, anchors, iterations=30):
    mask=np.asarray(mask,bool);s=np.asarray(scores,np.float32).copy()
    anchors=np.asarray(anchors)
    if s.shape!=mask.shape+(4,) or anchors.shape!=mask.shape:raise ValueError('Grid shape mismatch')
    if not np.isfinite(s).all() or (s<0).any():raise ValueError('Finite nonnegative scores required')
    if not mask.any() or ((anchors<0)|(anchors>4)).any():raise ValueError('Invalid envelope or anchor code')
    if ((anchors>0)&~mask).any():raise ValueError('Anchors outside envelope')
    if not isinstance(iterations,int) or iterations<0:raise ValueError('Nonnegative integer iterations required')
    total=s.sum(axis=-1)
    if (total[mask]<=0).any():raise ValueError('Every envelope cell needs initial scores')
    s/=np.maximum(total[...,None],1e-20);s[~mask]=0
    fixed=anchors>0
    kernel=np.zeros((3,3,3),np.float32)
    for axis,weight in enumerate([1.,1.,4.]):
        for side in [0,2]:
            ix=[1,1,1];ix[axis]=side;kernel[tuple(ix)]=weight
    degree=convolve(mask.astype(np.float32),kernel,mode='constant')
    initial=s.copy()
    for _ in range(iterations):
        for c in range(4):
            smooth=convolve(s[...,c],kernel,mode='constant')/np.maximum(degree,1)
            s[...,c]=.85*smooth+.15*initial[...,c]
        s[~mask]=0
        for c in range(4):s[...,c][fixed]=(anchors[fixed]==c+1)
    for c in range(4):s[...,c][fixed]=(anchors[fixed]==c+1)
    labels=np.zeros(mask.shape,np.uint8);labels[mask]=s[mask].argmax(axis=1)+1
    assert np.all(labels[fixed]==anchors[fixed])
    margin=np.zeros(mask.shape,np.float32);rank=np.sort(s[mask],axis=1);margin[mask]=rank[:,-1]-rank[:,-2]
    return labels,margin

def shared_faces(labels):
    """Each inter-domain quad appears ONCE with both adjacent domain codes.

    Coordinates are integer lattice corners; orientation is +axis. Domain 0
    denotes exterior. No separately fitted surfaces or snap tolerance involved.
    """
    a=np.pad(labels,1);faces=[];pairs=[]
    for axis in range(3):
        left=[slice(1,-1)]*3;right=left.copy();left[axis]=slice(0,-1);right[axis]=slice(1,None)
        lo=a[tuple(left)];hi=a[tuple(right)];ix=np.argwhere(lo!=hi)
        other=[j for j in range(3) if j!=axis]
        corners=np.zeros((4,3),int);corners[1,other[0]]=1;corners[2,other]=1;corners[3,other[1]]=1
        faces.append(ix[:,None,:]+corners);pairs.append(np.c_[lo[tuple(ix.T)],hi[tuple(ix.T)]])
    return np.concatenate(faces),np.concatenate(pairs)
