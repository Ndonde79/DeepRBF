# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Validated Surpac binary pair reader and open-pit excavation support integration."""
from pathlib import Path
import struct
import numpy as np
from shapely import polygons,points
from shapely.strtree import STRtree

def read_surpac_binary_pair(dtm,strfile):
    """Read this verified legacy binary record layout, rejecting other variants."""
    dtm=Path(dtm);strfile=Path(strfile);raw=dtm.read_bytes();header=raw.split(b'\r\n',1)[0]
    if header.split(b',',1)[0].decode().lower()!=strfile.name.lower():raise ValueError('DTM references a different STR')
    # Locate the first typed triangle following the trisolation metadata.
    meta=raw.find(b'algorithm=legacy\0')
    if meta<0:raise ValueError('Unsupported DTM metadata')
    pos=meta+len(b'algorithm=legacy\0');rows=[]
    while pos+33<=len(raw) and raw[pos:pos+4]==b'\0\0\0\3':
        row=struct.unpack_from('>8i',raw,pos)
        if row[1]!=len(rows)+1 or raw[pos+32]!=0:raise ValueError('Invalid triangle sequence or delimiter')
        rows.append(row);pos+=33
    if not rows or raw[pos:]!=b'\xff'*8:raise ValueError('Unsupported/truncated DTM tail')
    rows=np.asarray(rows,int)
    if (rows[:,2:5]<=0).any() or (rows[:,5:]<0).any() or (rows[:,5:]>len(rows)).any():raise ValueError('Invalid triangle references')
    raw= strfile.read_bytes();first=raw.find(b'\r\n');second=raw.find(b'\r\n',first+2)
    if second<0 or raw[second+2:second+6]!=b'\0'*4:raise ValueError('Unsupported STR header')
    pos=second+6;vertices=[];strings=[]
    while pos+29<=len(raw):
        string,description=struct.unpack_from('>iB',raw,pos)
        if string==0:break
        if string<0 or description!=0:raise ValueError('Unsupported STR record description/layout')
        y,x,z=struct.unpack_from('>ddd',raw,pos+5);vertices.append((x,y,z));strings.append(string);pos+=29
    tail=raw[pos:]
    if not tail.endswith(b'END\0') or tail[:-4].strip(b'\0'):raise ValueError('Unsupported STR tail')
    vertices=np.asarray(vertices,float);faces=rows[:,2:5]-1
    if not np.isfinite(vertices).all() or faces.max()>=len(vertices):raise ValueError('Invalid coordinates or out-of-range vertex')
    return vertices,faces,dict(dtm_header=header.decode(),vertices=len(vertices),triangles=len(faces),unused_vertices=int(len(vertices)-len(np.unique(faces))),strings=sorted(set(strings)),record_layout='STR int32 string + zero description byte + 3 big-endian float64 YXZ; DTM 8 big-endian int32 + null delimiter')

class PitSurface:
    def __init__(self,vertices,faces):
        self.vertices=np.asarray(vertices,float);raw_faces=np.asarray(faces)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not len(self.vertices) or not np.isfinite(self.vertices).all():raise ValueError('Finite nonempty N by 3 vertices required')
        if raw_faces.ndim!=2 or raw_faces.shape[1]!=3 or not len(raw_faces) or not np.issubdtype(raw_faces.dtype,np.integer) or (raw_faces<0).any() or (raw_faces>=len(self.vertices)).any():raise ValueError('Valid integer triangle indices required')
        self.faces=raw_faces.astype(int);tri=self.vertices[self.faces]
        a,b,c=tri[:,:,:2].transpose(1,0,2);den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        self.vertical_faces=int(np.sum(np.abs(den)<=1e-10));valid=np.abs(den)>1e-10
        self.tri=tri[valid];self.den=den[valid];self.tree=STRtree(polygons(self.tri[:,:,:2]))
    def at(self,xy):
        xy=np.asarray(xy,float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Finite XY required')
        lower=np.full(len(xy),np.inf);upper=np.full(len(xy),-np.inf)
        if len(xy):
            qi,ti=self.tree.query(points(xy),predicate='intersects')
            a,b,c=self.tri[ti].transpose(1,0,2);xx=xy[qi,0];yy=xy[qi,1];den=self.den[ti]
            w=((b[:,1]-c[:,1])*(xx-c[:,0])+(c[:,0]-b[:,0])*(yy-c[:,1]))/den
            u=((c[:,1]-a[:,1])*(xx-c[:,0])+(a[:,0]-c[:,0])*(yy-c[:,1]))/den
            zz=w*a[:,2]+u*b[:,2]+(1-w-u)*c[:,2]
            np.minimum.at(lower,qi,zz);np.maximum.at(upper,qi,zz)
        covered=np.isfinite(lower);ambiguous=covered&((upper-lower)>.05)
        height=np.where(covered&~ambiguous,lower,np.nan)
        return height,ambiguous

def excavation_fraction(z_centres,height_samples,*,block_height=5.):
    """XY midpoint quadrature and exact vertical integration above pit surface.

    Missing footprint samples contribute zero. Ambiguous hits must be separately
    audited by caller. No invented upper lid: the source rock block is the top.
    """
    z=np.asarray(z_centres,float);h=np.asarray(height_samples,float)
    if z.ndim!=1 or h.ndim!=2 or h.shape[0]!=len(z) or h.shape[1]==0 or not np.isfinite(z).all() or not np.isfinite(block_height) or block_height<=0:raise ValueError('Aligned block centres and nonempty surface samples required')
    fractions=np.clip((z[:,None]+block_height/2-h)/block_height,0,1)
    fractions[~np.isfinite(h)]=0
    return fractions.mean(1)
