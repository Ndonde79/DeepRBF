# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Regular rotated block-model CSV interchange, with explicit geometry metadata."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def export_leapfrog_csv(path,centers,attributes,*,block_size,minimum_corner,dimensions,rotation=None):
    """Write CSV, .csv.txt import settings and .grid.json for Leapfrog.

    Rotation rows are right-handed world axes. minimum_corner is in world XYZ.
    Attributes are a mapping or DataFrame with one row per center. Geometry is
    retained exactly; this function never regrids or infers missing grid cells.
    The output is numerically verified, not tested inside a live Leapfrog GUI.
    """
    path=Path(path);xyz=np.asarray(centers,float);cell=np.asarray(block_size,float)
    corner=np.asarray(minimum_corner,float);dims=np.asarray(dimensions)
    axes=np.eye(3) if rotation is None else np.asarray(rotation,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():raise ValueError('centers must be finite nonempty N by 3')
    if cell.shape!=(3,) or not np.isfinite(cell).all() or (cell<=0).any():raise ValueError('Invalid block_size')
    if corner.shape!=(3,) or not np.isfinite(corner).all():raise ValueError('Invalid minimum_corner')
    if dims.shape!=(3,) or not np.issubdtype(dims.dtype,np.integer) or (dims<=0).any():raise ValueError('dimensions must contain positive integers')
    if axes.shape!=(3,3) or not np.isfinite(axes).all() or not np.allclose(axes@axes.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(axes),1.):raise ValueError('Invalid rotation axes')
    fractional=(xyz-corner)@axes.T/cell-.5;ijk=np.rint(fractional).astype(int)
    if not np.allclose(fractional,ijk,atol=1e-6,rtol=0) or (ijk<0).any() or (ijk>=dims).any():raise ValueError('Centroids do not match grid definition')
    if len(np.unique(ijk,axis=0))!=len(ijk):raise ValueError('Duplicate block centroids')
    data=pd.DataFrame(attributes).reset_index(drop=True)
    if len(data)!=len(xyz):raise ValueError('Attributes must match centers')
    reserved={'X','Y','Z','dX','dY','dZ','I','J','K','BLOCK_ID'}
    if reserved.intersection(data.columns):raise ValueError('Attributes contain reserved geometry column names')
    geometry=pd.DataFrame(dict(X=xyz[:,0],Y=xyz[:,1],Z=xyz[:,2],dX=cell[0],dY=cell[1],dZ=cell[2],
                               I=ijk[:,0],J=ijk[:,1],K=ijk[:,2],BLOCK_ID=np.arange(1,len(xyz)+1)))
    output=pd.concat([geometry,data],axis=1).sort_values(['K','J','I'])
    angles=-Rotation.from_matrix(axes.T).as_euler('ZYX',degrees=True)
    metadata=dict(rows=len(xyz),block_size_xyz=cell.tolist(),minimum_corner_world=corner.tolist(),
                  size_in_blocks_xyz=dims.tolist(),rotation_type='Custom Rotation',
                  rotations=[dict(axis=a,angle_degrees=float(v)) for a,v in zip('ZYX',angles)],
                  convention='Left handed, subsequent axes relative to previous rotation',indices='zero-based')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');output.to_csv(temporary,index=False,float_format='%.9f');temporary.replace(path)
    path.with_suffix('.grid.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    text='Leapfrog regular block model\nCentroids: X Y Z, world coordinates in metres\n'
    text+='Block size X Y Z: '+' '.join(map(str,cell))+'\n'
    text+='Size in blocks X Y Z: '+' '.join(map(str,dims))+'\n'
    text+='Minimum corner X Y Z (world): '+' '.join(f'{v:.9f}' for v in corner)+'\n'
    text+='Custom Rotation, left handed; each axis relative to the previous rotation:\n'
    text+='\n'.join(f'{a}: {v:.9f} degrees' for a,v in zip('ZYX',angles))
    text+='\nRotation origin with offsets: OFF\nImport via Block Models > Import Block Model > Regular.\nConfirm ALL centroids match the grid before finishing.\n'
    Path(str(path)+'.txt').write_text(text,encoding='utf-8')
    return metadata
