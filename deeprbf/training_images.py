# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Pool multiple training images without inventing patterns at tile seams."""
import numpy as np
from .snesim import PatternDatabase,SNESIMMineralSimulator,UNKNOWN


class PatternAtlasDatabase(PatternDatabase):
    def __init__(self,tiles,categories,offsets):
        self.tiles=tuple(np.asarray(tile,dtype=np.int64) for tile in tiles)
        super().__init__(self.tiles[0],categories,offsets)

    def _scan(self):
        original=self.ti;patterns=[];centers=[]
        try:
            for tile in self.tiles:
                self.ti=tile
                p,c=super()._scan();patterns.append(p);centers.append(c)
        finally:self.ti=original
        return np.concatenate(patterns),np.concatenate(centers)


class SNESIMAtlasSimulator(SNESIMMineralSimulator):
    """SNESIM with pooled within-image counts, never cross-image neighborhoods.

    Tiles must share physical cell size and geological coordinate convention.
    Pattern counts weight tiles by valid centers; no implicit geological mixing
    or stationarity correction is inferred by this class.
    """
    def __init__(self,training_images,config=None,categories=None):
        tiles=tuple(np.asarray(t) for t in training_images)
        if not tiles or any(t.ndim not in (2,3) or t.ndim!=tiles[0].ndim for t in tiles):
            raise ValueError('Training images must be nonempty 2D or 3D arrays of matching dimensionality')
        if any(t.size==0 for t in tiles):raise ValueError('Training images cannot have empty dimensions')
        if any(not np.issubdtype(t.dtype,np.integer) for t in tiles):raise ValueError('Integer categories required')
        if any((t==UNKNOWN).any() for t in tiles):raise ValueError('Atlas requires complete images; use an incomplete-TI algorithm for sparse inputs')
        inferred=np.unique(np.concatenate([np.unique(t) for t in tiles]))
        categories=inferred if categories is None else np.asarray(categories)
        if any(not np.isin(t,categories).all() for t in tiles):raise ValueError('Image contains an undeclared category')
        self.training_images=tuple(t.astype(np.int64,copy=True) for t in tiles)
        super().__init__(tiles[0],config,categories)

    def _make_pattern_database(self,offsets):
        return PatternAtlasDatabase(self.training_images,self.categories,offsets)
