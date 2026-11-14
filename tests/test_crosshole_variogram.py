# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from deeprbf.crosshole_variogram import fit_crosshole_variogram

class CrossholeVariogramTests(unittest.TestCase):
    def test_reproducible_finite_fit_and_pair_cap(self):
        rng=np.random.default_rng(51);x=rng.uniform(0,250,(120,3));h=np.repeat(np.arange(8),15);y=2+np.sin(x[:,0]/80)+rng.normal(0,.1,120)
        a=fit_crosshole_variogram(x,y,h,pair_cap=3);b=fit_crosshole_variogram(x,y,h,pair_cap=3)
        self.assertEqual(a,b);self.assertGreater(a['sill'],0);self.assertGreater(a['range'],0);self.assertLessEqual(max(a['pairs']),28*3)
    def test_no_crosshole_distances(self):
        a=fit_crosshole_variogram(np.zeros((40,3)),np.ones(40),np.repeat(np.arange(4),10))
        self.assertEqual(a['status'],'INSUFFICIENT_LAG_BINS')
    def test_invalid_hole_support(self):
        with self.assertRaises(ValueError):fit_crosshole_variogram(np.ones((40,3)),np.ones(40),np.zeros(40))

if __name__=='__main__':unittest.main()
