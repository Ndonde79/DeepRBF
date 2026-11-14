# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import deeprbf
import unittest
import numpy as np
from deeprbf.training_images import PatternAtlasDatabase,SNESIMAtlasSimulator
from deeprbf.snesim import SNESIMConfig,UNKNOWN


class AtlasTests(unittest.TestCase):
    def test_no_artificial_seam_patterns(self):
        tiles=[np.zeros((9,10),dtype=int),np.ones((9,10),dtype=int)]
        db=PatternAtlasDatabase(tiles,np.array([0,1]),np.array([[0,1],[0,-1],[1,0]]))
        self.assertTrue(np.all(db.patterns==db.centers[:,None]))
        np.testing.assert_allclose(db.prior,[.5,.5])
        p,_,_=db.query(np.array([0]),np.array([0]),1)
        np.testing.assert_array_equal(p,[1,0])

    def test_joint_simulation_conditioning_repeatability_and_tile_copy(self):
        rng=np.random.default_rng(88);tiles=[rng.integers(10,12,(12,14)),rng.integers(10,12,(12,14))]
        config=SNESIMConfig(n_realizations=2,n_multigrids=2,max_conditioning_nodes=8,search_radii=(2,2,1),seed=41)
        engine=SNESIMAtlasSimulator(tiles,config)
        hard=np.full((7,8),UNKNOWN,dtype=int);hard[1,1]=10;hard[5,5]=11
        a,_=engine.simulate(hard.shape,hard_data=hard);tiles[0][:]=99
        b,_=engine.simulate(hard.shape,hard_data=hard)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,1,1],10);np.testing.assert_array_equal(a[:,5,5],11)

    def test_incomplete_images_rejected(self):
        with self.assertRaises(ValueError):SNESIMAtlasSimulator([np.array([[10,UNKNOWN],[11,10]])])
        with self.assertRaises(ValueError):SNESIMAtlasSimulator([np.empty((0,3),dtype=int)])

    def test_conceptual_profiles_are_seeded_and_distinct(self):
        images=[]
        for name in deeprbf.GEOLOGICAL_PROFILES:
            a=deeprbf.conceptual_training_image(name)
            np.testing.assert_array_equal(a,deeprbf.conceptual_training_image(name))
            self.assertEqual(a.shape,(40,40,48));np.testing.assert_array_equal(np.unique(a),np.arange(5))
            images.append(a)
        for i,a in enumerate(images):
            for b in images[i+1:]:self.assertFalse(np.array_equal(a,b))


if __name__=='__main__':unittest.main()
