# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from deeprbf.shared_geology import joint_partition,shared_faces

class SharedGeologyTests(unittest.TestCase):
    def test_single_volume_and_internal_interface(self):
        a=np.ones((2,3,4),np.uint8)
        f,p=shared_faces(a)
        self.assertEqual(len(f),52)
        self.assertFalse(((p[:,0]>0)&(p[:,1]>0)).any())
        a[1]=2;f,p=shared_faces(a)
        self.assertEqual(int(((p[:,0]>0)&(p[:,1]>0)).sum()),12)
        self.assertEqual(len({tuple(sorted(map(tuple,x))) for x in f}),len(f))

    def test_envelope_and_anchor_constraints(self):
        mask=np.ones((4,3,2),bool);mask[0,0,0]=False
        s=np.ones(mask.shape+(4,),np.float32);a=np.zeros(mask.shape,np.uint8);a[1,1,1]=3;a[3,1,1]=2
        labels,margin=joint_partition(mask,s,a,5)
        self.assertTrue((labels[mask]>0).all())
        self.assertTrue((labels[~mask]==0).all())
        self.assertEqual(labels[1,1,1],3);self.assertEqual(labels[3,1,1],2)
        self.assertTrue(np.isfinite(margin).all())
        immediate,_=joint_partition(mask,s,a,0)
        self.assertEqual(immediate[1,1,1],3)

    def test_missing_initial_scores_rejected(self):
        with self.assertRaises(ValueError):joint_partition(np.ones((2,2,2),bool),np.zeros((2,2,2,4)),np.zeros((2,2,2),int))

if __name__=='__main__':unittest.main()
