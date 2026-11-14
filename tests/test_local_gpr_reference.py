# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Independent dense-reference checks for the experimental local GP."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('local_gpr', Path(__file__).parents[1]/'deeprbf/local_gpr.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LocalGPReference(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
        self.y=np.array([1.,2.,3.,4.])
        self.g=module.LocalGPR(self.x,self.y,np.arange(4),axes=np.eye(3),ranges=[3]*3,
            radii=[10]*3,sill=2,nugget=.1)

    def test_dense_gp_and_augmented_ok(self):
        q=np.array([[.3,.4,0.]])
        r=self.g.predict(q)
        K=self.g.kernel(self.x,self.x)+np.eye(4)*2*(.1+1e-8)
        k=self.g.kernel(self.x,q).ravel()
        self.assertAlmostEqual(r['gpr'][0],self.y.mean()+k@np.linalg.solve(K,self.y-self.y.mean()))
        A=np.block([[K,np.ones((4,1))],[np.ones((1,4)),np.zeros((1,1))]])
        w=np.linalg.solve(A,np.r_[k,1])[:4]
        self.assertAlmostEqual(r['ok'][0],w@self.y)

    def test_integrated_block_reference(self):
        q=np.array([[.3,.4,0.]])
        r=self.g.predict(q,block_size=[2,2,2],discretization=2)
        points=q+np.array([[a,b,c] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5]])
        K=self.g.kernel(self.x,self.x)+np.eye(4)*2*(.1+1e-8)
        k=self.g.kernel(self.x,points).mean(1)
        v=self.g.kernel(points,points).mean()-k@np.linalg.solve(K,k)
        self.assertAlmostEqual(r['gpr_latent_variance'][0],v)
        self.assertTrue(np.isnan(r['observation_variance'][0]))

    def test_exact_conditioning(self):
        g=module.LocalGPR(self.x,self.y,np.arange(4),axes=np.eye(3),ranges=[3]*3,
            radii=[10]*3,sill=2,nugget=0,jitter=0)
        np.testing.assert_allclose(g.predict(self.x)['gpr'],self.y)

    def test_unsupported_is_not_filled(self):
        self.assertTrue(np.isnan(self.g.predict([[100,100,100]])['gpr'][0]))

    def test_noisy_duplicate_locations(self):
        g=module.LocalGPR(np.repeat(self.x,2,axis=0),np.repeat(self.y,2),np.arange(8),
            axes=np.eye(3),ranges=[3]*3,radii=[10]*3,sill=2,nugget=.1)
        self.assertTrue(np.isfinite(g.predict([[.3,.4,0.]])['gpr']).all())

if __name__=='__main__': unittest.main()
