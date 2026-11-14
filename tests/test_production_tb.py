# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import sys, unittest, os
from pathlib import Path
_dll = os.add_dll_directory(str(Path(sys.prefix)/"Library"/"bin")) if os.name == "nt" else None
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from deeprbf_full_turning_bands_v3 import (VariogramSpec,TBSpec,TurningBandsFieldV3,conditional_tb_simulation_v3,clean_conditioning_data)
class ProductionTB(unittest.TestCase):
    def test_covariance_against_independent_formulas(self):
        q=np.array([[0.,0.,0.],[.15,0,0],[.5,0,0],[1.,0,0]])
        for model in ('gaussian','exponential'):
            spec=VariogramSpec(model=model,sill=.85,nugget=.15,ranges=(1,1,1))
            values=np.array([TurningBandsFieldV3(spec,TBSpec(bands=96,seed=487),r).evaluate(q) for r in range(3000)])
            d=np.linalg.norm(q[:,None]-q,axis=2)
            expected=.85*np.exp(-3*(d*d if model=='gaussian' else d))+np.eye(4)*.15
            np.testing.assert_allclose(np.cov(values,rowvar=False),expected,atol=.07,rtol=0)
    def test_chunk_order_duplicates_and_hard_data(self):
        rng=np.random.default_rng(51); x=rng.uniform(0,10,(18,3)); y=np.exp(x[:,0]/10)
        q=np.vstack([x[:3],rng.uniform(0,10,(20,3)),x[:1]])
        spec=VariogramSpec(nugget=.1,ranges=(8,5,3));tb=TBSpec(bands=128,max_neighbors=10)
        a,_=conditional_tb_simulation_v3(x,y,q,spec,tb,4)
        b=np.concatenate([conditional_tb_simulation_v3(x,y,v,spec,tb,4)[0] for v in (q[:7],q[7:15],q[15:])],axis=1)
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)
        np.testing.assert_allclose(a[:,:3],np.tile(y[:3],(4,1)),atol=2e-7,rtol=0)
        np.testing.assert_array_equal(a[:,0],a[:,-1])
        c,_=conditional_tb_simulation_v3(x,y,q[::-1],spec,tb,4)
        np.testing.assert_allclose(a,c[:,::-1],atol=1e-6,rtol=0)
    def test_conflicting_duplicates_rejected(self):
        with self.assertRaises(ValueError):clean_conditioning_data(np.zeros((2,3)),np.array([1.,2.]))
    def test_unsupported_spherical_rejected(self):
        with self.assertRaises(ValueError):TurningBandsFieldV3(VariogramSpec(model='spherical'),TBSpec(),0)
if __name__=='__main__':unittest.main(verbosity=2)

