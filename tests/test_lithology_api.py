# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
import deeprbf as d
import tempfile
from pathlib import Path
import pandas as pd


class LithologyAPI(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        self.y=np.array([10,20,10])
        self.q=np.vstack([self.x,[[.2,.4,0],[.8,.3,.1],[.2,.4,0]]])
        self.bounds=np.array([[[-np.inf,0],[-np.inf,np.inf]],[[0,np.inf],[-np.inf,np.inf]]])

    def test_public_methods_honor_arbitrary_codes_and_duplicates(self):
        for method,kwargs in [(d.estimate_lithology_ik,{}),(d.simulate_lithology_sis,dict(n_realizations=3)),
                              (d.simulate_plurigaussian,dict(n_realizations=3,bands=64,gibbs_sweeps=12,truncation_bounds=self.bounds))]:
            result=method(self.x,self.y,self.q,ranges=(2,2,2),**kwargs)
            self.assertEqual(result.probabilities.shape,(6,2))
            np.testing.assert_array_equal(result.modal[:3],self.y)
            np.testing.assert_allclose(result.probabilities.sum(1),1)
            np.testing.assert_array_equal(result.probabilities[3],result.probabilities[5])
            self.assertEqual(result.diagnostics['backend'],'hybrid_native_gibbs_numpy_scipy' if method==d.simulate_plurigaussian else 'native_cpp_sparse')
            if result.realizations is not None:
                self.assertEqual(result.realizations.shape,(3,6))
                np.testing.assert_array_equal(result.realizations[:,:3],np.tile(self.y,(3,1)))
                np.testing.assert_array_equal(result.realizations[:,3],result.realizations[:,5])

    def test_invalid_data_and_rule_fail_explicitly(self):
        with self.assertRaises(ValueError):d.estimate_lithology_ik([[0,0,0],[0,0,0]],[10,20],self.q)
        with self.assertRaises(ValueError):d.estimate_lithology_ik(self.x,self.y,self.q,ranges=(0,2,2))
        with self.assertRaises(ValueError):d.simulate_plurigaussian(self.x,self.y,self.q)
        with self.assertRaises(ValueError):d.simulate_plurigaussian(self.x,self.y,self.q,truncation_bounds=np.tile(self.bounds[:1],(2,1,1)))

    def test_public_snesim_export(self):
        ti=np.zeros((12,12),int);ti[4:8,:]=1
        hard=np.full((6,6),-2147483648,int);hard[2,2]=1
        cfg=d.SNESIMConfig(n_realizations=2,max_conditioning_nodes=8,n_multigrids=1,min_replicates=2)
        sims,diag=d.simulate_snesim(ti,(6,6),config=cfg,hard_data=hard)
        np.testing.assert_array_equal(sims[:,2,2],[1,1])

    def test_desurvey_and_rotated_csv_roundtrip(self):
        np.testing.assert_allclose(d.minimum_curvature([0,0,100],[0],[90],[0],[0,50]),[[0,0,100],[0,0,50]],atol=1e-10)
        with self.assertRaises(ValueError):d.minimum_curvature([0,0,0],[0,0],[90,90],[0,0],[1])
        z=np.deg2rad(105);x=np.deg2rad(65)
        rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        axes=(rz@rx).T;corner=np.array([480000.,1700000.,1400.]);cell=np.array([40,40,20])
        centers=corner+(np.array([[0,0,0],[1,2,3]])+.5)*cell@axes
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.csv'
            metadata=d.export_leapfrog_csv(path,centers,{'LITHOLOGY':['A','B']},block_size=cell,minimum_corner=corner,dimensions=[3,4,5],rotation=axes)
            output=pd.read_csv(path)
            np.testing.assert_allclose(output[['X','Y','Z']],centers,atol=1e-7)
            np.testing.assert_array_equal(output[['I','J','K']],[[0,0,0],[1,2,3]])
            np.testing.assert_allclose([v['angle_degrees'] for v in metadata['rotations']],[-105,0,-65],atol=1e-8)
            self.assertTrue(Path(str(path)+'.txt').exists())


if __name__=='__main__':unittest.main()
