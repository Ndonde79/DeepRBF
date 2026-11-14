# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('gp',Path(__file__).parents[1]/'deeprbf/log_grade_gp.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LogGradeGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xy=np.array([[0,0],[100,0],[0,100],[100,100],[50,50],[75,25]],float)
        cls.model=m.LogGradeGP().fit(cls.xy,[1,2,3,4,2,2.5],np.full(6,.1))
    def test_positive_and_moment_identity(self):
        r=self.model.predict(self.xy)
        self.assertTrue((r['mean_grade']>0).all());self.assertTrue((r['latent_log_variance']>=0).all())
        np.testing.assert_allclose(r['mean_grade'],np.exp(r['latent_log_mean']+.5*r['latent_log_variance']))
    def test_batch_invariance(self):
        r=self.model.predict(self.xy)['mean_grade']
        np.testing.assert_allclose(r,np.r_[self.model.predict(self.xy[:3])['mean_grade'],self.model.predict(self.xy[3:])['mean_grade']])
    def test_invalid_grade(self):
        with self.assertRaises(ValueError):m.LogGradeGP().fit(self.xy,np.zeros(6),np.ones(6))
    def test_invalid_noise(self):
        with self.assertRaises(ValueError):m.LogGradeGP().fit(self.xy,np.ones(6),-np.ones(6))
    def test_translation_invariance(self):
        shift=np.array([410000,9680000]);other=m.LogGradeGP().fit(self.xy+shift,[1,2,3,4,2,2.5],np.full(6,.1))
        np.testing.assert_allclose(self.model.predict(self.xy)['mean_grade'],other.predict(self.xy+shift)['mean_grade'],rtol=1e-4)
if __name__=='__main__':unittest.main()
