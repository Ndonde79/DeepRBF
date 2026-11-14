# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest,importlib.util
from pathlib import Path
import numpy as np
sp=importlib.util.spec_from_file_location('paired',Path(__file__).parents[1]/'deeprbf/paired_contact_surface.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class PairedContactsTest(unittest.TestCase):
    def setUp(self):
        self.xy=np.array([[0,0],[100,0],[0,100],[100,100],[50,50]],float)
        self.z=2+.1*self.xy[:,0]-.2*self.xy[:,1]
        self.model=m.PairedContactSurface(smoothing=0,neighbors=5,max_distance=200,edge_buffer=0).fit(self.xy,self.z,np.full(5,3.))
    def test_exact_contacts(self):
        z,t,ok=self.model.predict(self.xy)
        np.testing.assert_allclose(z,self.z,atol=1e-10);np.testing.assert_allclose(t,3);self.assertTrue(ok.all())
    def test_planar_reference(self):
        q=np.array([[25,30],[70,80]],float);z,t,ok=self.model.predict(q)
        np.testing.assert_allclose(z,2+.1*q[:,0]-.2*q[:,1],atol=1e-10)
        self.assertTrue(ok.all());self.assertTrue((t>0).all())
    def test_unsupported_is_missing(self):
        z,t,ok=self.model.predict([[500,500]])
        self.assertFalse(ok[0]);self.assertTrue(np.isnan(z[0]) and np.isnan(t[0]))
    def test_invalid_thickness(self):
        with self.assertRaises(ValueError):self.model.fit(self.xy,self.z,np.zeros(5))
    def test_duplicate_contacts(self):
        with self.assertRaises(ValueError):self.model.fit(np.zeros((5,2)),self.z,np.ones(5))
    def test_invalid_controls(self):
        with self.assertRaises(ValueError):m.PairedContactSurface(neighbors=1)

if __name__=='__main__':unittest.main()
