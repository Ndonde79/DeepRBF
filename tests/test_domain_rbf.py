# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from deeprbf.domain_rbf import DomainRBF

class DomainRBFTests(unittest.TestCase):
    def model(self,domains=(0,1),**kw):
        return DomainRBF([10,20],domains,ranges=[10,10,10],**kw)

    def test_unsupported_is_flagged_and_returns_training_prior(self):
        m=self.model().fit([[0,0,0],[1,0,0]],[10,20],['a','a'])
        p,d=m.predict([[5,0,0]])
        self.assertFalse(d['supported'][0]);np.testing.assert_allclose(p[0],m.prior)

    def test_collocated_conflicts_are_fractional_and_permutation_invariant(self):
        x=np.zeros((3,3));y=np.array([10,20,20]);h=np.array(['a','b','c']);w=np.array([2,1,3])
        for order in [np.arange(3),np.array([2,0,1])]:
            m=self.model().fit(x[order],y[order],h[order],w[order]);p,d=m.predict([[0,0,0]])
            np.testing.assert_allclose(p,[[1/3,2/3]]);self.assertTrue(d['exact_conditioned'][0])

    def test_single_domain_equals_flat_for_arbitrary_gate(self):
        m=self.model(domains=(0,0),gate_scale=[1,1,.1],min_holes=1).fit([[0,0,0],[10,0,0]],[10,20],['a','b'])
        a,_=m.predict([[3,0,0]],hierarchical=True);b,_=m.predict([[3,0,0]],hierarchical=False)
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_hole_quota_prevents_dense_hole_excluding_cross_hole_support(self):
        x=np.r_[np.column_stack([np.linspace(0,1,20),np.zeros((20,2))]),[[2,0,0],[3,0,0]]]
        m=self.model(max_per_hole=1,max_neighbors=3).fit(x,[10]*20+[20,20],['a']*20+['b','c'])
        p,d=m.predict([[0,1,0]])
        self.assertEqual(d['support_holes'][0],3);self.assertTrue(d['supported'][0]);self.assertGreater(p[0,1],.5)

    def test_input_controls_reject_invalid_data(self):
        with self.assertRaises(ValueError):DomainRBF([10,20],[0,1],ranges=[0,1,1])
        with self.assertRaises(ValueError):self.model().fit([[0,0,0]],[30],['a'])

    def test_probability_calibration_preserves_exact_conditioning(self):
        m=self.model(min_holes=1).fit([[0,0,0],[10,0,0]],[10,20],['a','b'])
        p,_=m.predict([[0,0,0],[2,0,0]],prior_mix=1.)
        np.testing.assert_allclose(p[0],[1,0]);np.testing.assert_allclose(p[1],m.prior)
        with self.assertRaises(ValueError):m.predict([[2,0,0]],prior_mix=-.1)

if __name__=='__main__':unittest.main()
