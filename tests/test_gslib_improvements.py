# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from scipy.special import ndtri
import deeprbf as d

class GslibImprovements(unittest.TestCase):
    def test_covariance_against_gslib_formulas(self):
        for kind,formula in [(d.VariogramModelType.Spherical,lambda h:max(0.,1-1.5*h+.5*h**3) if h<1 else 0.),(d.VariogramModelType.Exponential,lambda h:np.exp(-3*h)),(d.VariogramModelType.Gaussian,lambda h:np.exp(-3*h*h))]:
            vg=d.NestedVariogram(.2);st=d.VariogramStructure();st.type=kind;st.range=4.;st.contribution=.8;vg.add_structure(st)
            self.assertAlmostEqual(vg.covariance(0),1.)
            for distance in [.1,1.,4.,8.]:self.assertAlmostEqual(vg.covariance(distance),.8*formula(distance/4.),places=13)
    def test_midpoints_ties_weights_roundtrip(self):
        values=np.array([10.,1.,10.,100.]); t=d.NormalScoreTransform(values,[1,2,3,4])
        np.testing.assert_allclose(t.scores,ndtri([.1,.4,.8]),atol=1e-14)
        np.testing.assert_array_equal(t.inverse_transform(t.transform(values)),values)
        self.assertEqual(t.transform([10])[0],t.transform([10,10])[1])
        np.testing.assert_array_equal(t.inverse_transform([-100,100]),[1,100])
    def test_permutation_and_weight_scaling(self):
        a=d.NormalScoreTransform([9,1,4,4],[2,3,4,5]);b=d.NormalScoreTransform([4,9,4,1],np.array([5,2,4,3])*1e200)
        np.testing.assert_allclose(a.scores,b.scores)
        self.assertEqual(d.NormalScoreTransform([7,7]).transform([7])[0],0.)
    def test_invalid_normal_scores(self):
        for values,weights in [([],None),([1,float('nan')],None),([1,2],[1,0]),([1,2],[1]),([1,2],[-1,2])]:
            with self.assertRaises(ValueError):d.NormalScoreTransform(values,weights)
    def test_sgs_exact_conditioning_and_duplicate_targets(self):
        xyz=[[0,0,0],[1,0,0]];values=[2.,100.];targets=xyz+[[.5,0,0],[.5,0,0]]
        result=d.run_sgs_with_range(xyz,values,targets,n_realizations=8,min_neighbors=5,max_neighbors=5,nugget=.4)
        np.testing.assert_array_equal(result[:2],np.tile(np.array(values)[:,None],(1,8)))
        np.testing.assert_array_equal(result[2],result[3])
        with self.assertRaises(ValueError):d.run_sgs_with_range([xyz[0],xyz[0]],[1,2],targets)
    def test_sis_exact_conditioning_and_duplicates(self):
        xyz=[[0,0,0],[1,0,0]];targets=xyz+[[.5,0,0],[.5,0,0]]
        result=d.run_sis(xyz,[10,20],targets,[10,20],n_realizations=8,max_neighbors=5,min_neighbors=5)
        np.testing.assert_array_equal(result[:2],np.tile([[10],[20]],(1,8)))
        np.testing.assert_array_equal(result[2],result[3])
        with self.assertRaises(ValueError):d.run_sis([xyz[0],xyz[0]],[10,20],targets,[10,20])
    def test_anisotropic_sis_exact_conditioning(self):
        xyz=[[0,0,0],[1,0,0]];targets=xyz+[[.5,0,0],[.5,0,0]]
        variograms=[]
        for cat in [10,20]:
            v=d.IndicatorVariogram();v.category=cat;variograms.append(v)
        result=d.run_sis_anisotropic(xyz,[10,20],targets,[10,20],variograms,n_realizations=5,min_neighbors=5,max_neighbors=5)
        np.testing.assert_array_equal(result[:2],np.tile([[10],[20]],(1,5)))
        np.testing.assert_array_equal(result[2],result[3])
    def test_gaussian_space_not_restandardized(self):
        # Mean 10, std 10: an unconditioned target must still draw near N(0, sill).
        r=d.run_sgs_with_range([[0,0,0],[1,0,0]],[0,20],[[100,0,0]],n_realizations=2000,min_neighbors=2,max_neighbors=2,search_radius=1,sill=4,nugget=.1,standardize=False)
        self.assertLess(abs(r.mean()),.2);self.assertLess(abs(r.var()-4),.5)
    def test_normal_score_workflow_skewed_values(self):
        xyz=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]];values=[1.,2.,10.,500.];targets=xyz+[[.3,.3,.3]]
        a=d.run_sgs_normal_score(xyz,values,targets,n_realizations=5,min_neighbors=2,seed=17)
        b=d.run_sgs_normal_score(xyz,values,targets,n_realizations=5,min_neighbors=2,seed=17)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:4],np.tile(np.array(values)[:,None],(1,5)))
        self.assertTrue(np.isfinite(a).all());self.assertTrue(((a>=1)&(a<=500)).all())
if __name__=='__main__':unittest.main(verbosity=2)
