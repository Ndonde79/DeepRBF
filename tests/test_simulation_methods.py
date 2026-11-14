# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from scipy.integrate import quad
from deeprbf import GaussianCovariance, TurningBandsSimulator, DirectBlockSimulator, simulate_tb, simulate_dbs

class SimulationMethods(unittest.TestCase):
    def test_covariance_models_and_rotation(self):
        rotation=np.array([[0,1,0],[-1,0,0],[0,0,1]])
        cov=GaussianCovariance(ranges=[2,4,8],rotation=rotation,sill=2)
        np.testing.assert_allclose(cov.matrix([[0,0,0]],[[1,0,0],[0,1,0]])[0],2*np.exp(-3*np.array([.25,.5])**2))
        self.assertAlmostEqual(GaussianCovariance('exponential').matrix([[0,0,0]],[[1,0,0]])[0,0],np.exp(-3))
    def test_tb_spectra_reproduce_covariance(self):
        points=np.array([[0,0,0],[.2,0,0],[0,.4,0],[.2,.2,.2],[1,0,0]])
        for name in ['gaussian','exponential']:
            covariance=GaussianCovariance(name)
            tb=TurningBandsSimulator(covariance,bands=16000,spectrum_seed=24)
            np.testing.assert_allclose(tb.covariance_matrix(points),covariance.matrix(points),atol=.022,rtol=0)
            np.testing.assert_allclose(np.diag(tb.covariance_matrix(points)),1,atol=1e-12)
    def test_tb_conditioning_duplicates_and_chunk_consistency(self):
        tb=TurningBandsSimulator(bands=128)
        x=[[0,0,0],[1,0,0]];y=[2.,-1.];q=x+[[.2,.3,.1],[.6,0,0],[.2,.3,.1]]
        a=tb.simulate(q,sample_xyz=x,sample_values=y,n_realizations=4,chunk_size=1)
        b=tb.simulate(q,sample_xyz=x,sample_values=y,n_realizations=4,chunk_size=10)
        np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_array_equal(a[:2],np.tile(np.array(y)[:,None],(1,4)))
        np.testing.assert_allclose(a[2],a[4],atol=1e-12)
        reversed_result=tb.simulate(q[::-1],sample_xyz=x,sample_values=y,n_realizations=4)
        np.testing.assert_allclose(a,reversed_result[::-1],atol=1e-12)
    def test_tb_conditional_distribution(self):
        tb=TurningBandsSimulator(bands=256,mean=.2)
        x=[[0,0,0],[1,0,0]];q=[[.3,0,0],[.6,.2,0]];y=np.array([.4,-.2]);noise=.1
        kd=tb.covariance_matrix(x)+noise*np.eye(2);cross=tb.covariance_matrix(q,x)
        expected_mean=.2+cross@np.linalg.solve(kd,y-.2)
        expected_cov=tb.covariance_matrix(q)-cross@np.linalg.solve(kd,cross.T)
        result=tb.simulate(q,sample_xyz=x,sample_values=y,measurement_variance=noise,n_realizations=12000,seed=92)
        np.testing.assert_allclose(result.mean(axis=1),expected_mean,atol=.025)
        np.testing.assert_allclose(np.cov(result),expected_cov,atol=.025)
    def test_dbs_point_limit_conditional_moments(self):
        cov=GaussianCovariance(ranges=2,sill=1.5);db=DirectBlockSimulator(cov,mean=.2)
        x=np.array([[0,0,0],[1,1,0]]);q=np.array([[.5,0,0],[1,0,0]]);y=np.array([.4,-.2])
        k=lambda a,b:1.5*np.exp(-3*np.sum((a[:,None]-b[None,:])**2,axis=2)/4)
        kd=k(x,x);cross=k(q,x)
        m=db.conditional_moments(q,[0,0,0],sample_xyz=x,sample_values=y)
        np.testing.assert_allclose(m.mean,.2+cross@np.linalg.solve(kd,y-.2),atol=1e-12)
        np.testing.assert_allclose(m.covariance,k(q,q)-cross@np.linalg.solve(kd,cross.T),atol=1e-12)
    def test_dbs_block_variance_independent_integral(self):
        lengths=np.array([1.,1.5,.75]);ranges=np.array([3.,4.,2.]);sill=2.
        expected=sill
        for length,r in zip(lengths,ranges):
            expected*=2/length**2*quad(lambda h:(length-h)*np.exp(-3*(h/r)**2),0,length,epsabs=1e-12)[0]
        db=DirectBlockSimulator(GaussianCovariance(ranges=ranges,sill=sill),quadrature_order=8)
        actual=db.conditional_moments([[0,0,0]],lengths).covariance[0,0]
        self.assertAlmostEqual(actual,expected,places=10)
        self.assertLess(actual,sill)
    def test_dbs_covariance_and_sample_distribution(self):
        db=DirectBlockSimulator(GaussianCovariance(ranges=2),quadrature_order=3)
        q=[[.5,0,0],[1,0,0]];kw=dict(sample_xyz=[[0,0,0]],sample_values=[.7])
        m=db.conditional_moments(q,[.5,.5,.5],**kw)
        a=db.simulate(q,[.5,.5,.5],n_realizations=16000,seed=71,**kw)
        np.testing.assert_allclose(a.mean(axis=1),m.mean,atol=.025)
        np.testing.assert_allclose(np.cov(a),m.covariance,atol=.025)
        self.assertGreater(m.covariance[0,1],0)
    def test_dbs_hard_point_and_identical_blocks(self):
        db=DirectBlockSimulator()
        a=db.simulate([[0,0,0],[1,0,0],[1,0,0]],[[0,0,0],[.5,.5,.5],[.5,.5,.5]],sample_xyz=[[0,0,0]],sample_values=[3.],n_realizations=4)
        np.testing.assert_array_equal(a[0],3.)
        np.testing.assert_allclose(a[1],a[2],atol=5e-8)
    def test_dbs_quadrature_convergence(self):
        cov=GaussianCovariance(ranges=2)
        values=[DirectBlockSimulator(cov,quadrature_order=o).conditional_moments([[0,0,0]],[1,1,1]).covariance[0,0] for o in [2,4,8]]
        self.assertLess(abs(values[1]-values[2]),abs(values[0]-values[2]))
    def test_invalid_inputs_and_explicit_limits(self):
        for factory in [lambda:GaussianCovariance(ranges=[0,1,1]),lambda:GaussianCovariance(rotation=np.ones((3,3))),lambda:TurningBandsSimulator(GaussianCovariance('spherical')),lambda:TurningBandsSimulator(bands=0),lambda:DirectBlockSimulator(quadrature_order=9)]:
            with self.assertRaises(ValueError):factory()
        for simulator in [TurningBandsSimulator(),DirectBlockSimulator()]:
            with self.assertRaises(ValueError):
                args=([[0,0,0]],) if isinstance(simulator,TurningBandsSimulator) else ([[0,0,0]],[1,1,1])
                simulator.simulate(*args,sample_xyz=[[0,0,0],[0,0,0]],sample_values=[1,2])
        with self.assertRaises(ValueError):DirectBlockSimulator(max_blocks=1).simulate([[0,0,0],[1,0,0]],[1,1,1])
    def test_public_entry_points(self):
        x=[[0,0,0]];y=[1.];q=[[.2,0,0]]
        self.assertEqual(simulate_tb(x,y,q,n_realizations=3).shape,(1,3))
        self.assertEqual(simulate_dbs(x,y,q,[.2,.2,.2],n_realizations=3).shape,(1,3))
    def test_empty_targets_and_repeatability(self):
        tb=TurningBandsSimulator(bands=16);db=DirectBlockSimulator()
        self.assertEqual(tb.simulate([],n_realizations=3).shape,(0,3))
        self.assertEqual(db.simulate([],[1,1,1],n_realizations=3).shape,(0,3))
        np.testing.assert_array_equal(db.simulate([[0,0,0]],[1,1,1],seed=9),db.simulate([[0,0,0]],[1,1,1],seed=9))
if __name__=='__main__':unittest.main(verbosity=2)
