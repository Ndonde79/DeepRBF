# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import os, sys
_dll_handle = os.add_dll_directory(os.path.join(sys.prefix, "Library", "bin")) if os.name=="nt" else None
import numpy as np
import deeprbf_recovered as d

class NumericalRegression(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(123).uniform(-1,1,(24,3))
        self.y=np.sin(self.x[:,0])+self.x[:,1]**2-self.x[:,2]
        self.q=np.random.default_rng(44).uniform(-1,1,(12,3))
        d.NativeExactRBFImplicitModel.force_iterative_solver(False)
    def tearDown(self):
        d.NativeExactRBFImplicitModel.force_iterative_solver(False)
    def reference_rbf(self, kernel='gaussian', weights=None, reg=1e-6):
        def phi(r):
            return {'gaussian':lambda:np.exp(-(r/1.3)**2),'linear':lambda:r,'cubic':lambda:r**3,'multiquadric':lambda:np.hypot(r,1.3),'inverse_multiquadric':lambda:1/np.hypot(r,1.3)}[kernel]()
        n=len(self.x);P=np.c_[np.ones(n),self.x]
        K=phi(np.linalg.norm(self.x[:,None]-self.x,axis=2))
        K+=np.diag(np.full(n,reg) if weights is None else reg/weights**2)
        A=np.block([[K,P],[P.T,np.zeros((4,4))]])
        c=np.linalg.solve(A,np.r_[self.y,np.zeros(4)])
        return phi(np.linalg.norm(self.q[:,None]-self.x,axis=2))@c[:n]+np.c_[np.ones(len(self.q)),self.q]@c[n:]
    def test_dense_rbf_against_numpy(self):
        for kernel in ['gaussian','linear','cubic','multiquadric','inverse_multiquadric']:
            with self.subTest(kernel=kernel):
                m=d.NativeExactRBFImplicitModel(kernel,1.3,1e-6);m.fit(self.x,self.y)
                a=m.predict(self.q); b=self.reference_rbf(kernel); np.testing.assert_allclose(a,b,rtol=2e-7,atol=2e-7)
                sb=m.solver_backend().lower()
                self.assertTrue(any(k in sb for k in ('mkl','fallback_dense','eigen_full_pivot_lu','eigen_partial_pivot_lu')), f'unexpected solver backend: {sb}')
    def test_iterative_matches_dense(self):
        m=d.NativeExactRBFImplicitModel('gaussian',1.3,1e-6)
        d.NativeExactRBFImplicitModel.force_iterative_solver(True)
        m.fit(self.x,self.y)
        np.testing.assert_allclose(m.predict(self.q),self.reference_rbf(),rtol=2e-4,atol=2e-4)
    def test_weighted_trend(self):
        w=np.linspace(.5,2,len(self.x));m=d.NativeTrendImplicitModel('gaussian',1.3,.02)
        m.fit_signed_constraints(self.x,self.y,w)
        np.testing.assert_allclose(m.predict(self.q),self.reference_rbf(weights=w,reg=.02),rtol=1e-9,atol=1e-9)
        m.clear_trend_transform();self.assertFalse(m.is_fitted())
    def test_kriging_reference_and_nested(self):
        x=self.x[:8];v=self.y[:8];q=np.array([[4.,3.,2.]])
        def cov(r):
            h=r/2.;return np.where(r==0,1.,.8*np.where(h<1,1-1.5*h+.5*h**3,0))
        C=cov(np.linalg.norm(x[:,None]-x,axis=2));c=cov(np.linalg.norm(x-q,axis=1))
        a=np.block([[C+np.eye(8)*1e-10,np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
        sol=np.linalg.solve(a,np.r_[c,1]);mean=sol[:8]@v;var=1-sol[:8]@c-sol[8]
        got=d.ordinary_kriging_with_range(x,v,q,8,1,100,2,1,.2)
        np.testing.assert_allclose(got['estimate'],[mean],atol=1e-7)
        np.testing.assert_allclose(got['variance'],[var],atol=1e-7)
        self.assertGreater(got['variance'][0],1.)
        vg=d.NestedVariogram(.2);s=d.VariogramStructure();s.type=d.VariogramModelType.Spherical;s.range=2.;s.contribution=.8;vg.add_structure(s)
        nested=d.ordinary_kriging_nested(x,v,q,vg,8,1,100)
        np.testing.assert_allclose(nested['estimate'],got['estimate'],atol=1e-7)
        np.testing.assert_allclose(nested['variance'],got['variance'],atol=1e-7)
    def test_nugget_covariance(self):
        self.assertEqual(d.spherical_covariance(0,1,.2),1)
        self.assertEqual(d.spherical_covariance(2,1,.2),0)
        self.assertAlmostEqual(d.spherical_covariance(.5,1,.2),.8*(1-.75+.0625))
    def test_invalid_inputs(self):
        for args in [([],[],[[0,0,0]],0,1,2,1,1,.1),([[0,0,0]],[],[[0,0,0]],2,1,2,1,1,.1)]:
            with self.assertRaises((ValueError,RuntimeError)):d.ordinary_kriging_with_range(*args)
        with self.assertRaises((ValueError,RuntimeError)):d.fit_nested_variogram([1,2],[.2,.4],max_structures=2)
        with self.assertRaises((ValueError,RuntimeError,TypeError)):
            d.snesim_native_static(np.array([9],dtype=np.int32),
                                   np.array([d.SNESIM_UNKNOWN],dtype=np.int32),
                                   categories=[0,1], target_proportions=[.5,.5])
    def test_snesim_reproducibility_and_hard_data(self):
        ti=np.array([i%2 for i in range(50)],dtype=np.int32).reshape(1,50)
        cfg=d.SnesimNativeConfig()
        cfg.radius_x=3.0; cfg.radius_y=0.0; cfg.radius_z=0.0
        cfg.max_conditioning_nodes=4; cfg.min_replicates=1
        cfg.n_multigrids=1; cfg.seed=42
        hard=np.full((1,16),d.SNESIM_UNKNOWN,dtype=np.int32)
        a=d.snesim_native_static(ti,hard,categories=[0,1],config=cfg)['values']
        b=d.snesim_native_static(ti,hard,categories=[0,1],config=cfg)['values']
        self.assertTrue(np.array_equal(a,b))
        self.assertTrue(set(np.unique(a)).issubset({0,1}))
    def test_cokriging_categories_and_variance(self):
        lmc=d.LinearModelOfCoregionalization();s=d.LMCStructure()
        s.range_major=s.range_minor=s.range_vertical=2.;s.covariance_matrix=[[.25,-.25],[-.25,.25]];lmc.add_structure(s)
        x=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]];cat=[10,20,10,20]
        got=d.ordinary_cokriging(x,cat,x,[10,20],lmc,4,1,10)
        est=np.array(got['estimates']).reshape(-1,2)
        np.testing.assert_allclose(est,[[1,0],[0,1],[1,0],[0,1]],atol=1e-6)
        self.assertTrue(np.isfinite(got['variances']).all())
        bad=d.LinearModelOfCoregionalization();t=d.LMCStructure();t.covariance_matrix=[[1,-.9,-.9],[-.9,1,-.9],[-.9,-.9,1]];bad.add_structure(t)
        self.assertFalse(d.is_positive_definite(bad))
    def test_geological_signed_plane(self):
        m=d.NativeGeologicalImplicitModel('gaussian',1.,1e-6); m.set_use_polynomial_drift(True)
        m.fit_constraints([[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,1,1]],[[0,0,-1],[1,1,-1]])
        np.testing.assert_allclose(m.predict([[.2,.3,0],[.2,.3,.5],[.2,.3,-.5]]),[0,.5,-.5],atol=2e-5)
    def test_variogram_fit_matches_generated_curve(self):
        h=np.linspace(.1,5,50);gamma=.1+.9*(1-np.exp(-h/1.2))
        vg=d.fit_nested_variogram(h,gamma,max_structures=1)
        recovered=np.array([vg.sill()-vg.covariance(float(v)) for v in h])
        self.assertLess(np.sqrt(np.mean((recovered-gamma)**2)),.025)
if __name__=='__main__':unittest.main(verbosity=2)
