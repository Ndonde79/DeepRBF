# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Independent numerical, support and native/Python parity acceptance tests."""
import unittest
import numpy as np
import deeprbf as d
from deeprbf.domain_rbf import DomainRBF

class AuditedEngineTests(unittest.TestCase):
    def test_finite_interval_observations_match_independent_covariance_integrals(self):
        x=np.array([[0.,0,0],[5,0,0],[0,5,0],[5,5,1]])
        values=np.array([1.,3.,2.,8.]);c,s=self.controls();c.family='gaussian';c.ranges=[10,15,5]
        observations=[self.target(point,(0,0,2),(1,1,8)) for point in x]
        target=self.target([2,2,.5],(2,2,2),(3,3,3))
        obs=[point+np.c_[np.zeros(8),np.zeros(8),(np.arange(8)+.5)/8*2-1] for point in x]
        grid=np.array(np.meshgrid(*[np.arange(3)]*3,indexing='ij')).reshape(3,-1).T
        block=np.array(target.center)+(grid+.5)/3*2-1
        def integral(a,b):return np.exp(-np.sum(((a[:,None]-b[None,:])/np.array(c.ranges))**2,axis=2)).mean()
        matrix=np.ones((5,5));matrix[4,4]=0
        for i in range(4):
            for j in range(4):matrix[i,j]=integral(obs[i],obs[j])
        rhs=np.r_[[integral(points,block) for points in obs],1.]
        solution=np.linalg.solve(matrix,rhs)
        result=d.audited_ordinary_kriging(x,values,[0,1,2,3],[0]*4,[target],c,s,observation_support=observations)
        self.assertTrue(result.diagnostics[0].success)
        self.assertAlmostEqual(result.estimate[0],solution[:4]@values,places=9)
        self.assertAlmostEqual(result.variance[0],integral(block,block)-solution@rhs,places=10)
        c.nugget=.1
        with self.assertRaises(ValueError):d.audited_ordinary_kriging(x,values,[0,1,2,3],[0]*4,[target],c,s,observation_support=observations)

    def test_explicit_practical_range_convention(self):
        c=d.CovarianceSpec();c.ranges=[500.,300.,150.]
        self.assertEqual(c.range_convention,'length_scale')
        for family in ('exponential','gaussian'):
            c.family=family;c.range_convention='practical'
            for axis in range(3):
                lag=np.zeros(3);lag[axis]=c.ranges[axis]
                self.assertAlmostEqual(c.covariance([0,0,0],lag),np.exp(-3),places=13)
        c.range_convention='unknown'
        with self.assertRaises(ValueError):c.validate()

    def test_legacy_block_midpoint_and_invalid_extent(self):
        import deeprbf_recovered as native
        x=[[0.,0,0],[4.,0,0],[0.,4,0],[0.,0,4]];y=[0.,1,2,3]
        b=native.Block();b.xmin=0;b.xmax=2;b.ymin=0;b.ymax=2;b.zmin=0;b.zmax=2;b.discretization=[1,1,1]
        block=native.block_kriging(x,y,[b],4,1,100.,10.,1.,0.)
        point=native.ordinary_kriging_with_range(x,y,[[1.,1,1]],4,1,100.,10.,1.,0.)
        np.testing.assert_allclose(block['estimate'],point['estimate'],atol=1e-8)
        np.testing.assert_allclose(block['variance'],point['variance'],atol=1e-8)
        b.xmin=3
        with self.assertRaises(ValueError):native.block_kriging(x,y,[b])

    def test_equidistant_candidate_boundary_uses_input_index(self):
        x=np.array([[1.,0,0]]*8+[[-1.,0,0]]*8)
        model=DomainRBF(np.array([0,1]),np.array([0,1]),ranges=[1.,1.,1.],max_neighbors=3,candidate_neighbors=4,max_per_hole=1,min_holes=1)
        model.fit(x,np.array([0]*8+[1]*8),np.arange(16))
        p,flags=model.predict([[0.,0,0]])
        n,nflags=model.predict_native([[0.,0,0]])
        np.testing.assert_allclose(p,n,atol=1e-12)
        self.assertGreater(p[0,0],.99)
        np.testing.assert_array_equal(flags['support_holes'],nflags['support_holes'])

    def controls(self):
        c=d.CovarianceSpec();c.family='exponential';c.ranges=[10,20,30]
        s=d.NeighborhoodSpec();s.radii=[100,100,100];s.min_holes=1;s.min_neighbors=1;s.max_per_hole=0
        return c,s

    def target(self,center,size=(0,0,0),discretization=(1,1,1),axes=None):
        b=d.RotatedSupport();b.center=center;b.size=size;b.discretization=discretization
        if axes is not None:b.axes=axes
        return b

    def test_covariance_formulas_and_observation_nugget(self):
        c,_=self.controls();c.sill=2.;c.nugget=.3
        for family,formula in [('spherical',lambda h:max(0.,1-1.5*h+.5*h**3) if h<1 else 0.),('exponential',lambda h:np.exp(-h)),('gaussian',lambda h:np.exp(-h*h))]:
            c.family=family
            for h in [0,.1,.8,1,3]:self.assertAlmostEqual(c.covariance([0,0,0],[10*h,0,0]),1.7*formula(h),places=12)
            self.assertAlmostEqual(c.covariance([0,0,0],[0,0,0],True),2.)

    def test_independent_augmented_system_and_sill_scaling(self):
        x=np.array([[0,0,0],[4,0,0],[0,4,0],[0,0,4]],float);y=np.array([1,2,-1,3.]);q=np.array([1,1,1.]);c,s=self.controls()
        delta=x[:,None,:]-x[None,:,:];a=np.ones((5,5));a[:4,:4]=np.exp(-np.linalg.norm(delta/np.array(c.ranges),axis=2));a[4,4]=0
        rhs=np.r_[np.exp(-np.linalg.norm((x-q)/np.array(c.ranges),axis=1)),1.];w=np.linalg.solve(a,rhs)
        for sill in [1e-15,1.,1e15]:
            c.sill=sill;r=d.audited_ordinary_kriging(x,y,[0,1,2,3],[0]*4,[self.target(q)],c,s)
            self.assertTrue(r.diagnostics[0].success,r.diagnostics[0].status)
            self.assertAlmostEqual(r.estimate[0],w[:4]@y,places=10)
            self.assertAlmostEqual(r.variance[0]/sill,1-w@rhs,places=10)
            self.assertLess(r.diagnostics[0].relative_residual,1e-10)

    def test_rank_failure_and_regularization_disclosure(self):
        c,s=self.controls();x=[[0,0,0],[0,0,0]];q=[self.target([1,0,0])]
        r=d.audited_ordinary_kriging(x,[1,1],[0,1],[0,0],q,c,s)
        self.assertEqual(r.diagnostics[0].status,'RANK_DEFICIENT');self.assertTrue(np.isnan(r.estimate[0]))
        r=d.audited_ordinary_kriging(x,[1,1],[0,1],[0,0],q,c,s,regularization=1e-6)
        self.assertTrue(r.diagnostics[0].success);self.assertEqual(r.diagnostics[0].regularization,1e-6)
        with self.assertRaises(ValueError):d.audited_ordinary_kriging(x,[1,2],[0,1],[0,0],q,c,s)

    def test_domain_filter_octants_hole_quota_and_minimum_support(self):
        _,s=self.controls();s.restrict_domain=True;s.max_per_hole=1;s.min_holes=3;s.max_neighbors=3;s.candidate_neighbors=3
        x=[[.1,0,0],[.2,0,0],[1,0,0],[-1,0,0],[0,1,0]]
        r=d.select_neighborhood(x,[0,0,1,2,3],[9,9,1,1,1],[0,0,0],1,s)
        self.assertTrue(r.supported);self.assertEqual(r.n_holes,3);self.assertEqual(set(r.indices),{2,3,4})
        s.max_per_octant=1;s.restrict_domain=False;s.max_neighbors=8;s.candidate_neighbors=10;s.min_holes=1
        r=d.select_neighborhood(x,list(range(5)),[0]*5,[0,0,0],0,s)
        self.assertEqual(len(r.indices),2)
        s.min_holes=4;r=d.select_neighborhood(x,list(range(5)),[0]*5,[0,0,0],0,s);self.assertFalse(r.supported)

    def test_rotated_blocks_match_independent_midpoint_quadrature(self):
        c,s=self.controls();x=np.array([[-8,0,0],[8,0,0],[0,9,0],[0,0,9]],float);y=np.array([1,2,3,4]);q=np.array([0.,0.,0.])
        theta=.6;axes=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
        block=self.target(q,[4,2,2],[4,4,4],axes)
        r=d.audited_ordinary_kriging(x,y,[0,1,2,3],[0]*4,[block],c,s)
        u=(np.indices((4,4,4)).reshape(3,-1).T+.5)/4-.5;points=(u*np.array([4,2,2]))@axes+q
        ranges=np.array(c.ranges);cov=lambda a,b:np.exp(-np.linalg.norm((a[:,None]-b[None,:])/ranges,axis=2))
        a=np.ones((5,5));a[:4,:4]=cov(x,x);a[4,4]=0;rhs=np.r_[cov(x,points).mean(1),1];w=np.linalg.solve(a,rhs)
        self.assertAlmostEqual(r.estimate[0],w[:4]@y,places=10)
        self.assertAlmostEqual(r.variance[0],cov(points,points).mean()-w@rhs,places=10)

    def test_block_discretization_converges_and_point_limit(self):
        c,s=self.controls();x=[[-10,0,0],[10,0,0],[0,10,0],[0,0,10]];y=[1,2,3,4];variances=[]
        for n in [2,4,8]:
            r=d.audited_ordinary_kriging(x,y,list(range(4)),[0]*4,[self.target([0,0,0],[4,4,4],[n,n,n])],c,s)
            self.assertTrue(r.diagnostics[0].success);variances.append(r.variance[0])
        self.assertLess(abs(variances[2]-variances[1]),abs(variances[1]-variances[0]))
        exact=d.audited_ordinary_kriging(x,y,list(range(4)),[0]*4,[self.target(x[0])],c,s)
        self.assertAlmostEqual(exact.estimate[0],1.,places=10);self.assertLess(exact.variance[0],1e-10)

    def test_native_rbf_matches_python_with_gates_calibration_and_quotas(self):
        rng=np.random.default_rng(117);x=rng.uniform(-10,10,(300,3));q=np.r_[rng.uniform(-5,5,(20,3)),x[:3]]
        cats=np.array([10,20,30]);labels=rng.choice(cats,len(x));holes=np.arange(len(x))%20;support=rng.uniform(.2,2,len(x));ranges=np.array([10,8,6.])
        axes=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
        s=d.NeighborhoodSpec();s.radii=(3*ranges).tolist();s.axes=axes
        s.min_neighbors=1;s.min_holes=3;s.max_neighbors=24;s.max_per_hole=3;s.candidate_neighbors=128
        for hierarchical in [False,True]:
            m=DomainRBF(cats,[0,0,1],ranges=ranges,rotation=axes,gate_scale=[1,1,.35]).fit(x,labels,holes,support)
            expected,flags=m.predict(q,hierarchical=hierarchical,prior_mix=.2)
            result=d.categorical_rbf_native(x,labels,holes,support,cats,[0,0,1],q,ranges,axes,[1,1,.35],s,hierarchical,.2)
            np.testing.assert_allclose(np.array(result.probabilities).reshape(len(q),3),expected,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(result.support_holes,flags['support_holes']);np.testing.assert_array_equal(result.supported,flags['supported'])
            native,nd=m.predict_native(q,hierarchical=hierarchical,prior_mix=.2)
            np.testing.assert_allclose(native,expected,atol=1e-10,rtol=0)

    def test_dense_native_gaussian_conditional_distribution_and_duplicates(self):
        c,_=self.controls();x=np.array([[0,0,0],[4,0,0]],float);y=np.array([.2,-.3]);q=np.array([[1,1,0],[2,0,0],[1,1,0],[0,0,0]],float)
        r=d.simulate_gaussian_native(x,y,q,c,12000,914,mean=.1)
        sims=np.array(r.simulations).reshape(len(q),12000)
        covariance=lambda a,b:np.exp(-np.linalg.norm((a[:,None]-b[None,:])/np.array(c.ranges),axis=2))
        dd=covariance(x,x);qd=covariance(q,x);expected=.1+qd@np.linalg.solve(dd,y-.1);cc=covariance(q,q)-qd@np.linalg.solve(dd,qd.T)
        np.testing.assert_allclose(r.conditional_mean,expected,atol=1e-10)
        np.testing.assert_allclose(np.array(r.conditional_covariance).reshape(len(q),len(q)),cc,atol=1e-10)
        np.testing.assert_allclose(sims.mean(1),expected,atol=.018)
        np.testing.assert_allclose(np.cov(sims),cc,atol=.014)
        np.testing.assert_array_equal(sims[0],sims[2]);np.testing.assert_array_equal(sims[3],np.full(12000,y[0]))
        again=d.simulate_gaussian_native(x,y,q,c,12000,914,mean=.1)
        np.testing.assert_array_equal(again.simulations,r.simulations)

    def test_native_gaussian_variogram_and_category_proportions(self):
        from scipy.special import ndtri
        c,_=self.controls();q=np.column_stack([np.arange(20)*2,np.zeros((20,2))])
        r=d.simulate_gaussian_native([],[],q,c,4000,271)
        sims=np.array(r.simulations).reshape(len(q),4000)
        self.assertLess(abs(sims.mean()),.05);self.assertLess(abs(sims.var()-1),.06)
        for lag in [1,3,8]:
            measured=.5*np.mean((sims[lag:]-sims[:-lag])**2);expected=1-np.exp(-2*lag/10)
            self.assertLess(abs(measured-expected),.035)
        # Fixed truncation of a Gaussian field must reproduce known marginal
        # category probabilities; this is not a test of conditional PGS geology.
        proportions=np.bincount(np.digitize(sims.ravel(),ndtri([.2,.7])),minlength=3)/sims.size
        np.testing.assert_allclose(proportions,[.2,.5,.3],atol=.025)

    def test_native_memory_limits_prevent_oversized_dense_jobs(self):
        c,_=self.controls()
        with self.assertRaises(ValueError):d.simulate_gaussian_native([],[],[[0,0,0]]*3000,c,1,1)
        with self.assertRaises(ValueError):d.simulate_gaussian_native([],[],[[0,0,0]],c,16000001,1)

    def test_native_conflicting_coincidences_have_exact_fractions(self):
        _,s=self.controls();r=d.categorical_rbf_native([[0,0,0]]*3,[10,20,20],[0,1,2],[2,1,3],[10,20],[0,1],[[0,0,0]],[1,1,1],np.eye(3),[1,1,1],s,False,.9)
        np.testing.assert_allclose(r.probabilities,[1/3,2/3]);self.assertTrue(r.exact_conditioned[0])

    def test_invalid_axes_ranges_and_quadrature_fail_before_compute(self):
        c,s=self.controls();c.axes=[[1,0,0],[1,0,0],[0,0,1]]
        with self.assertRaises(ValueError):c.validate()
        c,_=self.controls();s.radii=[1,0,1]
        with self.assertRaises(ValueError):s.validate()
        _,s=self.controls()
        with self.assertRaises(ValueError):d.audited_ordinary_kriging([[0,0,0]],[1],[0],[0],[self.target([0,0,0],[1,1,1],[32]*3)],c,s)

if __name__=='__main__':unittest.main(verbosity=2)
