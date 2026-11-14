# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import unittest
import numpy as np
from deeprbf.categorical import IndicatorCalibration,fit_indicator_models,estimate_full_indicator


class FullIndicatorTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0]],float)
        self.y=np.array([10,20,10,20]);self.h=np.array(['a','b','c','d'])
        self.model=IndicatorCalibration(np.array([10,20]),np.array([.5,.5]),
            np.array([[10,10,10],[30,30,30]]),np.eye(3),{})

    def test_independent_single_neighbor_formula(self):
        result=estimate_full_indicator(self.x[:1],self.y[:1],self.h[:1],[[2,0,0]],self.model,
            min_holes=1,max_neighbors=1,search_ranges=(5,5,5))
        raw=np.array([.5+.5*np.exp(-.6)/(1+1e-8),.5-.5*np.exp(-.2)/(1+1e-8)])
        np.testing.assert_allclose(result.probabilities[0],raw/raw.sum(),rtol=1e-10)

    def test_support_and_exact_data(self):
        result=estimate_full_indicator(self.x,self.y,self.h,np.vstack([self.x,[[1000,0,0],[5,5,0]]]),self.model,
                                       search_ranges=(20,20,20))
        np.testing.assert_array_equal(result.modal[:4],self.y)
        self.assertEqual(result.diagnostics['support'][4]['status'],'UNSUPPORTED')
        np.testing.assert_array_equal(result.probabilities[4],[.5,.5])
        self.assertEqual(result.diagnostics['support'][5]['holes'],4)
        np.testing.assert_allclose(result.probabilities.sum(1),1.)

    def test_quotas_and_conflicts(self):
        h=np.array(['a','a','a','b'])
        result=estimate_full_indicator(self.x,self.y,h,[[5,5,0]],self.model,max_per_hole=1,
                                       search_ranges=(20,20,20))
        self.assertEqual(result.diagnostics['support'][0]['samples'],2)
        self.assertEqual(result.diagnostics['support'][0]['status'],'UNSUPPORTED')
        with self.assertRaises(ValueError):
            estimate_full_indicator(np.zeros((2,3)),[10,20],['a','b'],[[0,0,0]],self.model)

    def test_hole_balancing_reproducibility_and_weak_fit_flag(self):
        x=np.array([[0,0,i] for i in range(9)]+[[10,0,0]],float)
        y=np.array([10]*9+[20]);holes=np.array(['a']*9+['b'])
        a=fit_indicator_models(x,y,holes,max_pairs=200,min_positive_pairs=1000)
        b=fit_indicator_models(x,y,holes,max_pairs=200,min_positive_pairs=1000)
        np.testing.assert_allclose(a.prior,[.5,.5])
        np.testing.assert_array_equal(a.ranges,b.ranges)
        self.assertTrue(all(r['status']=='REVIEW_INSUFFICIENT_PAIRS' for r in a.diagnostics['categories']))

    def test_pattern_query_matches_independent_brute_force(self):
        from deeprbf.snesim import PatternDatabase
        rng=np.random.default_rng(812)
        db=PatternDatabase(rng.integers(0,3,(9,10,11)),np.arange(3),
                           np.array([[0,0,1],[0,1,0],[1,0,0],[0,0,-1]]))
        for _ in range(100):
            pos=rng.permutation(4);values=rng.integers(0,3,4);minimum=int(rng.integers(1,20))
            expected=db.prior.copy();rep=len(db.patterns);dropped=len(pos)
            for keep in range(4,0,-1):
                mask=np.all(db.patterns[:,pos[:keep]]==values[:keep],axis=1)
                if mask.sum()>=minimum:
                    expected=np.bincount(db.centers[mask],minlength=3)/mask.sum()
                    rep=int(mask.sum());dropped=4-keep;break
            p,n,b=db.query(pos,values,minimum)
            np.testing.assert_allclose(p,expected);self.assertEqual((n,b),(rep,dropped))
            p[:]=0 # Callers cannot poison a cached distribution.
            again,_,_=db.query(pos,values,minimum)
            np.testing.assert_allclose(again,expected)


if __name__=='__main__':unittest.main()
