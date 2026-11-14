# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pandas as pd
from deeprbf import CovarianceSpec, NeighborhoodSpec
from deeprbf.grade import GradeEstimator
from deeprbf.grade_benchmark import compare_idw2, paired_grade_scores
from deeprbf.sample_reconciliation import unique_sample_rows


def test_idw_respects_target_domain_and_whole_hole_quota():
    x = np.column_stack(([1,2,3,.25],np.zeros(4),np.zeros(4)))
    c = CovarianceSpec()
    c.family = 'exponential'
    c.ranges = [10]*3
    s = NeighborhoodSpec()
    s.radii = [10]*3
    s.min_neighbors = s.min_holes = 2
    s.max_per_hole = 1
    s.restrict_domain = True
    m = GradeEstimator(x,[1,100,10,1000],['H1','H1','H2','H3'],[1,1,1,2],covariance=c,search=s)
    result, ref = compare_idw2(m,[[0,0,0],[0,0,0]],[1,2])
    np.testing.assert_allclose(ref[0],1.9)
    assert np.isnan(ref[1]) and not result['supported'][1]


def test_paired_scoring_does_not_hide_unsupported_targets():
    g = dict(coverage=.7,scored=1,abs_relative_bias=.1,rmse_ratio=.95)
    out = paired_grade_scores([1,2,3],[1,np.nan,3],[1,2,3],[1,2,1],2,g)
    assert out['idw2']['pass_all']
    assert out['paired_idw2']['coverage']==.5 and not out['paired_idw2']['pass_all']


def test_sample_id_aliases_quarantine_ambiguous_keys():
    samples = pd.DataFrame(dict(hole_id=[' h1 ','H1','H2','H2','H3','H3'],
                                sample_id=['17.0','17','0017','17','',None]))
    result = unique_sample_rows(samples)
    assert result.sample_id.tolist()==['0017','17']
