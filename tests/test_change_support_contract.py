# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Independent arithmetic moments and support-contract regression checks."""
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
import deeprbf_recovered as native


def block():
    b=native.Block()
    b.xmin=0.; b.xmax=10.; b.ymin=0.; b.ymax=0.; b.zmin=0.; b.zmax=0.
    b.discretization=[2,1,1]
    return b


def test_lognormal_preserves_arithmetic_moments():
    result=native.lognormal_correction(2.,4.,.25)
    assert result.block_mean == 2.
    assert result.block_variance == 1.
    b=block(); vrf=native.compute_variance_reduction_factor(b,10.,1.,0.)
    nodes,weights=hermgauss(60); weights/=np.sqrt(np.pi)
    sigma2=np.log(2.); mu=np.log(2.)-sigma2/2
    point=np.exp(mu+np.sqrt(2*sigma2)*nodes)
    transformed=np.asarray(native.apply_change_of_support_to_realizations(
        point[:,None].tolist(),2.,4.,b,10.,1.,0.,'lognormal'))
    assert weights@transformed == pytest.approx(2.,abs=1e-10)
    assert weights@(transformed-2.)**2 == pytest.approx(4*vrf,abs=1e-9)


def test_midpoints_and_spatial_averages_do_not_shrink_twice():
    b=block()
    # Separation is 5 m: spherical correlation = 1 - 1.5*.5 + .5*.5^3.
    expected=(2+2*(1-1.5*.5+.5*.5**3))/4
    assert native.compute_variance_reduction_factor(b,10.,1.,0.) == pytest.approx(expected)
    assert native.apply_change_of_support_to_realizations([[1.,5.],[2.,8.]],2.,4.,b,10.,1.,0.,'average') == [3.,5.]


@pytest.mark.parametrize('rows',[[[]],[[1.],[2.,3.]],[[float('nan')]],[[0.]]])
def test_invalid_lognormal_realizations_rejected(rows):
    with pytest.raises(ValueError):
        native.apply_change_of_support_to_realizations(rows,2.,4.,block(),10.,1.,0.,'lognormal')


@pytest.mark.parametrize('mean,variance,vrf',[(0.,1.,.5),(2.,-1.,.5),(2.,1.,-1.),(2.,1.,1.1),(2.,float('nan'),.5)])
def test_invalid_moments_rejected(mean,variance,vrf):
    with pytest.raises(ValueError): native.lognormal_correction(mean,variance,vrf)


def test_unbounded_quadrature_rejected():
    b=block(); b.discretization=[1000000,1000000,1000000]
    with pytest.raises(ValueError): native.compute_variance_reduction_factor(b,10.,1.,0.)
