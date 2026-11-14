# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
import numpy as np
import pytest
from deeprbf.snesim import SNESIMConfig,simulate_snesim,build_template_offsets


def test_rotated_long_axis_is_not_clipped_to_unrotated_box():
    offsets=build_template_offsets(3,(6.,1.,1.),128,(90.,0.,0.))
    assert any(np.array_equal(p,[0,6,0]) for p in offsets)
    assert any(np.array_equal(p,[0,-6,0]) for p in offsets)


@pytest.mark.parametrize('kwargs',[dict(n_realizations=0),dict(hard_data=np.zeros((3,3),float)),dict(soft_probabilities=np.full((3,3,2),np.nan))])
def test_invalid_python_mps_inputs_do_not_silently_change_meaning(kwargs):
    ti=np.indices((7,7)).sum(0)%2
    with pytest.raises(ValueError):simulate_snesim(ti,(3,3),config=SNESIMConfig(n_multigrids=1,search_radii=(1,1,0)),**kwargs)


def test_fractional_training_image_rejected():
    with pytest.raises(ValueError):simulate_snesim(np.array([[0.,1.5],[1.5,0.]]),(2,2))
