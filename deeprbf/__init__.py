# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""DeepRBF native engine with explicit Python geological modelling backends."""
from . import _runtime
from deeprbf_recovered import *
from deeprbf_recovered import __version__ as __native_version__
__version__ = '3.5.0'
if __native_version__ != __version__:
    raise ImportError(f'DeepRBF Python {__version__} requires native {__version__}; loaded {__native_version__}. Select the matching build or install the matching wheel.')
import deeprbf_core as _core
def runtime_status():
    return dict(package="deepfusion-deeprbf", version=__version__, native_core_available=True,
                python_fallback_available=False, native_core=_core.runtime_status(),
                lithology_backend='native_cpp_sparse_ik_sis; hybrid_native_gibbs_pgs',
                snesim_backend='python_advanced; native_cpp_indexed_static')

from .normal_score import NormalScoreTransform, run_sgs_normal_score

from .simulation import GaussianCovariance, TurningBandsSimulator, DirectBlockSimulator, BlockMoments, simulate_tb, simulate_dbs
from .lithology import LithologyResult, estimate_lithology_ik, simulate_lithology_sis, simulate_plurigaussian
from .snesim import SNESIMConfig, SNESIMMineralSimulator, simulate_snesim
from .resource import LocalEstimator
from .desurvey import minimum_curvature
from .io import export_leapfrog_csv
from .categorical import IndicatorCalibration, fit_indicator_models, estimate_full_indicator
from .training_images import SNESIMAtlasSimulator
from .geological_priors import GEOLOGICAL_PROFILES, conceptual_training_image
from .domain_rbf import DomainRBF
from .ensemble import simulate_sis_ensemble, simulate_pgs_ensemble, simulate_snesim_ensemble
from .grade import GradeEstimator
from .sparse_pgs import simulate_plurigaussian_sparse
from .recoverable import RecoverableMoments, recoverable_moments, aggregate_joint_realizations
from .contact_rbf import ContactRBF, BalancedContactRBF
from .paired_contact_surface import PairedContactSurface
