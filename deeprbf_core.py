# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Compatibility import name for the recovered native extension. New historical API."""
from deeprbf_recovered import *
import deeprbf_recovered as _native
__version__ = _native.__version__
def runtime_status():
    return dict(name="deeprbf_core", engine="DeepRBF recovered C++", version=__version__,
                compiled_native=True, backend="recovered_augmented_rbf_cpp",
                native_module=_native.__file__, mkl=_native.__has_mkl__,
                openmp=_native.__has_openmp__, scalfmm=_native.__has_scalfmm__,
                gaussian_convention="exp(-(r/epsilon)^2)")
