# DeepRBF

DeepRBF is a C++ engine with a Python interface for implicit geological modelling with radial basis functions
(RBF) and for the geostatistics used inside the resulting shapes.

- **Implicit modelling:** exact RBF interpolation with a linear drift (Gaussian, linear, cubic, thin-plate,
  multiquadric kernels), dense or iterative (GMRES with a two-level domain-decomposition preconditioner) solvers,
  a hierarchical (H-matrix/ACA) product for large fits with long-tailed kernels (3.5.0), contact-based ore shells,
  paired lens surfaces and categorical (domain) probability models.
- **Geostatistics:** variograms and cross-variograms, ordinary and block kriging, co-kriging with a linear model
  of coregionalisation, indicator kriging and sequential indicator simulation, truncated (pluri-)Gaussian
  simulation with Gibbs conditioning, SNESIM multiple-point simulation, change of support.
- **Optional acceleration:** Intel oneMKL for dense solves, ScalFMM 3 (fast multipole method) for Gaussian
  evaluations, OpenMP throughout.
- **SVGP** (`deeprbf.svgp`, optional PyTorch): sparse variational Gaussian processes for ore shells and grades.

Version 3.5.0. The first public source release was 3.4.0. See [CHANGELOG.md](CHANGELOG.md).

## Documentation

- [User Guide](docs/user_guide.md) ([PDF](docs/DeepRBF_User_Guide.pdf)): grade shells and geostatistics from drillhole
  data, for resource geologists.
- [Technical Reference Manual](docs/technical_reference.md) ([PDF](docs/DeepRBF_Technical_Reference.pdf)): kernels,
  solvers, every workflow setting and report field, the native and Python interfaces, building and testing.

## Install

Python 3.11 on 64-bit Windows: download the wheel from the
[latest release](https://github.com/Ndonde79/DeepRBF/releases/latest), check it against `SHA256SUMS`, and install it.

```
certutil -hashfile deepfusion_deeprbf-3.5.0-cp311-cp311-win_amd64.whl SHA256
pip install "deepfusion_deeprbf-3.5.0-cp311-cp311-win_amd64.whl[workflow]"
```

The `workflow` extra adds scikit-image for the grade-shell workflow. The release wheel is portable: OpenMP only,
without oneMKL or ScalFMM. On other platforms, build from source.

## Build from source

Requirements: CMake 3.20+, a C++20 compiler (MSVC 2026 or GCC 11+), Eigen 3.4, pybind11, NumPy.

```
cmake -S . -B build -DDEEPRBF_EIGEN_ROOT=<path to eigen3> -DDEEPRBF_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

Options: `-DDEEPRBF_ENABLE_MKL=ON -DDEEPRBF_MKL_ROOT=<oneMKL>` and
`-DDEEPRBF_ENABLE_SCALFMM=ON -DDEEPRBF_SCALFMM_ROOT=<ScalFMM 3 install>`. `pip wheel . --no-build-isolation`
builds the portable wheel through `setup.py`.

## Quickstart: grade shell from drillholes

`examples/kilamongo` holds a public gold dataset (Kilamongo South, Buhemba, Tanzania: collar, survey and assay CSVs),
three settings files and a reference 0.3 g/t Au indicator shell (Surpac `.str`/`.dtm`).

```
cd examples/kilamongo
python -m deeprbf.workflow.shell kilamongo_indicator.json
```

This composites the holes, fits a spherical-kernel indicator RBF along the trend in the settings file, contours the
0.3 g/t shell and writes Surpac `.str`/`.dtm`, DXF and a JSON report (data honouring, volume, comparison with the
reference shell) to `out/`. `kilamongo_signed_distance.json` and `kilamongo_contacts.json` run the other two methods.
See [examples/kilamongo/README.md](examples/kilamongo/README.md) and
[examples/workflow/README.md](examples/workflow/README.md) for the settings.

## Example: implicit contact model

```python
import numpy as np
import deeprbf_recovered as d

# points: (n, 3) array in a local frame (e.g. metres / 100); values: +1 inside, -1 outside, 0 on contacts
model = d.NativeExactRBFImplicitModel("linear", 1.0, 0.0)    # kernel, epsilon, regularisation
model.fit(points.tolist(), values.tolist())                   # dense up to 1 GB, iterative beyond
field = np.asarray(model.predict(grid.tolist()))             # > 0 inside the modelled unit
print(model.solver_backend(), model.last_predict_backend())
```

Use the `linear` or `cubic` kernel for contacts. The Gaussian kernel suits smooth fields; at a range of tens of
metres it cannot separate contacts logged one to two metres apart down a hole.

## Benchmarks

`benchmarks/` contains the scripts and raw results (JSON lines) behind the timing and accuracy figures in the
software paper. See [benchmarks/README.md](benchmarks/README.md).

## Tests

`ctest` runs the native tests and the numerical regression test. The Python tests in `tests/` run with
`python -m pytest tests` (or `python tests/<file>.py` for the unittest files) once the built module is on
`PYTHONPATH`. `test_svgp.py` needs PyTorch.

## Licence

BSD 3-Clause, see [LICENSE](LICENSE). Third-party components keep their own licences: Eigen (MPL-2.0),
pybind11 (BSD-3-Clause), and, when enabled, ScalFMM (CeCILL-C) and Intel oneMKL (Intel Simplified Software
Licence). The portable wheel contains code compiled from the Eigen and pybind11 headers; it does not contain
ScalFMM or oneMKL.

## How to cite

If you use DeepRBF, please cite the software release (metadata in [CITATION.cff](CITATION.cff); GitHub's
"Cite this repository" button gives APA and BibTeX):

> Ndonde, R. W. (2026). *DeepRBF: implicit geological modelling and geostatistics with radial basis functions*
> (Version 3.5.0) [Computer software]. DeepFusion Geoscience Limited. https://github.com/Ndonde79/DeepRBF

A software paper describing DeepRBF (release 3.4.2) has been submitted to *Computers & Geosciences*.

Models built with DeepRBF are drafts for Competent Person review.
