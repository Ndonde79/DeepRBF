# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Run scientific simulation tests without importing optional native bindings."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deeprbf_simulation_standalone', ROOT / 'deeprbf' / 'simulation.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
source = (ROOT / 'tests' / 'test_simulation_methods.py').read_text(encoding='utf-8')
source = source.replace('from deeprbf import GaussianCovariance, TurningBandsSimulator, DirectBlockSimulator, simulate_tb, simulate_dbs', 'from deeprbf_simulation_standalone import GaussianCovariance, TurningBandsSimulator, DirectBlockSimulator, simulate_tb, simulate_dbs')
namespace = {'__name__': 'deeprbf_scientific_tests', '__file__': str(ROOT / 'tests' / 'test_simulation_methods.py')}
exec(compile(source, namespace['__file__'], 'exec'), namespace)
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(namespace['SimulationMethods'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
