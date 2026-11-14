# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Regression for the observed Windows delay-load crash with an unactivated PATH."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class RuntimeBootstrap(unittest.TestCase):
    def test_clean_path_numpy_scipy_native(self):
        code=r'''
import sys,ctypes,json
from pathlib import Path
ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002)
root=Path(sys.argv[1]);import os;sys.path[:0]=[str(root/'scripts'),os.environ.get('DEEPRBF_MODULE_DIR',str(root/'build-production/module'))]
import runtime_bootstrap
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import deeprbf_recovered as native
A=np.array([[4.,1.],[1.,3.]]);b=np.array([1.,2.])
for i in range(5):
 np.testing.assert_allclose(np.linalg.solve(A,b),[1/11,7/11])
 np.testing.assert_allclose(cho_solve(cho_factor(A),b),[1/11,7/11])
x=np.random.default_rng(123).normal(size=(20,3))
m=native.NativeExactRBFImplicitModel('gaussian',1.,.01);m.fit(x,x[:,0]);np.testing.assert_allclose(m.predict(x),x[:,0],atol=1e-8)
get=ctypes.windll.kernel32.GetModuleHandleW;get.restype=ctypes.c_void_p;get.argtypes=[ctypes.c_wchar_p]
name=ctypes.windll.kernel32.GetModuleFileNameW;name.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong]
loaded={}
for dll in ('libblas.dll','liblapack.dll','mkl_rt.3.dll'):
 h=get(dll)
 if h:
  out=ctypes.create_unicode_buffer(32768);name(h,out,len(out));loaded[dll]=out.value
  assert Path(out.value).resolve().parent==(Path(sys.prefix)/'Library/bin').resolve(),out.value
print(json.dumps(dict(status='PASS',loaded=loaded)))
'''
        env=os.environ.copy()
        env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
        for _ in range(3):
            result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertIn('PASS',result.stdout)
            print(result.stdout,flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
