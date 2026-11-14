# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Process-local Conda DLL setup, before importing NumPy/SciPy/native modules.

Conda BLAS/LAPACK dispatch shims can use LoadLibrary internally. Adding a
Python DLL directory alone does not configure those PATH-based lookups.
"""
import os
from pathlib import Path
import sys

_handles=[]

def configure():
    if os.name!='nt':return
    prefix=Path(sys.prefix)
    directories=[prefix,prefix/'Library/bin',prefix/'DLLs']
    existing=[str(p) for p in directories if p.is_dir()]
    os.environ['PATH']=os.pathsep.join(existing+[os.environ.get('PATH','')])
    for directory in existing:
        _handles.append(os.add_dll_directory(directory))
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','1')

configure()
