"""Build a platform wheel containing the same CMake native engine and wrappers."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11
        root=Path(__file__).resolve().parent
        build=(root/self.build_temp/'cmake').resolve()
        cmake=os.environ.get('CMAKE_COMMAND') or shutil.which('cmake')
        if not cmake:
            bundled=Path(sys.prefix)/'Lib/site-packages/cmake/data/bin/cmake.exe'
            cmake=str(bundled) if bundled.exists() else None
        if not cmake: raise RuntimeError('CMake is required to build the native wheel')
        eigen=os.environ.get('EIGEN3_ROOT',str(Path(sys.prefix)/'Library/include/eigen3'))
        args=[cmake,'-S',str(root),'-B',str(build),
              '-DPython3_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+pybind11.get_cmake_dir(),
              '-DDEEPRBF_EIGEN_ROOT='+eigen,'-DDEEPRBF_BUILD_TESTS=OFF',
              '-DDEEPRBF_PORTABLE_TESTS=OFF','-DDEEPRBF_ENABLE_MKL=OFF','-DDEEPRBF_ENABLE_SCALFMM=OFF']
        if os.name=='nt': args+=['-G',os.environ.get('CMAKE_GENERATOR','Visual Studio 18 2026'),'-A','x64']
        subprocess.run(args,check=True)
        subprocess.run([cmake,'--build',str(build),'--config','Release','--target','deeprbf_recovered','--parallel','2'],check=True)
        candidates=list((build/'module').glob('deeprbf_recovered*'))
        candidates=[p for p in candidates if p.suffix in ('.pyd','.so')]
        if len(candidates)!=1: raise RuntimeError('Expected one built native extension')
        destination=Path(self.get_ext_fullpath(ext.name));destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(candidates[0],destination)

setup(packages=find_packages(),py_modules=['deeprbf_core'],
      ext_modules=[Extension('deeprbf_recovered',sources=[])],cmdclass={'build_ext':CMakeBuild})
