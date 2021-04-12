from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("arm_cython.pyx")
)

# NOTE: Run `python setup.py build_ext --inplace`