from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("arm_cython.pyx")
)

# NOTE: To execute successfully, run as `python setup.py build_ext --inplace`