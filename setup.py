from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="nonradiative_rates._fastfold",
    sources=["src/nonradiative_rates/_fastfold.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
