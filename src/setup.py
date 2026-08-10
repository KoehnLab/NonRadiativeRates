# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import sysconfig, pathlib

ext = Extension(
    name="fastfold",
    sources=["fastfold.pyx"],
    include_dirs=[sysconfig.get_paths()["include"]],   # Python headers
    language="c",                     # or "c++" if needed
)

setup(
    ext_modules=cythonize([ext],
                          compiler_directives={"language_level": "3"}),
)

