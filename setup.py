from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "include/effects/floyd_steinberg_dither.pyx",  # relative path to your pyx
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)
