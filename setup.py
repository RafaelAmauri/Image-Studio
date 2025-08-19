from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "include.effects.dithering.floyd_steinberg",
        ["include/effects/dithering/floyd_steinberg.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)