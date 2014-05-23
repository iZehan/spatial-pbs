"""
Created on 21 Mar 2012

@author: zw606
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("balltree", ["balltree.pyx"],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         libraries=['m'])]

setup(
    name='BallTree',
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules, requires=['numpy']
)
