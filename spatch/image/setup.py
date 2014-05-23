"""
Created on 04/09/13

@author: zw606
"""
#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("gdt", ["gdt.pyx"])]

setup(
    name='gdt',
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
    requires=['numpy']
)
__author__ = 'Ze'
