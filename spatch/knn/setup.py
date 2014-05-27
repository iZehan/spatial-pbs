"""
Created on 21 Mar 2012

@author: Zehan Wang
"""
from distutils.core import setup
from distutils.extension import Extension
import os
from Cython.Distutils import build_ext
import numpy
from numpy.distutils.misc_util import Configuration

ext_modules = [Extension("balltree", ["spatch/knn/balltree.pyx"],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         libraries=['m'])]

setup(
    name='balltree',
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules, requires=['numpy']
)


# def configuration(parent_package="", top_path=None):
#     config = Configuration("knn", parent_package, top_path)
#     libraries = []
#     if os.name == 'posix':
#         libraries.append('m')
#     config.add_extension("balltree",
#                          sources=["balltree.c"],
#                          include_dirs=[numpy.get_include()],
#                          libraries=libraries,
#                          extra_compile_args=["-O3", "-fopenmp"],
#                          extra_link_args=['-fopenmp'])
#
#     return config
#
# if __name__ == "__main__":
#     from numpy.distutils.core import setup
#     setup(**configuration().todict())
