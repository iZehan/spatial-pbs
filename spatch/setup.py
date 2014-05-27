"""
Created on 21 May 2014

@author: Zehan Wang
"""
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('spatch', parent_package, top_path)

    config.add_subpackage("image")
    config.add_subpackage("knn")
    config.add_subpackage("segmentation")
    config.add_subpackage("utilities")

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())