from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):

    _build_ext.user_options.append(
        ('enable-openmp', None,
         "Enable OpenMP during compilation")
    )
    _build_ext.boolean_options.append('enable-openmp')

    # def __init__(self, dist):
    #     super(_build_ext, self).__init__(dist)

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.enable_openmp = None

    def finalize_options(self):
        _build_ext.finalize_options(self)

    def build_extension(self, ext):
        if self.enable_openmp:
            ext.extra_compile_args.append('-fopenmp')
            ext.extra_link_args.append('-fopenmp')
        _build_ext.build_extension(self, ext)

#debug=True
#debug=False
#extra_compile_args = ['-O0', '-g']
#extra_link_args = ['-g']
extra_compile_args = ['-O3']
extra_link_args = []

import numpy
include_dirs = [numpy.get_include()]

extensions = [Extension('c_image_weak_learner', ['c_image_weak_learner.pyx'], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, include_dirs=include_dirs)]

setup(
	cmdclass={'build_ext': build_ext},
    name = 'AITDistributedRandomForest',
    #ext_modules = cythonize('c_image_weak_learner.pyx', gdb_debug=True),
    #ext_modules = cythonize(extensions, gdb_debug=debug),
    ext_modules = cythonize(extensions),
)
