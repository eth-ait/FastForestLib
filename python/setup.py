from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

#debug=True
#debug=False
#extra_compile_args = ['-O0', '-g']
#extra_link_args = ['-g']
extra_compile_args = ['-O3']
extra_link_args = []
extra_compile_args.append('-fopenmp')
extra_link_args.append('-fopenmp')

include_dirs = [numpy.get_include()]

extensions = [Extension('c_image_training_context', ['c_image_training_context.pyx'], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, include_dirs=include_dirs)]

setup(
    name = 'AITDistributedRandomForest',
    #ext_modules = cythonize('c_image_training_context.pyx', gdb_debug=True),
    #ext_modules = cythonize(extensions, gdb_debug=debug),
    ext_modules = cythonize(extensions),
)
