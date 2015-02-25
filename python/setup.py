from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'AITDistributedRandomForest',
    ext_modules = cythonize('c_image_training_context.pyx'),
)
