# FastForestLib

A library for training and evaluating random forests.

The folder _cpp_ contains the C++ implementation (multithreaded and distributed variant).

The folder _utils_ contains some MATLAB and python scripts to convert data and forests between different formats.

The folder _python_ contains an older Python implementation using Cython. The testing code could still be useful for forests trained with the C++ code.

The folder _data_ contains some MATLAB scripts to generate synthetic data and to convert MATLAB data format to CSV data format. Check it out for details on the data formats. To some extend it can also be used for testing the code.

## Dependencies
The library uses _boost_, _CImg_ (included), _TCLAP_ (included) and _cereal_ (included).
The distributed code requires _boost-mpi_ for communication.

## Compiling
```
mkdir cpp/build
cd cpp/build
cmake .. # Optionally modify CMake configurations
# On a Linux system you might typically do something like this:
#cmake -DWITH_MPI=TRUE -DWITH_MATLAB=TRUE -DMATLAB_INCLUDE_DIRS=/usr/local/Matlab/R2015a/extern/include/ -DMATLAB_LIB_DIR=/usr/local/Matlab/R2015a/bin/glnxa64/ -DPNG_SKIP_SETJMP_CHECK=TRUE ..
make -j4
```

## Programs
_depth_forest_trainer_: Trains a new forest depth-first
_level_forest_trainer_: Trains a new forest breadh-first
_dist_forest_trainer_: Trains a new forest in a distributed manner using MPI.
_forest_predictor_: Test an existing forest on a dataset.
_forest_converter_: Converts a forest in JSON or binary format to MATLAB format.

## Data file format

CSV format:
Data and label images are given as individual image files. A .csv file contains path to the data and label images.

MATLAB format:
The data is given as a Matlab .mat file with two fields: `data`, `labels` (default names, could be changed).
Both fields contains a 3-dimensional array of size `WxHxN`,
where `N` is the number of images, `W` is the width and `H` is the height of the images.
The arrays should be of type double.
Labels should be from `0` to `(C-1)`, where `C` is the number of classes. Negative labels are considered as background pixels and will be ignored.

## Open issues

### Unit tests

This is research code, so please excuse the lack of testing. I will try to add coverage in the future but this is currently low on my priority list.

### Scaling of distributed implementation

The distributed trainer is working but the scaling is suboptimal because the tree is trained level-wise and communication between nodes grows quickly with tree-depth. I will improve the scaling as soon as possible by introducing a switch to depth-first training as soon as the reached tree-level has as many nodes as workers are available.

### Checkpointing

Checkpointing of forest is implemented for the level-based training code. However, it is not well-tested. It will be improved together with the distributed code.
