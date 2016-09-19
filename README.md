# FastForestLib

A library for training and evaluating random forests.

The folder _cpp_ contains the C++ implementation (multithreaded and distributed variant).

The folder _utils_ contains some MATLAB and python scripts to convert data and forests between different formats.

The folder _python_ contains an older Python implementation using Cython. The testing code could still be useful for forests trained with the C++ code.

The folder _data_ contains some MATLAB scripts to generate synthetic data and to convert MATLAB data format to CSV data format. Check it out for details on the data formats. To some extend it can also be used for testing the code.

### Disclaimer ###

This code has been tested and no major bugs have been found. Nevertheless, this software is provided "as is", without warranty of any kind.

## Dependencies
The library uses _boost_ (tested with 1.59.0 and 1.60.0), _Eigen_ (tested with 3.2.0 and 3.2.8), _CImg_ (included), _TCLAP_ (included) and _cereal_ (included). The code also uses _RapidJSON_ but this is already included in _cereal_.
The distributed code requires _boost-mpi_ for communication.

## Compiling

### Linux and OS X

```
mkdir cpp/build
cd cpp/build
cmake .. # Optionally modify CMake configuration to enable/disable multi-threading, MPI, Matlab, HDF5 support etc.
# On a Linux system with MPI you might typically do something like this:
#cmake -DWITH_MPI=TRUE ..
# ... or this if you want Matlab support:
#cmake -DWITH_MPI=TRUE -DWITH_MATLAB=TRUE -DMATLAB_INCLUDE_DIRS=/usr/local/Matlab/R2015a/extern/include/ -DMATLAB_LIB_DIR=/usr/local/Matlab/R2015a/bin/glnxa64/ ..
make -j4
```

### Windows

The code compiles with Visual Studio 2013 and Visual Studio 2015. You need to compile boost (tested with 1.59.0) which requires libpng and zlib.
I would recommend to compile everything with 64bit support.

## Programs
_depth_forest_trainer_: Trains a new forest depth-first
_level_forest_trainer_: Trains a new forest breadh-first
_dist_forest_trainer_: Trains a new forest in a distributed manner using MPI.
_forest_predictor_: Predict labels for a dataset. Can also be used to evaluate a dataset with ground-truth.
_forest_converter_: Converts a forest in JSON or binary format to MATLAB format.

## Data input file format

CSV format:
Data and label images are given as individual image files. A .csv file contains path to the data and label images.

MATLAB format:
The data is given as a Matlab .mat file with two fields: `data`, `labels` (default names, could be changed).
Both fields contains a 3-dimensional array of size `WxHxN`,
where `N` is the number of images, `W` is the width and `H` is the height of the images.
The arrays should be of type double.
Labels should be from `0` to `(C-1)`, where `C` is the number of classes. Negative labels are considered as background pixels and will be ignored.

## Data output file format

HDF5 and MATLAB format:
Predictions can be output as a HDF5 or MATLAB file. In both cases the file contains a dataset/matrix with the predicted labels for each input image (in the same order as the input images).

## Open issues

### Unit tests

This is research code, so please excuse the lack of testing. I will try to add coverage in the future but this is currently low on my priority list.

### Scaling of distributed implementation

The distributed trainer is working but the scaling is suboptimal because the tree is trained level-wise and communication between nodes grows quickly with tree-depth. I will improve the scaling as soon as possible by introducing a switch to depth-first training as soon as the reached tree-level has as many nodes as workers are available.

### Checkpointing

Checkpointing of forest is implemented for the level-based training code. However, it is not well-tested. It will be improved together with the distributed code.
