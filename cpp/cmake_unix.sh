BUILD_TYPE=Release

MATLAB_INCLUDE_DIR="/Applications/MATLAB_R2015a.app/extern/include/"
MATLAB_LIB_DIR="/Applications/MATLAB_R2015a.app/bin/maci64/"
export CC="/usr/local/bin/gcc-5"
export CXX="/usr/local/bin/g++-5"

cmake ../ -DMATLAB_INCLUDE_DIR=$MATLAB_INCLUDE_DIR -DMATLAB_LIB_DIR=$MATLAB_LIB_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -G "Unix Makefiles" $@
