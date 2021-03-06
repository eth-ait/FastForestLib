BUILD_TYPE=Release
#BUILD_TYPE=Debug

EIGEN3_INCLUDE_DIR=/usr/include/eigen3/
CEREAL_INCLUDE_DIR=${HOME}/opt/cereal-1.1.2/include/
TCLAP_INCLUDE_DIR=${HOME}/opt/tclap-1.2.1/include/
#MATLAB_INCLUDE_DIR=/site/opt/matlab/r2015a/x64/extern/include/
#MATLAB_LIB_DIR=/site/opt/matlab/r2015a/x64/bin/glnxa64/
BOOST_ROOT=${HOME}/opt/boost-1.59.0/
CIMG_INCLUDE_DIR=${HOME}/opt/CImg-1.6.8/
#export CC="/usr/local/bin/gcc-5"
#export CXX="/usr/local/bin/g++-5"

cmake ../ -DEIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR \
        -DCEREAL_INCLUDE_DIR=$CEREAL_INCLUDE_DIR \
        -DTCLAP_INCLUDE_DIR=$TCLAP_INCLUDE_DIR \
	-DCIMG_INCLUDE_DIR=$CIMG_INCLUDE_DIR \
	-DBOOST_ROOT=$BOOST_ROOT \
	-DPNG_SKIP_SETJMP_CHECK=1 \
	-DWITHOUT_MATLAB=1 \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -G "Unix Makefiles" \
	--no-warn-unused-cli \
	$@

        #-DMATLAB_INCLUDE_DIR=$MATLAB_INCLUDE_DIR \
        #-DMATLAB_LIB_DIR=$MATLAB_LIB_DIR \

