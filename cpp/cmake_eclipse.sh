#BUILD_TYPE=Debug
BUILD_TYPE=RelWithDebInfo

ECLIPSE_VERSION=4.2

cmake `dirname $0` \
    -DCMAKE_ECLIPSE_VERSION=$ECLIPSE_VERSION \
    -DCMAKE_CXX_COMPILER_ARG1=-std=c++11 \
    -DCMAKE_CXX_STANDARD=11 \
	-DPNG_SKIP_SETJMP_CHECK=1 \
	-DWITHOUT_MATLAB=1 \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -G "Eclipse CDT4 - Unix Makefiles" \
	--no-warn-unused-cli \
	$@
