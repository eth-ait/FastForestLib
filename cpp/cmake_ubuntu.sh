BUILD_TYPE=Debug

cmake ../ \
	-DPNG_SKIP_SETJMP_CHECK=1 \
	-DWITHOUT_MATLAB=1 \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -G "Unix Makefiles" \
	--no-warn-unused-cli \
	$@
