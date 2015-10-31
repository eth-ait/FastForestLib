# - Try to find CImg lib
#
#  CIMG_FOUND - system has CImg lib
#  CIMG_INCLUDE_DIR - the CImg include directory

# Copyright (c) 2015, Benjamin Hepp <benjamin.hepp@posteo.de>

macro(_cimg_check_path)

  if(EXISTS "${CIMG_INCLUDE_DIR}/CImg.h")
    set(CIMG_OK TRUE)
  endif()

  if(NOT CIMG_OK)
    message(STATUS "CImg include path was specified but not CImg.h file was found: ${CIMG_INCLUDE_DIR}")
  endif()

endmacro()

if(NOT CIMG_INCLUDE_DIR)

  find_path(CIMG_INCLUDE_DIR NAMES CImg.h
	PATHS
	${CMAKE_INSTALL_PREFIX}/include
	${KDE4_INCLUDE_DIR}
  )

endif()

if(CIMG_INCLUDE_DIR)
  _cimg_check_path()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CImg DEFAULT_MSG CIMG_INCLUDE_DIR CIMG_OK)

mark_as_advanced(CIMG_INCLUDE_DIR)
