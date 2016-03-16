# - Try to find Xlib lib
#
#  XLIB_FOUND - system has Xlib lib
#  XLIB_INCLUDE_DIR - the Xlib include directory

# Copyright (c) 2015, Benjamin Hepp <benjamin.hepp@posteo.de>

macro(_xlib_check_path)

  if(EXISTS "${XLIB_INCLUDE_DIR}/xlib/cereal.hpp")
    set(XLIB_OK TRUE)
  endif()

  if(NOT XLIB_OK)
    message(STATUS "Cereal include path was specified but no xlib.hpp file was found: ${XLIB_INCLUDE_DIR}")
  endif()

endmacro()

if(NOT XLIB_INCLUDE_DIR)

  find_path(XLIB_INCLUDE_DIR NAMES xlib/cereal.hpp
	PATHS
	${CMAKE_INSTALL_PREFIX}/include
	${KDE4_INCLUDE_DIR}
	#PATH_SUFFIXES xlib
  )

endif()

if(XLIB_INCLUDE_DIR)
  _xlib_check_path()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Xlib DEFAULT_MSG XLIB_INCLUDE_DIR XLIB_OK)

mark_as_advanced(XLIB_INCLUDE_DIR)
