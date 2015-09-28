# - Try to find cereal lib
#
#  CEREAL_FOUND - system has cereal lib
#  CEREAL_INCLUDE_DIR - the cereal include directory

# Copyright (c) 2015, Benjamin Hepp <benjamin.hepp@posteo.de>

macro(_cereal_check_path)

  if(EXISTS "${CEREAL_INCLUDE_DIR}/cereal/cereal.hpp")
    set(CEREAL_OK TRUE)
  endif()

  if(NOT CEREAL_OK)
    message(STATUS "Cereal include path was specified but not cereal.hpp file was found: ${CEREAL_INCLUDE_DIR}")
  endif()

endmacro()

if(NOT CEREAL_INCLUDE_DIR)

  find_path(CEREAL_INCLUDE_DIR NAMES cereal/cereal.hpp
	PATHS
	${CMAKE_INSTALL_PREFIX}/include
	${KDE4_INCLUDE_DIR}
	#PATH_SUFFIXES cereal
  )

endif()

if(CEREAL_INCLUDE_DIR)
  _cereal_check_path()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cereal DEFAULT_MSG CEREAL_INCLUDE_DIR CEREAL_OK)

mark_as_advanced(CEREAL_INCLUDE_DIR)
