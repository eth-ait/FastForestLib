# - Try to find cereal lib
#
#  Cereal_FOUND - system has cereal lib
#  Cereal_INCLUDE_DIR - the cereal include directory

# Copyright (c) 2015, Benjamin Hepp <benjamin.hepp@posteo.de>

macro(_cereal_check_path)

  if(EXISTS "${Cereal_INCLUDE_DIR}/cereal/cereal.hpp")
    set(Cereal_OK TRUE)
  endif()

  if(NOT Cereal_OK)
    message(STATUS "Cereal include path was specified but no cereal.hpp file was found: ${Cereal_INCLUDE_DIR}")
  endif()

endmacro()

if(NOT Cereal_INCLUDE_DIR)

  find_path(Cereal_INCLUDE_DIR NAMES cereal/cereal.hpp
	PATHS
	${CMAKE_INSTALL_PREFIX}/include
	${KDE4_INCLUDE_DIR}
	#PATH_SUFFIXES cereal
  )

endif()

if(Cereal_INCLUDE_DIR)
  _cereal_check_path()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cereal DEFAULT_MSG Cereal_INCLUDE_DIR Cereal_OK)

mark_as_advanced(Cereal_INCLUDE_DIR)
