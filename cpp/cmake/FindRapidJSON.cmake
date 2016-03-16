# - Try to find RapidJSON lib
#
#  RapidJSON_FOUND - system has RapidJSON lib
#  RapidJSON_INCLUDE_DIR - the RapidJSON include directory

# Copyright (c) 2015, Benjamin Hepp <benjamin.hepp@posteo.de>

macro(_rapidjson_check_path)

  if(EXISTS "${RapidJSON_INCLUDE_DIR}/rapidjson/document.h")
    set(RapidJSON_OK TRUE)
  endif()

  if(NOT RapidJSON_OK)
    message(STATUS "RapidJSON include path was specified but no document.h file was found: ${RapidJSON_INCLUDE_DIR}")
  endif()

endmacro()

if(NOT RapidJSON_INCLUDE_DIR)

  if (Cereal_FOUND)
    set(RapidJSON_INCLUDE_DIR "${Cereal_INCLUDE_DIR}/cereal/external/")
    set(RapidJSON_OK TRUE)
  else()
    find_path(RapidJSON_INCLUDE_DIR NAMES rapidjson/document.h
    PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    ${KDE4_INCLUDE_DIR}
    #PATH_SUFFIXES rapidjson
    )
  endif()

endif()

if(RapidJSON_INCLUDE_DIR)
  _rapidjson_check_path()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RapidJSON DEFAULT_MSG RapidJSON_INCLUDE_DIR RapidJSON_OK)

mark_as_advanced(RapidJSON_INCLUDE_DIR)
