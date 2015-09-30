cmake_minimum_required(VERSION 2.8.11)

project("DistRandomForest")

set(CMAKE_MODULE_PATH ${DistRandomForest_SOURCE_DIR}/cmake)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(OpenMP)
	if (OPENMP_FOUND)
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
endif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")

find_package(Boost 1.4.0 REQUIRED)

find_package(Eigen3 3.1.2 REQUIRED)

find_package(Cereal REQUIRED)

if(!${WIN32} AND !${APPLE})
	find_package(MATLAB REQUIRED)
	set(MATLAB_LIB_DIR ${MATLAB_MX_LIBRARY})
endif(!${WIN32} AND !${APPLE})

include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${MATLAB_INCLUDE_DIR} )
include_directories( ${CEREAL_INCLUDE_DIR} )

link_directories( ${MATLAB_LIB_DIR} )

set(headers ait.h image_weak_learner.h weak_learner.h forest_trainer.h histogram_statistics.h forest.h tree.h node.h matlab_file_io.h eigen_matrix_io.h)
#file(GLOB headers
#	"*.h"
#)

if(${WIN32})
    set(MATLAB_LIBRARIES libmat libmx)
else(${WIN32})
    set(MATLAB_LIBRARIES mat mx)
endif(${WIN32})

add_executable(forest_trainer main_trainer.cpp ${headers})
add_executable(dist_forest_trainer distributed_trainer.cpp ${headers} distributed_forest_trainer.h)

target_link_libraries(forest_trainer ${MATLAB_LIBRARIES})
target_link_libraries(dist_forest_trainer ${MATLAB_LIBRARIES})

target_compile_features(forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
target_compile_features(dist_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
