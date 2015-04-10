cmake_minimum_required(VERSION 2.8.11)

project("AITDistributedRandomForest")

set(CMAKE_MODULE_PATH ${AITDistributedRandomForest_SOURCE_DIR}/cmake)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(OpenMP)
endif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")

find_package(cereal REQUIRED)

find_package(Eigen3 3.1.2 REQUIRED)

if(!${MSVC} AND !${APPLE})
	find_package(MATLAB REQUIRED)
	set(MATLAB_LIB_DIR ${MATLAB_MX_LIBRARY})
endif(!${MSVC} AND !${APPLE})

include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${MATLAB_INCLUDE_DIR} )
include_directories( ${cereal_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS})

link_directories( ${MATLAB_LIB_DIR} )

file(GLOB headers
	"*.h"
    "forest_trainer.cpp"
)

add_executable(forest_trainer main_trainer.cpp ${headers})

if(${MSVC})
	target_compile_features(forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
else(${MSVC})
	set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
endif(${MSVC})


if(${MSVC})
    set(MATLAB_LIBRARIES libmat libmx)
else(${MSVC})
    set(MATLAB_LIBRARIES mat mx)
endif(${MSVC})

target_link_libraries(forest_trainer ${MATLAB_LIBRARIES})
