cmake_minimum_required(VERSION 2.8.11)

project("AITDistributedRandomForest")

set(CMAKE_MODULE_PATH ${AITDistributedRandomForest_SOURCE_DIR}/cmake)

find_package(Eigen3 3.1.2 REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${MATLAB_INCLUDE_DIR} )

link_directories( ${MATLAB_LIB_DIR} )

file(GLOB headers
	"*.h"
    "forest_trainer.cpp"
)

add_executable(forest_trainer main_trainer.cpp ${headers})

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x" ${CMAKE_CXX_FLAGS})

target_compile_features(forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)

#target_link_libraries(forest_trainer libmat libmx)
target_link_libraries(forest_trainer mat mx)
