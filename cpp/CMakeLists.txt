cmake_minimum_required(VERSION 2.8.11)

project("AITDistributedRandomForest")

set(CMAKE_MODULE_PATH ${AITDistributedRandomForest_SOURCE_DIR}/cmake)

find_package(Eigen3 3.1.2 REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${MATLAB_INCLUDE_DIR} )

link_directories( ${MATLAB_LIB_DIR} )

add_executable(forest_trainer forest_trainer.cpp)

target_link_libraries(forest_trainer libmat libmx)
