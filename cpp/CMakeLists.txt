cmake_minimum_required(VERSION 3.1.3)

project("DistRandomForest")

cmake_policy(SET CMP0004 OLD)

set(CMAKE_MODULE_PATH ${DistRandomForest_SOURCE_DIR}/cmake)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(OpenMP)
	if (OPENMP_FOUND)
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
endif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")

find_package(Boost 1.4.0 COMPONENTS serialization mpi log filesystem system REQUIRED)
find_package(Eigen3 3.1.2 REQUIRED)
find_package(Cereal REQUIRED)
find_package(TCLAP REQUIRED)
find_package(CImg REQUIRED)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS cimg_use_png)
find_package(MPI REQUIRED)
find_package(X11 REQUIRED)
find_package(PNG REQUIRED)

# Workaround for libpng problem on some platforms
if(PNG_SKIP_SETJMP_CHECK)
	add_definitions(-DPNG_SKIP_SETJMP_CHECK)
endif()

if(WITHOUT_MATLAB)
	set(MATLAB_INCLUDE_DIR "")
	set(MATLAB_LIB_DIR "")
	set(MATLAB_LIBRARIES "")
else()
	if(!${WIN32} AND !${APPLE})
		find_package(MATLAB)
		set(MATLAB_LIB_DIR ${MATLAB_MX_LIBRARY})
	endif(!${WIN32} AND !${APPLE})
endif()

include_directories(${Boost_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
if(!WITHOUT_MATLAB)
	include_directories(${MATLAB_INCLUDE_DIR})
endif()
include_directories(${TCLAP_INCLUDE_DIR})
include_directories(${CEREAL_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${CIMG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
if(${UNIX})
	include_directories(${X11_INCLUDE_DIR})
endif()


if(!WITHOUT_MATLAB)
	link_directories(${MATLAB_LIB_DIR})
endif()

set(headers ait.h image_weak_learner.h training.h weak_learner.h
	histogram_statistics.h forest.h tree.h node.h matlab_file_io.h eigen_matrix_io.h
	iterator_utils.h mpl_utils.h logger.h csv_utils.h serialization_utils.h
	bagging_wrapper.h)
#file(GLOB headers
#	"*.h"
#)

if(!WITHOUT_MATLAB)
	if(${WIN32})
	    set(MATLAB_LIBRARIES libmat libmx)
	else(${WIN32})
	    set(MATLAB_LIBRARIES mat mx)
	endif(${WIN32})
endif()

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:AIT_DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS AIT_TESTING)

if(!WITHOUT_MATLAB)
	add_executable(depth_forest_trainer depth_trainer.cpp ${headers} depth_forest_trainer.h)
	target_link_libraries(depth_forest_trainer ${MATLAB_LIBRARIES} ${PNG_LIBRARIES})
	target_compile_features(depth_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
endif()

add_executable(level_forest_trainer level_trainer.cpp ${headers} level_forest_trainer.h bagging_wrapper.h)
target_compile_definitions(level_forest_trainer PUBLIC SERIALIZE_WITH_BOOST)
target_link_libraries(level_forest_trainer ${PNG_LIBRARIES})
target_compile_features(level_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
if(${UNIX})
	if(!WITHOUT_MATLAB)
		target_link_libraries(depth_forest_trainer ${X11_LIBRARIES})
	endif()
	target_link_libraries(level_forest_trainer ${Boost_LIBRARIES} ${X11_LIBRARIES})
endif()

if(${UNIX})
	add_executable(dist_forest_trainer distributed_trainer.cpp ${headers} distributed_forest_trainer.h distributed_bagging_wrapper.h)
	target_compile_definitions(dist_forest_trainer PUBLIC SERIALIZE_WITH_BOOST)
	# On some Unix systems MPI_COMPILE_FLAGS is " -fexceptions"
	string(STRIP ${MPI_COMPILE_FLAGS} MPI_COMPILE_FLAGS)
	target_compile_options(dist_forest_trainer PRIVATE ${MPI_COMPILE_FLAGS})
	target_link_libraries(dist_forest_trainer ${Boost_LIBRARIES} ${MPI_LIBRARIES} ${MPI_LINK_FLAGS} ${X11_LIBRARIES} ${PNG_LIBRARIES})
	target_compile_features(dist_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
endif()
message(${MPI_COMPILE_FLAGS})

#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

#enable_testing()
#add_test(dist_forest_trainer_test ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Debug/dist_forest_trainer -d ../../data/trainingData.mat)
