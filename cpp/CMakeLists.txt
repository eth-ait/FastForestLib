cmake_minimum_required(VERSION 3.1.3)

project("DistRandomForest")

cmake_policy(SET CMP0004 OLD)

# Add some cmake macros
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add default location of TCLAP and CImg headers
set(TCLAP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/contrib/tclap-1.2.1/include" CACHE PATH "TCLAP include path")
set(CIMG_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/contrib/CImg-1.6.9" CACHE PATH "CImg include path")
set(Cereal_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/contrib/cereal-1.1.2/include" CACHE PATH "Cereal include path")
# Cereal already provides a version of RapidJSON that conflicts with newer versions (see FindRapidJSON.cmake script)
#set(RapidJSON_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/contrib/rapidjson/include" CACHE PATH "RapidJSON include path")

# Set default option for MATLAB
if(NOT DEFINED WITH_MATLAB)
    set(WITH_MATLAB FALSE CACHE BOOL "With MATLAB")
endif()

# Set default option for HDF5
if(NOT DEFINED WITH_HDF5)
    set(WITH_HDF5 TRUE CACHE BOOL "With HDF5")
endif()

# Set default option for MPI
if(NOT DEFINED WITH_MPI)
    set(WITH_MPI FALSE CACHE BOOL "With MPI")
endif()

# Set default option for threading
if(NOT DEFINED WITH_MULTITHREADING)
    set(WITH_MULTITHREADING TRUE CACHE BOOL "With multi-threading")
endif()

# Set default option for profiling output
if(NOT DEFINED WITH_PROFILING)
    set(WITH_PROFILING FALSE CACHE BOOL "With profiling output")
endif()

# Put some variables into the cache
if(${WITH_MATLAB})
	set(MATLAB_LIB_DIR "" CACHE PATH "Matlab library path")
	set(MATLAB_INCLUDE_DIRS "" CACHE PATH "Matlab include path")
endif()
if("${UNIX}")
	set(PNG_SKIP_SETJMP_CHECK TRUE CACHE BOOL "Skip PNG SETJMP check. Required for some platforms.")
else()
	set(PNG_SKIP_SETJMP_CHECK FALSE CACHE BOOL "Skip PNG SETJMP check. Required for some platforms.")
endif()

# Use libc++ if using Clang
#if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++")
#endif()

# Not using OpenMP anymore. Using C++ threads now.
# Enable OpenMP if not using Clang
#if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#    find_package(OpenMP)
#	 if (OPENMP_FOUND)
#		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#	endif()
#endif()

# Find required packages (threads, boost, mpi, eigen3, cereal, tclap, cimg)
find_package(Threads REQUIRED)
set(Boost_USE_DEBUG_RUNTIME OFF)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# Prevent boost auto-linking
add_definitions("-DBOOST_ALL_NO_LIB")
if(${WITH_MPI})
	find_package(MPI REQUIRED)
	find_package(Boost 1.4.0 COMPONENTS serialization mpi log filesystem system REQUIRED)
else()
	find_package(Boost 1.4.0 COMPONENTS serialization log filesystem system REQUIRED)
endif()
if(${WITH_MPI})
	find_package(HDF5 COMPONENTS CXX REQUIRED)
endif()
find_package(Eigen3 3.1.2 REQUIRED)
find_package(Cereal REQUIRED)
find_package(TCLAP REQUIRED)
find_package(CImg REQUIRED)
find_package(RapidJSON REQUIRED)

# Enable PNG support in CImg
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS cimg_use_png)
find_package(X11)
find_package(PNG)
# Workaround for libpng problem on some platforms
if(${PNG_SKIP_SETJMP_CHECK})
	add_definitions(-DPNG_SKIP_SETJMP_CHECK)
endif()

# Add MATLAB Libraries
if(${WITH_MATLAB})
	if(${WIN32})
	    set(MATLAB_LIBRARIES libmat libmx)
	else()
	    set(MATLAB_LIBRARIES mat mx)
	endif()
	link_directories(${MATLAB_LIB_DIR})
endif()

# Add include directories
include_directories(${Boost_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
if(${WITH_MATLAB})
	include_directories(${MATLAB_INCLUDE_DIRS})
endif()
include_directories(${TCLAP_INCLUDE_DIR})
include_directories(${Cereal_INCLUDE_DIR})
include_directories(${RapidJSON_INCLUDE_DIR})
include_directories(${CIMG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
if (${WITH_HDF5})
	include_directories(${HDF5_INCLUDE_DIRS})
endif()
if(${UNIX})
	include_directories(${X11_INCLUDE_DIR})
endif()

# Specify source files to use
set(headers ait.h image_weak_learner.h training.h weak_learner.h
	histogram_statistics.h forest.h tree.h node.h matlab_file_io.h hdf5_file_io.h eigen_matrix_io.h io_utils.h
	iterator_utils.h mpl_utils.h logger.h csv_utils.h serialization_utils.h evaluation_utils.h
	bagging_wrapper.h config_utils.h common.h)
#file(GLOB headers
#	"*.h"
#)

# Set WITH_MATLAB macro for cpp files
if(${WITH_MATLAB})
	set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<BOOL:${WITH_MATLAB}>:WITH_MATLAB=1>)
endif()

# Set WITH_HDF5 macro for cpp files
if(${WITH_HDF5})
	set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<BOOL:${WITH_HDF5}>:WITH_HDF5=1>)
endif()

# Set AIT_DEBUG macro for cpp files if in debug mode
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:AIT_DEBUG=1>)
# Set AIT_TESTING macro for cpp files if in debug mode
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:AIT_TESTING=1>)
# Set AIT_TESTING macro for all build modes
#set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS AIT_TESTING=1)

# Set AIT_MULTI_THREADING macro if multi-threading is enabled
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<BOOL:${WITH_MULTITHREADING}>:AIT_MULTI_THREADING=1>)

# Executable target: depth_forest_trainer
add_executable(depth_forest_trainer depth_trainer.cpp common.cpp ${headers} depth_forest_trainer.h)
target_link_libraries(depth_forest_trainer ${PNG_LIBRARIES})
target_link_libraries(depth_forest_trainer ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(depth_forest_trainer ${Boost_LIBRARIES})
target_link_libraries(depth_forest_trainer ${MATLAB_LIBRARIES})
if("${WITH_MPI}")
	target_include_directories(depth_forest_trainer PUBLIC ${MPI_INCLUDE_PATH})
	target_link_libraries(depth_forest_trainer ${MPI_LIBRARIES})
endif()
target_compile_features(depth_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
if("${UNIX}")
	target_link_libraries(depth_forest_trainer ${X11_LIBRARIES})
endif()

# Executable target: forest_converter
if(${WITH_MATLAB})
    add_executable(forest_converter forest_converter.cpp ${headers})
    target_link_libraries(forest_converter ${MATLAB_LIBRARIES})
    target_compile_features(forest_converter PRIVATE cxx_auto_type cxx_variadic_templates)
endif()

# Executable target: level_forest_trainer
add_executable(level_forest_trainer level_trainer.cpp common.cpp ${headers} level_forest_trainer.h bagging_wrapper.h)
target_link_libraries(level_forest_trainer ${PNG_LIBRARIES})
target_link_libraries(level_forest_trainer ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(level_forest_trainer ${Boost_LIBRARIES})
target_link_libraries(level_forest_trainer ${MATLAB_LIBRARIES})
if("${WITH_MPI}")
	target_include_directories(level_forest_trainer PUBLIC ${MPI_INCLUDE_PATH})
	target_link_libraries(level_forest_trainer ${MPI_LIBRARIES})
endif()
target_compile_features(level_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
if("${UNIX}")
	target_link_libraries(level_forest_trainer ${X11_LIBRARIES})
endif()

# Executable target: dist_forest_trainer
if("${UNIX}" AND "${WITH_MPI}")
	add_executable(dist_forest_trainer distributed_trainer.cpp common.cpp ${headers} distributed_forest_trainer.h distributed_bagging_wrapper.h)
	target_compile_definitions(dist_forest_trainer PUBLIC SERIALIZE_WITH_BOOST=1)
	# On some Unix systems MPI_COMPILE_FLAGS is " -fexceptions"
	string(STRIP "${MPI_COMPILE_FLAGS}" MPI_COMPILE_FLAGS)
	target_compile_options(dist_forest_trainer PRIVATE "${MPI_COMPILE_FLAGS}")
	target_include_directories(dist_forest_trainer PUBLIC ${MPI_INCLUDE_PATH})
	target_link_libraries(dist_forest_trainer ${Boost_LIBRARIES})
	target_link_libraries(dist_forest_trainer ${MPI_LIBRARIES} ${MPI_LINK_FLAGS})
	target_link_libraries(dist_forest_trainer ${X11_LIBRARIES} ${PNG_LIBRARIES})
	target_link_libraries(dist_forest_trainer ${MATLAB_LIBRARIES})
	target_link_libraries (dist_forest_trainer ${CMAKE_THREAD_LIBS_INIT})
	target_compile_features(dist_forest_trainer PRIVATE cxx_auto_type cxx_variadic_templates)
endif()

# Executable target: forest_predictor
add_executable(forest_predictor forest_predictor.cpp common.cpp ${headers})
target_link_libraries(forest_predictor ${PNG_LIBRARIES})
target_link_libraries(forest_predictor ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(forest_predictor ${Boost_LIBRARIES})
target_link_libraries(forest_predictor ${MATLAB_LIBRARIES})
if (${WITH_HDF5})
	target_link_libraries(forest_predictor ${HDF5_LIBRARIES})
endif()
if("${WITH_MPI}")
	target_include_directories(forest_predictor PUBLIC ${MPI_INCLUDE_PATH})
	target_link_libraries(forest_predictor ${MPI_LIBRARIES})
endif()
target_compile_features(forest_predictor PRIVATE cxx_auto_type cxx_variadic_templates)
if("${UNIX}")
	target_link_libraries(forest_predictor ${X11_LIBRARIES})
endif()

# SET AIT_PROFILE or AIT_PROFILE_DISTRIBUTED macro for cpp files if profiling output is enabled
if(${WITH_MATLAB})
    target_compile_definitions(forest_converter PRIVATE $<$<BOOL:${WITH_PROFILING}>:AIT_PROFILE=1>)
endif()
target_compile_definitions(depth_forest_trainer PRIVATE $<$<BOOL:${WITH_PROFILING}>:AIT_PROFILE=1>)
target_compile_definitions(level_forest_trainer PRIVATE $<$<BOOL:${WITH_PROFILING}>:AIT_PROFILE=1>)
target_compile_definitions(forest_predictor PRIVATE $<$<BOOL:${WITH_PROFILING}>:AIT_PROFILE=1>)
if(${WITH_MPI})
    target_compile_definitions(dist_forest_trainer PRIVATE $<$<BOOL:${WITH_PROFILING}>:AIT_PROFILE_DISTRIBUTED=1>)
endif()

# C++11 support (should be done through target_compile_features)
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

#enable_testing()
#add_test(dist_forest_trainer_test ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Debug/dist_forest_trainer -d ../../data/trainingData.mat)
