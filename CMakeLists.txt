include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLight)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

# forge
#[[ requirements:
 	sudo apt-get install libboost-all-dev
 	sudo apt-get install libglfw3
 	sudo apt-get install libglfw3-dev
 	sudo apt-get install libfreetype6
 	sudo apt-get install fontconfig
	sudo apt-get install libglm-dev
#]]
set(BUILD_SHARED_LIBS_ORIGINAL ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(forge ${CMAKE_BINARY_DIR}/forge EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_ORIGINAL} CACHE BOOL "" FORCE)
mark_as_advanced(BUILD_SHARED_LIBS_ORIGINAL)

#target_compile_definitions(forge PRIVATE NDEBUG)  # otherwise it doesn't compile due to a missing include
#target_link_libraries(forge PUBLIC -lGL)

# CudaLight
add_subdirectory(CudaLight ${CMAKE_BINARY_DIR}/CudaLight EXCLUDE_FROM_ALL)

# PdeFiniteDifferenceKernels
add_subdirectory(PdeFiniteDifferenceKernels ${CMAKE_BINARY_DIR}/PdeFiniteDifferenceKernels EXCLUDE_FROM_ALL)

create_library(
    NAME
        PdeFiniteDifferenceSolverManager
    SOURCES
        PdeFiniteDifferenceSolverManager/FiniteDifferenceManager.cpp
    DEPENDENCIES
        CudaLight PdeFiniteDifferenceKernels
    PUBLIC_INCLUDE_DIRECTORIES
        PdeFiniteDifferenceSolverManager
)

create_executable(
    NAME
        PdeFiniteDifferenceSolver
    SOURCES
        PdeFiniteDifferenceSolver/main.cpp
    DEPENDENCIES
        PdeFiniteDifferenceSolverManager
    PUBLIC_INCLUDE_DIRECTORIES
	.
)

create_cuda_executable(
    NAME
        PdeFiniteDifferenceSolverGpuPlot
    SOURCES
        PdeFiniteDifferenceSolverGpuPlot/main.cu
    DEPENDENCIES
        PdeFiniteDifferenceSolverManager
    PUBLIC_INCLUDE_DIRECTORIES
	.
    SYSTEM_DEPENDENCIES
        forge
)

create_test(
    NAME
        PdeUnitTests
    SOURCES
        UnitTests/main.cpp

        UnitTests/AdvectionDiffusion1DTests.cpp
        UnitTests/AdvectionDiffusion2DTests.cpp

        UnitTests/WaveEquation1DTests.cpp
        UnitTests/WaveEquation2DTests.cpp
    PUBLIC_INCLUDE_DIRECTORIES
        ${GTEST_INCLUDE_DIR}
    DEPENDENCIES
        PdeFiniteDifferenceSolverManager
    SYSTEM_DEPENDENCIES
        gtest pthread
)
