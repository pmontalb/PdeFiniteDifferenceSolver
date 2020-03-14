include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLight)

include(cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

# forge
include_directories(../forge)
include_directories(../forge/include)
# [git@github.com:arrayfire/forge.git]
# git clone git@github.com:arrayfire/forge.git
# sudo apt-get install libboost-all-dev
# sudo apt-get install libglfw3
# sudo apt-get install libglfw3-dev
# sudo apt-get install libfreetype6
# sudo apt-get install fontconfig
# sudo apt-get install libglm-dev
add_subdirectory(../forge ${CMAKE_BINARY_DIR}/forge EXCLUDE_FROM_ALL)

# CudaLight
add_subdirectory(../CudaLight ${CMAKE_BINARY_DIR}/CudaLight EXCLUDE_FROM_ALL)

# PdeFiniteDifferenceKernels
add_subdirectory(../PdeFiniteDifferenceKernels ${CMAKE_BINARY_DIR}/PdeFiniteDifferenceKernels EXCLUDE_FROM_ALL)

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
		../forge ../forge/include .
    SYSTEM_DEPENDENCIES
        forge -lGL
)

find_package(GTest REQUIRED)
create_test(
    NAME
        PdeUnitTests
    SOURCES
        UnitTests/main.cpp
        UnitTests/stdafx.cpp

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