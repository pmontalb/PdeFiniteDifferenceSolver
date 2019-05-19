cmake_minimum_required(VERSION 3.14)
project(PdeFiniteDifferenceSolver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)
enable_language(CUDA)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

# forge
include_directories(../forge)
include_directories(../forge/include)
# https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux
add_subdirectory(../forge ${CMAKE_BINARY_DIR}/forge EXCLUDE_FROM_ALL)

# CudaLight
include_directories(../CudaLight/CudaLight)
include_directories(../CudaLightKernels)
include_directories(../NpyCpp/Npy++)
include_directories(../NpyCpp/MemoryMapping)
add_subdirectory(../CudaLight ${CMAKE_BINARY_DIR}/CudaLight EXCLUDE_FROM_ALL)

# PdeFiniteDifferenceKernels
include_directories(../PdeFiniteDifferenceKernels)
add_subdirectory(../PdeFiniteDifferenceKernels ${CMAKE_BINARY_DIR}/PdeFiniteDifferenceKernels EXCLUDE_FROM_ALL)

include_directories(PdeFiniteDifferenceSolver)
include_directories(PdeFiniteDifferenceSolverGpuPlot)
include_directories(PdeFiniteDifferenceSolverManager)
include_directories(UnitTests)

# set warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(WARNINGS -Weverything -Werror
            -Wno-c++98-compat -Wno-c++98-compat-pedantic
            -Wno-reserved-id-macro -Wno-switch-enum -Wno-unknown-pragmas -Wno-disabled-macro-expansion
            -Wno-unneeded-internal-declaration -Wno-weak-vtables -Wno-unused-function -Wno-padded -Wno-unused-macros)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-function)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wfatal-errors -fstrict-aliasing -pedantic -pedantic-errors)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(FATAL "MSVC not supported")
endif()

add_library(
        PdeFiniteDifferenceSolverManager
        PdeFiniteDifferenceSolverManager/FiniteDifferenceManager.cpp
)
target_compile_options(PdeFiniteDifferenceSolverManager PRIVATE ${WARNINGS})
target_link_libraries(PdeFiniteDifferenceSolverManager CudaLight PdeFiniteDifferenceKernels)

add_executable(
        PdeFiniteDifferenceSolver

        PdeFiniteDifferenceSolver/main.cpp
        PdeFiniteDifferenceSolver/stdafx.cpp
)
target_link_libraries(PdeFiniteDifferenceSolver PdeFiniteDifferenceSolverManager)

add_executable(
        PdeFiniteDifferenceSolverGpuPlot

        PdeFiniteDifferenceSolverGpuPlot/main.cu
)
target_compile_options(PdeFiniteDifferenceSolverGpuPlot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61 -m64 -restrict --compiler-options -Wall>)
target_compile_options(PdeFiniteDifferenceSolverGpuPlot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -Wno-unknown-pragmas>)
target_link_libraries(PdeFiniteDifferenceSolverGpuPlot PdeFiniteDifferenceSolverManager forge -lcudart -lcublas -lcusolver -lcusparse)

add_executable(
        PdeUnitTests

        UnitTests/main.cpp
        UnitTests/stdafx.cpp

        UnitTests/AdvectionDiffusion1DTests.cpp
        UnitTests/AdvectionDiffusion2DTests.cpp

        UnitTests/WaveEquation1DTests.cpp
        UnitTests/WaveEquation2DTests.cpp
)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(PdeUnitTests PRIVATE -Wno-absolute-value)
endif()
target_link_libraries(PdeUnitTests PdeFiniteDifferenceSolverManager gtest pthread)