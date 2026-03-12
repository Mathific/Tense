# BSD 3-Clause License
#
# Copyright (c) 2021-2026, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Project
cmake_minimum_required(VERSION 3.16)
project(Tense VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Variables
set(TENSE_MASTER OFF CACHE INTERNAL "")
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(TENSE_MASTER ON CACHE INTERNAL "")
endif()

set(TENSE_VERSION ${PROJECT_VERSION} CACHE INTERNAL "")
option(TENSE_ENABLE_FLAGS "Build for Native Architecture" ON)
option(TENSE_ENABLE_TESTS "Build Tense Tests" ${TENSE_MASTER})
option(TENSE_ENABLE_EXAMPLES "Build Tense Examples" ${TENSE_MASTER})
option(TENSE_ENABLE_INSTALL "Install Tense Library" ${TENSE_MASTER})
option(TENSE_ENABLE_BLAS "Enable BLAS support" ON)
option(TENSE_ENABLE_OPENMP "Enable OpenMP Support" ON)

# Files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tense/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gen/tense/config.h")
file(GLOB HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tense/*.h"
    "${CMAKE_CURRENT_BINARY_DIR}/gen/tense/config.h")
file(GLOB VECTOR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tense/vector/*.h")
file(GLOB MATRIX_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tense/matrix/*.h")
file(GLOB TENSOR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tense/tensor/*.h")

# Library
add_library(Tense INTERFACE)
target_include_directories(Tense INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/>")
# target_compile_options(Tense INTERFACE
#     "-Rpass=loop-vectorize"
#     "-Rpass-missed=loop-vectorize"
#     "-Rpass-analysis=loop-vectorize")

if(TENSE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(Tense INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(Tense INTERFACE TENSE_USE_OPENMP)

    message("Tense OpenMP Enabled!")
else()
    message("Tense OpenMP Enabled!")
endif()

if(TENSE_ENABLE_BLAS)
    option(BLASW_INSTALL "" ${TENSE_MASTER})
    add_subdirectory(blasw)
    target_link_libraries(Tense INTERFACE BLASW::BLASW)
    target_compile_definitions(Tense INTERFACE TENSE_USE_BLAS)

    message("Tense BLAS Enabled!")
else()
    message("Tense BLAS Enabled!")
endif()

if(TENSE_NATIVE_FLAGS)
    target_compile_options(Tense INTERFACE "-march=native" "-ffast-math")

    message("Tense Flags Enabled!")
else()
    message("Tense Flags Enabled!")
endif()

add_library(Tense::Tense ALIAS Tense)

# Installation
if(TENSE_ENABLE_INSTALL)
    install(FILES ${HEADERS}
        DESTINATION include/tense)
    install(FILES ${VECTOR_HEADERS}
        DESTINATION include/tense/matrix)
    install(FILES ${MATRIX_HEADERS}
        DESTINATION include/tense/matrix)
    install(FILES ${TENSOR_HEADERS}
        DESTINATION include/tense/tensor)

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "TenseConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TenseConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TenseConfig.cmake" @ONLY)
    file(GLOB MODULES
        "${CMAKE_CURRENT_BINARY_DIR}/TenseConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TenseConfigVersion.cmake")
    install(FILES ${MODULES}
        DESTINATION lib/cmake/tense)

    install(TARGETS Tense
        EXPORT TenseTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
    install(EXPORT TenseTargets
        NAMESPACE Tense::
        DESTINATION lib/cmake/tense)
    export(TARGETS Tense
        NAMESPACE Tense::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/TenseTargets.cmake")

    message("Tense Installation Enabled!")
else()
    message("Tense Installation Disabled!")
endif()

if(TENSE_ENABLE_TESTS)
    if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif(WIN32)

    set(BUILD_GMOCK OFF CACHE BOOL "")
    set(INSTALL_GTEST OFF CACHE BOOL "")
    add_subdirectory(./test/gtest/)

    add_executable(tests
        "test/tests.cpp" "test/matrix.cpp" "test/tensor.cpp"
        ${HEADERS} ${VECTOR_HEADERS} ${MATRIX_HEADERS} ${TENSOR_HEADERS})
    target_link_libraries(tests PUBLIC
        Tense::Tense gtest gtest_main)

    message("Tense Tests Enabled!")
else()
    message("Tense Tests Disabled!")
endif()

if(TENSE_ENABLE_EXAMPLES)
    add_executable(LeastSquares example/least_squares.cpp)
    target_link_libraries(LeastSquares PUBLIC Tense::Tense)

    add_executable(NeuralNetwork example/neural_network.cpp)
    target_link_libraries(NeuralNetwork PUBLIC Tense::Tense)

    message("Tense Examples Enabled!")
else()
    message("Tense Examples Disabled!")
endif()
