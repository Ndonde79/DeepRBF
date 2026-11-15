cmake_minimum_required(VERSION 3.20)
project(deeprbf LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Global C++ standard and compiler flags
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/bigobj /permissive- /EHsc /utf-8)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/DNDEBUG>)
    add_compile_options($<$<CONFIG:Debug>:/Od>)
    if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
    add_compile_options($<$<CONFIG:Debug>:-O0> $<$<CONFIG:Debug>:-g>)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(DEEPRBF_BUILD_PYTHON    "Build pybind11 Python module"            ON)
option(DEEPRBF_BUILD_TESTS     "Build tests"                            OFF)
option(DEEPRBF_ENABLE_OPENMP   "Enable OpenMP"                          ON)
option(DEEPRBF_ENABLE_MKL      "Enable Intel MKL (if available)"        OFF)
option(DEEPRBF_REQUIRE_MKL     "Fail if requested MKL is unavailable"  OFF)
option(DEEPRBF_ENABLE_SCALFMM  "Enable ScalFMM if available"            OFF)
option(DEEPRBF_REQUIRE_SCALFMM "Fail configuration if ScalFMM missing"  OFF)
option(DEEPRBF_SCALFMM_FORCE_SEQUENTIAL "Force sequential FMM traversal" OFF)
option(DEEPRBF_STRICT_SCALFMM_RUNTIME   "Strict ScalFMM runtime checks"  OFF)
option(DEEPRBF_FAST_MATH       "Enable fast math"                       OFF)
option(DEEPRBF_NATIVE_ARCH     "Enable -march=native (non-MSVC)"        OFF)
option(DEEPRBF_USE_NANOFLANN   "Use bundled nanoflann KD-tree header"   OFF)

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
set(DEEPRBF_VERSION_MAJOR 3)
set(DEEPRBF_VERSION_MINOR 4)
set(DEEPRBF_VERSION_PATCH 1)
set(DEEPRBF_VERSION "${DEEPRBF_VERSION_MAJOR}.${DEEPRBF_VERSION_MINOR}.${DEEPRBF_VERSION_PATCH}")

# -----------------------------------------------------------------------------
# Fast math / native arch
# -----------------------------------------------------------------------------
if(DEEPRBF_FAST_MATH)
    if(MSVC)
        add_compile_options(/fp:fast)
    else()
        add_compile_options(-ffast-math)
    endif()
endif()

if(DEEPRBF_NATIVE_ARCH AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# -----------------------------------------------------------------------------
# Eigen
# -----------------------------------------------------------------------------
if(DEFINED DEEPRBF_EIGEN_ROOT)
    set(EIGEN3_INCLUDE_DIR "${DEEPRBF_EIGEN_ROOT}")
endif()

find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    HINTS ${DEEPRBF_EIGEN_ROOT} ENV EIGEN3_ROOT
    PATHS /usr/include/eigen3 /usr/local/include/eigen3
          /opt/homebrew/include/eigen3
          C:/eigen-3.4.0 C:/eigen)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
        "Eigen 3 not found. Set -DDEEPRBF_EIGEN_ROOT=<path> or install Eigen3.")
endif()
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# -----------------------------------------------------------------------------
# OpenMP
# -----------------------------------------------------------------------------
set(_HAVE_OPENMP FALSE)
if(DEEPRBF_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(_HAVE_OPENMP TRUE)
        message(STATUS "OpenMP: enabled (${OpenMP_CXX_VERSION})")
    else()
        message(STATUS "OpenMP: requested but not found; building single-threaded")
    endif()
else()
    message(STATUS "OpenMP: disabled by user")
endif()

# -----------------------------------------------------------------------------
# MKL — single dynamic library model (mkl_rt.lib + mkl_rt.3.dll)
#
# Intel's MKLConfig.cmake forces MKL_LINK=static in the 2026.x releases and
# ignores overrides. We bypass find_package(MKL) entirely and link mkl_rt.lib
# directly. At runtime, mkl_rt.3.dll dispatches to the appropriate CPU kernels
# and threading layer, so only one PATH entry is needed.
# -----------------------------------------------------------------------------
set(_HAVE_MKL FALSE)
if(DEEPRBF_ENABLE_MKL)
    if(DEFINED DEEPRBF_MKL_ROOT)
        set(MKL_ROOT_DIR "${DEEPRBF_MKL_ROOT}")
    elseif(DEFINED ENV{MKLROOT})
        set(MKL_ROOT_DIR "$ENV{MKLROOT}")
    endif()

    if(MKL_ROOT_DIR)
        find_path(MKL_INCLUDE_DIR mkl.h
            HINTS "${MKL_ROOT_DIR}/include"
            PATHS "${MKL_ROOT_DIR}/include")
        find_library(MKL_RT_LIB mkl_rt
            HINTS "${MKL_ROOT_DIR}/lib"
            PATHS "${MKL_ROOT_DIR}/lib"
                  "${MKL_ROOT_DIR}/lib/intel64")
    endif()

    if(MKL_INCLUDE_DIR AND MKL_RT_LIB)
        set(_HAVE_MKL TRUE)
        message(STATUS "MKL: dynamic link via mkl_rt")
        message(STATUS "  MKL root    : ${MKL_ROOT_DIR}")
        message(STATUS "  MKL include : ${MKL_INCLUDE_DIR}")
        message(STATUS "  MKL runtime : ${MKL_RT_LIB}")
    else()
        if(DEEPRBF_REQUIRE_MKL)
            message(FATAL_ERROR "Required MKL not found: mkl.h/mkl_rt.lib under ${MKL_ROOT_DIR}")
        endif()
        message(STATUS "MKL: requested but mkl.h / mkl_rt.lib not found under ${MKL_ROOT_DIR}")
    endif()
else()
    message(STATUS "MKL: disabled by user")
endif()

# -----------------------------------------------------------------------------
# ScalFMM
# -----------------------------------------------------------------------------
set(_HAVE_SCALFMM FALSE)
if(DEEPRBF_ENABLE_SCALFMM)
    if(DEFINED DEEPRBF_SCALFMM_ROOT)
        list(APPEND CMAKE_PREFIX_PATH "${DEEPRBF_SCALFMM_ROOT}")
    endif()

    find_package(scalfmm QUIET CONFIG)
    if(scalfmm_FOUND)
        set(_HAVE_SCALFMM TRUE)
        message(STATUS "ScalFMM: found")
    elseif(DEEPRBF_REQUIRE_SCALFMM)
        message(FATAL_ERROR "ScalFMM requested and required but not found")
    else()
        message(STATUS "ScalFMM: requested but not found; using stub backend")
    endif()
else()
    message(STATUS "ScalFMM: disabled by user")
endif()

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
set(DEEPRBF_HEADERS
    include/deeprbf/core.hpp
    include/deeprbf/kernels.hpp
    include/deeprbf/spatial_hash.hpp
    include/deeprbf/nested_variogram.hpp
    include/deeprbf/local_variogram.hpp
    include/deeprbf/gmres.hpp
    include/deeprbf/mkl_solver.hpp
    include/deeprbf/rbf_model.hpp
    include/deeprbf/operator.hpp
    include/deeprbf/scalfmm_adapter.hpp
    include/deeprbf/scalfmm_config.h
    include/deeprbf/detail/scalfmm_backend.hpp
    include/deeprbf/domain_decomposition.hpp
    include/deeprbf/gc_geostatistics.hpp
    include/deeprbf/gc_cokriging.hpp
    include/deeprbf/gc_lmc_fit.hpp
    include/deeprbf/gc_block_kriging.hpp
    include/deeprbf/gc_change_of_support.hpp
    include/deeprbf/implicit_model.hpp
    include/deeprbf/geological_implicit_model.hpp
    include/deeprbf/mps_snesim.hpp
    include/deeprbf/sparse_compact.hpp
    include/deeprbf/gc_simulation.hpp
)

set(DEEPRBF_SOURCES
    src/nested_variogram.cpp
    src/local_variogram.cpp
    src/gmres.cpp
    src/mkl_solver.cpp
    src/operator.cpp
    src/scalfmm_adapter.cpp
    src/domain_decomposition.cpp
    src/rbf_model.cpp
    src/sparse_compact.cpp
    src/gc_geostatistics.cpp
    src/gc_simulation.cpp
    src/audited_engine.cpp
    src/indicator_simulation.cpp
    src/truncated_gibbs.cpp
    src/gc_cokriging.cpp
    src/gc_lmc_fit.cpp
    src/lmc_covariance_fit.cpp
    src/gc_block_kriging.cpp
    src/gc_change_of_support.cpp
    src/implicit_model.cpp
    src/geological_implicit_model.cpp
    src/mps_snesim.cpp
)

# ScalFMM backend source: only one of these two
if(_HAVE_SCALFMM)
    list(APPEND DEEPRBF_SOURCES src/scalfmm_backend_v3.cpp)
else()
    list(APPEND DEEPRBF_SOURCES src/scalfmm_backend_stub.cpp)
endif()

# Filter non-existent sources
set(DEEPRBF_SOURCES_EXISTING "")
foreach(src IN LISTS DEEPRBF_SOURCES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        list(APPEND DEEPRBF_SOURCES_EXISTING "${src}")
    else()
        message(FATAL_ERROR "Required source missing: ${src}")
    endif()
endforeach()
set(DEEPRBF_SOURCES "${DEEPRBF_SOURCES_EXISTING}")

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(deeprbf_core STATIC ${DEEPRBF_SOURCES} ${DEEPRBF_HEADERS})
add_library(deeprbf::core ALIAS deeprbf_core)

target_include_directories(deeprbf_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        "${EIGEN3_INCLUDE_DIR}"
)

target_compile_definitions(deeprbf_core PUBLIC
    DEEPRBF_VERSION="${DEEPRBF_VERSION}"
)

if(_HAVE_OPENMP)
    target_link_libraries(deeprbf_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(deeprbf_core PUBLIC DEEPRBF_HAS_OPENMP=1)
endif()

if(_HAVE_MKL)
    target_include_directories(deeprbf_core PRIVATE "${MKL_INCLUDE_DIR}")
    target_link_libraries(deeprbf_core PUBLIC "${MKL_RT_LIB}")
    target_compile_definitions(deeprbf_core PUBLIC DEEPRBF_HAS_MKL=1)
endif()

if(_HAVE_SCALFMM)
    if(TARGET scalfmm::scalfmm)
        target_link_libraries(deeprbf_core PUBLIC scalfmm::scalfmm)
    elseif(TARGET scalfmm::scalfmm-headers)
        target_link_libraries(deeprbf_core PUBLIC scalfmm::scalfmm-headers)
    elseif(TARGET scalfmm)
        target_link_libraries(deeprbf_core PUBLIC scalfmm)
    else()
        message(FATAL_ERROR "ScalFMM package found but no supported CMake target was exported")
    endif()
    target_compile_definitions(deeprbf_core PUBLIC DEEPRBF_HAS_SCALFMM=1)
    if(DEEPRBF_SCALFMM_FORCE_SEQUENTIAL)
        target_compile_definitions(deeprbf_core PUBLIC DEEPRBF_SCALFMM_FORCE_SEQUENTIAL=1)
    endif()
    if(DEEPRBF_STRICT_SCALFMM_RUNTIME)
        target_compile_definitions(deeprbf_core PUBLIC DEEPRBF_STRICT_SCALFMM_RUNTIME=1)
    endif()
endif()

# -----------------------------------------------------------------------------
# Python bindings
# -----------------------------------------------------------------------------
if(DEEPRBF_BUILD_PYTHON)
    find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        message(FATAL_ERROR
            "pybind11 not found. Install with: pip install pybind11, "
            "or set -Dpybind11_DIR=<path>/pybind11/share/cmake/pybind11")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")
        pybind11_add_module(deeprbf_recovered
            "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/audited_bindings.cpp")

        target_link_libraries(deeprbf_recovered PRIVATE deeprbf_core)

        target_include_directories(deeprbf_recovered PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${EIGEN3_INCLUDE_DIR}")

        target_compile_definitions(deeprbf_recovered PRIVATE
            DEEPRBF_VERSION="${DEEPRBF_VERSION}")

        if(_HAVE_OPENMP)
            target_link_libraries(deeprbf_recovered PRIVATE OpenMP::OpenMP_CXX)
        endif()

        set_target_properties(deeprbf_recovered PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/module"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/module"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/module"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/module"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/module"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/module")
    else()
        message(FATAL_ERROR "src/bindings.cpp is required")
    endif()
endif()

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(DEEPRBF_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(FATAL_ERROR "Tests requested but tests/CMakeLists.txt is missing")
    endif()
endif()

# -----------------------------------------------------------------------------
# Install rules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
set(DEEPRBF_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Relative install directory of the compiled Python extension")

install(TARGETS deeprbf_core
    EXPORT deeprbfTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/deeprbf
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

if(DEEPRBF_BUILD_PYTHON AND TARGET deeprbf_recovered)
    install(TARGETS deeprbf_recovered
        LIBRARY DESTINATION ${DEEPRBF_PYTHON_INSTALL_DIR}
        RUNTIME DESTINATION ${DEEPRBF_PYTHON_INSTALL_DIR})
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==================== DeepRBF configuration ====================")
message(STATUS "  Version          : ${DEEPRBF_VERSION}")
message(STATUS "  C++ standard     : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen            : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  OpenMP           : ${_HAVE_OPENMP}")
message(STATUS "  MKL              : ${_HAVE_MKL}")
message(STATUS "  ScalFMM          : ${_HAVE_SCALFMM}")
message(STATUS "  Python bindings  : ${DEEPRBF_BUILD_PYTHON}")
message(STATUS "  Fast math        : ${DEEPRBF_FAST_MATH}")
message(STATUS "  Native arch      : ${DEEPRBF_NATIVE_ARCH}")
message(STATUS "  Sources          : ${DEEPRBF_SOURCES}")
message(STATUS "  Python interpreter: ${Python3_EXECUTABLE}")
message(STATUS "===============================================================")
message(STATUS "")
option(DEEPRBF_PORTABLE_TESTS "Build independent native core tests without Python or ScalFMM" ON)
if(DEEPRBF_PORTABLE_TESTS)
    enable_testing()
    add_executable(deeprbf_portable_test tests/test_core_portable.cpp)
    target_link_libraries(deeprbf_portable_test PRIVATE deeprbf_core)
    add_test(NAME deeprbf_portable_core COMMAND deeprbf_portable_test)
    add_executable(deeprbf_audited_test tests/test_audited_engine.cpp)
    target_link_libraries(deeprbf_audited_test PRIVATE deeprbf_core)
    add_test(NAME deeprbf_audited_engine COMMAND deeprbf_audited_test)
endif()

if(DEEPRBF_BUILD_TESTS)
    add_executable(deeprbf_continuous_cokriging_test tests/test_continuous_cokriging.cpp)
    target_include_directories(deeprbf_continuous_cokriging_test PRIVATE ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME deeprbf_continuous_cokriging COMMAND deeprbf_continuous_cokriging_test)
endif()

if(DEEPRBF_BUILD_TESTS)
    add_executable(deeprbf_continuous_ordinary_cokriging_test tests/test_continuous_ordinary_cokriging.cpp)
    target_include_directories(deeprbf_continuous_ordinary_cokriging_test PRIVATE ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME deeprbf_continuous_ordinary_cokriging COMMAND deeprbf_continuous_ordinary_cokriging_test)
endif()

# Release optimization must not compile away the test assertions.
if(DEEPRBF_BUILD_TESTS OR DEEPRBF_PORTABLE_TESTS)
    add_executable(deeprbf_assertions_test tests/test_assertions_enabled.cpp)
    add_test(NAME native_test_assertions_enabled COMMAND deeprbf_assertions_test)
    function(deeprbf_enable_test_assertions directory)
        get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(kind ${target} TYPE)
            if(kind STREQUAL "EXECUTABLE")
                if(MSVC)
                    target_compile_options(${target} PRIVATE /UNDEBUG)
                else()
                    target_compile_options(${target} PRIVATE -UNDEBUG)
                endif()
            endif()
        endforeach()
        get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(child IN LISTS children)
            deeprbf_enable_test_assertions("${child}")
        endforeach()
    endfunction()
    deeprbf_enable_test_assertions("${CMAKE_CURRENT_SOURCE_DIR}")
endif()
