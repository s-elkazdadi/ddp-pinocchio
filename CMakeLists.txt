cmake_minimum_required(VERSION 3.16)
project(ddp CXX)

include(cmake/standard_project_settings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link this 'library' to use the warnings specified in compiler_warnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/compiler_warnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/static_analyzers.cmake)

option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if(ENABLE_PCH)
  target_precompile_headers(
    project_options
    INTERFACE
    <vector>
    <string>
    <map>
    <utility>
  )
endif()

# Conan dependencies
set(CONAN_REQUIRES
    # Boost Software License 1.0
    boost/1.73.0
    range-v3/0.11.0
    #
    # Apache License 2.0
    benchmark/1.5.0
    #
    # MIT License
    doctest/2.4.0
    gsl-lite/0.37.0
    fmt/7.0.2
    spdlog/1.7.0
    backward-cpp/1.5
    #
    # Mozilla Public License 2.0
    eigen/3.3.7
    #
    # New (Revised) BSD License
    blaze/3.7
    #
    # LGPL v3.0
    mpfr/4.0.2
)
target_compile_definitions(
  project_options INTERFACE SPDLOG_FMT_EXTERNAL gsl_CONFIG_DEFAULTS_VERSION=1
)
set(CONAN_OPTIONS)

include(cmake/conan.cmake)
run_conan()

find_package(PkgConfig REQUIRED)
pkg_check_modules(
  urdfdom
  REQUIRED
  IMPORTED_TARGET
  urdfdom
)
pkg_check_modules(
  pinocchio
  REQUIRED
  IMPORTED_TARGET
  pinocchio
)

add_library(ddp src/lib.cpp)
target_link_libraries(
  ddp
  PUBLIC project_options
         project_warnings
         CONAN_PKG::boost
         CONAN_PKG::benchmark
         CONAN_PKG::doctest
         CONAN_PKG::gsl-lite
         CONAN_PKG::fmt
         CONAN_PKG::spdlog
         CONAN_PKG::backward-cpp
         CONAN_PKG::eigen
         CONAN_PKG::mpfr
         PkgConfig::urdfdom
         PkgConfig::pinocchio
)
target_include_directories(ddp PUBLIC include)

add_library(pinocchio-double test/pinocchio_double.cpp)
add_library(pinocchio-mpfr-500 test/pinocchio_mpfr_500.cpp)
target_link_libraries(pinocchio-double PUBLIC ddp)
target_link_libraries(pinocchio-mpfr-500 PUBLIC ddp)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
