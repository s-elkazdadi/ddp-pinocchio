cmake_minimum_required(VERSION 3.16)
project(ddp-pinocchio CXX)

get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
  set(top_level FALSE)
else()
  set(top_level TRUE)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(ENABLE_DOCS_ONLY)
  add_subdirectory("docs")
  return()
endif()

if(top_level AND ENABLE_TESTING)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(cmake/compiler_warnings.cmake)
include(cmake/static_analyzers.cmake)
include(cmake/sanitizers.cmake)
include(cmake/conan.cmake)

add_library(ddp-pinocchio src/ddp.cpp)
target_include_directories(
  ddp-pinocchio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ddp-pinocchio PUBLIC mpfr-cxx veg)

# Conan dependencies
set(CONAN_REQUIRES
    # Boost Software License 1.0
    boost/1.74.0
    # MIT License
    gsl-lite/0.37.0
    fmt/7.1.3
    spdlog/1.8.2
    backward-cpp/1.5
    #
    # Mozilla Public License 2.0
    eigen/3.3.7
    #
    # New (Revised) BSD License
    blaze/3.8
    #
    # LGPL v3
    mpfr/4.1.0
)
target_compile_definitions(
  ddp-pinocchio PUBLIC SPDLOG_FMT_EXTERNAL gsl_CONFIG_DEFAULTS_VERSION=1
)
set(CONAN_OPTIONS)
run_conan()

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
pkg_check_modules(
  example-robot-data
  REQUIRED
  IMPORTED_TARGET
  example-robot-data
)
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

pkg_check_modules(
  hpp-fcl
  REQUIRED
  IMPORTED_TARGET
  hpp-fcl
)

target_link_libraries(
  ddp-pinocchio
  PUBLIC CONAN_PKG::boost
         CONAN_PKG::gsl-lite
         CONAN_PKG::fmt
         CONAN_PKG::spdlog
         CONAN_PKG::eigen
         CONAN_PKG::blaze
         CONAN_PKG::mpfr
         PkgConfig::urdfdom
         PkgConfig::hpp-fcl
         PkgConfig::pinocchio
         PkgConfig::example-robot-data
         OpenMP::OpenMP_CXX
)

if(top_level AND ENABLE_TESTING)
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)
  add_subdirectory(external/mpfr-cxx)
  add_subdirectory(external/veg)

  target_link_libraries(
    ddp-pinocchio INTERFACE project_options project_warnings
  )

  include(cmake/extra_local_settings.cmake)

  set_project_warnings(project_warnings)

  enable_sanitizers(project_options)
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

  enable_testing()
  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()
