# automatically enable doctest to generate ctest targets
include(../cmake/doctest.cmake)

add_library(doctest-main STATIC doctest_main.cpp)
target_link_libraries(
  doctest-main
  PUBLIC project_options
         ddp
         CONAN_PKG::doctest
         pinocchio-double
         pinocchio-mpfr-500
)
target_link_libraries(doctest-main PRIVATE project_options)

include_directories(../include)

add_executable(test-indexing indexing.cpp)
target_link_libraries(test-indexing PRIVATE doctest-main)

add_executable(test-mat-seq mat_seq.cpp)
target_link_libraries(test-mat-seq PRIVATE doctest-main)

add_executable(test-pinocchio pinocchio.cpp)
target_link_libraries(test-pinocchio PRIVATE doctest-main)

add_executable(test-pinocchio-ddp pinocchio_ddp.cpp)
target_link_libraries(test-pinocchio-ddp PRIVATE doctest-main)

doctest_discover_tests(test-indexing)
doctest_discover_tests(test-mat-seq)
doctest_discover_tests(test-pinocchio)
