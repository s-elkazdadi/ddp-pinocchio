# automatically enable doctest to generate ctest targets
include(../cmake/doctest.cmake)

add_library(doctest-main STATIC doctest_main.cpp)
target_link_libraries(
  doctest-main PUBLIC project_options ddp CONAN_PKG::doctest
)
target_link_libraries(doctest-main PRIVATE project_options)

include_directories(../include)

add_executable(test-indexing indexing.cpp)
target_link_libraries(test-indexing PRIVATE doctest-main)

add_executable(test-mat-seq mat_seq.cpp)
target_link_libraries(test-mat-seq PRIVATE doctest-main)

add_library(pinocchio-general-double pinocchio_double.cpp)
add_library(pinocchio-aba-double pinocchio_double.cpp)
add_library(pinocchio-frames-double pinocchio_double.cpp)

target_compile_definitions(
  pinocchio-general-double PRIVATE DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio-aba-double PRIVATE DDP_PINOCCHIO_ABA)
target_compile_definitions(pinocchio-frames-double PRIVATE DDP_PINOCCHIO_FRAMES)

add_library(pinocchio-general-mpfr-500 pinocchio_mpfr_500.cpp)
add_library(pinocchio-aba-mpfr-500 pinocchio_mpfr_500.cpp)
add_library(pinocchio-frames-mpfr-500 pinocchio_mpfr_500.cpp)

target_compile_definitions(
  pinocchio-general-mpfr-500 PRIVATE DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio-aba-mpfr-500 PRIVATE DDP_PINOCCHIO_ABA)
target_compile_definitions(
  pinocchio-frames-mpfr-500 PRIVATE DDP_PINOCCHIO_FRAMES
)

add_library(pinocchio-impl INTERFACE)
target_link_libraries(
  pinocchio-impl
  INTERFACE pinocchio-general-double
            pinocchio-aba-double
            pinocchio-frames-double
            pinocchio-general-mpfr-500
            pinocchio-aba-mpfr-500
            pinocchio-frames-mpfr-500
)

add_executable(test-pinocchio pinocchio.cpp)
target_link_libraries(test-pinocchio PRIVATE doctest-main pinocchio-impl)

add_executable(test-pinocchio-ddp pinocchio_ddp.cpp)
target_link_libraries(test-pinocchio-ddp PRIVATE doctest-main pinocchio-impl)

add_executable(test-pendulum-ddp pendulum_ddp.cpp)
target_link_libraries(test-pendulum-ddp PRIVATE ddp)

add_executable(test-pinocchio-jac pinocchio_jac.cpp)
target_link_libraries(test-pinocchio-jac PRIVATE doctest-main)

doctest_discover_tests(test-indexing)
doctest_discover_tests(test-mat-seq)
doctest_discover_tests(test-pinocchio)
