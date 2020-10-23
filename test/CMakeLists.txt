# automatically enable doctest to generate ctest targets
include(../cmake/doctest.cmake)

add_library(doctest-main STATIC doctest_main.cpp)
target_link_libraries(
  doctest-main PUBLIC project_options ddp CONAN_PKG::doctest
)
target_link_libraries(doctest-main PUBLIC project_options)

include_directories(../include)

add_executable(test-indexing indexing.cpp)
target_link_libraries(test-indexing PUBLIC doctest-main)

add_executable(test-mat-seq mat_seq.cpp)
target_link_libraries(test-mat-seq PUBLIC doctest-main)

add_library(pinocchio-general-double pinocchio_double.cpp)
add_library(pinocchio-aba-double pinocchio_double.cpp)
add_library(pinocchio-frames-double pinocchio_double.cpp)
target_link_libraries(pinocchio-general-double PUBLIC ddp)
target_link_libraries(pinocchio-aba-double PUBLIC ddp)
target_link_libraries(pinocchio-frames-double PUBLIC ddp)

target_compile_definitions(
  pinocchio-general-double PUBLIC DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio-aba-double PUBLIC DDP_PINOCCHIO_ABA)
target_compile_definitions(pinocchio-frames-double PUBLIC DDP_PINOCCHIO_FRAMES)

add_library(pinocchio-general-mpfr-1000 pinocchio_mpfr_1000.cpp)
add_library(pinocchio-aba-mpfr-1000 pinocchio_mpfr_1000.cpp)
add_library(pinocchio-frames-mpfr-1000 pinocchio_mpfr_1000.cpp)
target_link_libraries(pinocchio-general-mpfr-1000 PUBLIC ddp)
target_link_libraries(pinocchio-aba-mpfr-1000 PUBLIC ddp)
target_link_libraries(pinocchio-frames-mpfr-1000 PUBLIC ddp)

target_compile_definitions(
  pinocchio-general-mpfr-1000 PUBLIC DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio-aba-mpfr-1000 PUBLIC DDP_PINOCCHIO_ABA)
target_compile_definitions(
  pinocchio-frames-mpfr-1000 PUBLIC DDP_PINOCCHIO_FRAMES
)

add_library(pinocchio-impl INTERFACE)
target_link_libraries(
  pinocchio-impl
  INTERFACE pinocchio-general-double
            pinocchio-aba-double
            pinocchio-frames-double
            pinocchio-general-mpfr-1000
            pinocchio-aba-mpfr-1000
            pinocchio-frames-mpfr-1000
)

add_executable(test-pinocchio pinocchio.cpp)
target_link_libraries(test-pinocchio PUBLIC doctest-main pinocchio-impl)

add_executable(test-pinocchio-ddp pinocchio_ddp.cpp)
target_link_libraries(test-pinocchio-ddp PUBLIC doctest-main pinocchio-impl)

add_executable(test-pinocchio-spatial-eq-ddp pinocchio_spatial_eq_ddp.cpp)
target_link_libraries(test-pinocchio-spatial-eq-ddp PUBLIC doctest-main pinocchio-impl)

add_executable(test-pendulum-ddp pendulum_ddp.cpp)
target_link_libraries(test-pendulum-ddp PUBLIC ddp)

add_executable(test-pinocchio-jac pinocchio_jac.cpp)
target_link_libraries(test-pinocchio-jac PUBLIC doctest-main)

doctest_discover_tests(test-indexing)
doctest_discover_tests(test-mat-seq)
doctest_discover_tests(test-pinocchio)
