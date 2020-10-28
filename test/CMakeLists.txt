# automatically enable doctest to generate ctest targets
include(../cmake/doctest.cmake)

add_library(backward OBJECT backward.cpp)
target_link_libraries(backward PUBLIC CONAN_PKG::backward-cpp)
add_library(doctest_main STATIC doctest_main.cpp)
target_link_libraries(
  doctest_main PUBLIC project_options ddp CONAN_PKG::doctest
)

set(testlibs backward doctest_main)

include_directories(../include)

add_executable(test_indexing indexing.cpp)
target_link_libraries(test_indexing PUBLIC ${testlibs})

add_executable(test_mat_seq mat_seq.cpp)
target_link_libraries(test_mat_seq PUBLIC ${testlibs})

add_library(pinocchio_general_double pinocchio_double.cpp)
add_library(pinocchio_aba_double pinocchio_double.cpp)
add_library(pinocchio_frames_double pinocchio_double.cpp)
target_link_libraries(pinocchio_general_double PUBLIC ddp)
target_link_libraries(pinocchio_aba_double PUBLIC ddp)
target_link_libraries(pinocchio_frames_double PUBLIC ddp)

target_compile_definitions(
  pinocchio_general_double PUBLIC DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio_aba_double PUBLIC DDP_PINOCCHIO_ABA)
target_compile_definitions(pinocchio_frames_double PUBLIC DDP_PINOCCHIO_FRAMES)

add_library(pinocchio_general_mpfr_1000 pinocchio_mpfr_1000.cpp)
add_library(pinocchio_aba_mpfr_1000 pinocchio_mpfr_1000.cpp)
add_library(pinocchio_frames_mpfr_1000 pinocchio_mpfr_1000.cpp)
target_link_libraries(pinocchio_general_mpfr_1000 PUBLIC ddp)
target_link_libraries(pinocchio_aba_mpfr_1000 PUBLIC ddp)
target_link_libraries(pinocchio_frames_mpfr_1000 PUBLIC ddp)

target_compile_definitions(
  pinocchio_general_mpfr_1000 PUBLIC DDP_PINOCCHIO_GENERAL
)
target_compile_definitions(pinocchio_aba_mpfr_1000 PUBLIC DDP_PINOCCHIO_ABA)
target_compile_definitions(
  pinocchio_frames_mpfr_1000 PUBLIC DDP_PINOCCHIO_FRAMES
)

add_library(pinocchio_impl INTERFACE)
target_link_libraries(
  pinocchio_impl
  INTERFACE pinocchio_general_double
            pinocchio_aba_double
            pinocchio_frames_double
            pinocchio_general_mpfr_1000
            pinocchio_aba_mpfr_1000
            pinocchio_frames_mpfr_1000
)

add_executable(test_pinocchio pinocchio.cpp)
target_link_libraries(test_pinocchio PUBLIC ${testlibs} pinocchio_impl)

add_executable(test_pinocchio_ddp pinocchio_ddp.cpp)
target_link_libraries(test_pinocchio_ddp PUBLIC ${testlibs} pinocchio_impl)

add_executable(test_pinocchio_spatial_eq_ddp pinocchio_spatial_eq_ddp.cpp)
target_link_libraries(
  test_pinocchio_spatial_eq_ddp PUBLIC ${testlibs} pinocchio_impl
)

add_executable(test_pendulum_ddp pendulum_ddp.cpp)
target_link_libraries(test_pendulum_ddp PUBLIC ddp)

add_executable(test_cart_ddp cart_ddp.cpp)
target_link_libraries(test_cart_ddp PUBLIC ddp)

doctest_discover_tests(test_indexing)
doctest_discover_tests(test_mat_seq)
doctest_discover_tests(test_pinocchio)
