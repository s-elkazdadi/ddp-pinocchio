# automatically enable doctest to generate ctest targets
include(../cmake/doctest.cmake)

add_library(backward OBJECT src/backward.cpp)
target_link_libraries(backward PUBLIC CONAN_PKG::backward-cpp)
add_library(doctest_main STATIC src/doctest_main.cpp)
target_link_libraries(
  doctest_main PUBLIC project_options ddp CONAN_PKG::doctest
)

set(testlibs backward doctest_main)

include_directories(../include)

add_executable(test_indexing indexing.cpp)
target_link_libraries(test_indexing PUBLIC ${testlibs})

add_executable(test_mat_seq mat_seq.cpp)
target_link_libraries(test_mat_seq PUBLIC ${testlibs})

add_library(pinocchio_impl INTERFACE)
add_library(eigen_impl INTERFACE)
add_library(ddp_impl INTERFACE)

target_link_libraries(ddp_impl INTERFACE eigen_impl)

macro(pinocchio_codegen Scalar)
  add_library(pinocchio_${Scalar} src/pinocchio_${Scalar}.cpp)
  target_link_libraries(pinocchio_${Scalar} PUBLIC ddp)
  target_link_libraries(pinocchio_impl INTERFACE pinocchio_${Scalar})
endmacro()

macro(eigen_codegen Scalar)
  add_library(eigen_${Scalar} src/eigen_${Scalar}.cpp)
  target_link_libraries(eigen_impl INTERFACE eigen_${Scalar})
endmacro()

pinocchio_codegen(double)
pinocchio_codegen(mpfr_1000)

eigen_codegen(double)
eigen_codegen(mpfr_1000)
eigen_codegen(mpfr_500)

add_executable(test_pinocchio pinocchio.cpp)
target_link_libraries(test_pinocchio PUBLIC ${testlibs} ddp_impl pinocchio_impl)

add_executable(test_pinocchio_ddp pinocchio_ddp.cpp)
target_link_libraries(
  test_pinocchio_ddp PUBLIC ${testlibs} ddp_impl pinocchio_impl
)

add_executable(test_two_arms two_arms.cpp)
target_link_libraries(test_two_arms PUBLIC ${testlibs} ddp_impl pinocchio_impl)

add_executable(test_pinocchio_spatial_eq_ddp pinocchio_spatial_eq_ddp.cpp)
target_link_libraries(
  test_pinocchio_spatial_eq_ddp PUBLIC ${testlibs} ddp_impl pinocchio_impl
)

add_executable(test_pendulum_ddp pendulum_ddp.cpp)
target_link_libraries(test_pendulum_ddp PUBLIC ${testlibs} ddp_impl)

add_executable(test_cart_ddp cart_ddp.cpp)
target_link_libraries(test_cart_ddp PUBLIC ${testlibs} ddp_impl)

doctest_discover_tests(test_indexing)
doctest_discover_tests(test_mat_seq)
doctest_discover_tests(test_pinocchio)
