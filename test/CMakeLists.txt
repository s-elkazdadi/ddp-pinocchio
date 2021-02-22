include(../cmake/doctest.cmake)

include_directories(include)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(testlibs ddp-pinocchio backward_cpp_main doctest_main)

add_library(pinocchio_double src/pinocchio_double.cpp)
target_link_libraries(pinocchio_double ${testlibs})

add_executable(test_main src/test_main.cpp)
target_link_libraries(
  test_main
  ddp-pinocchio
  pinocchio_double
  backward_cpp_main
)
