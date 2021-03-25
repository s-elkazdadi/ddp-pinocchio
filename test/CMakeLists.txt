include(../cmake/doctest.cmake)

include_directories(include)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(testlibs ddp-pinocchio backward_cpp_main doctest_main)

add_library(pinocchio_double src/pinocchio_double.cpp)
add_library(pinocchio_long_double src/pinocchio_long_double.cpp)
add_library(pinocchio_mpfr src/pinocchio_mpfr.cpp)
add_library(pinocchio_mpfr2 src/pinocchio_mpfr2.cpp)
target_link_libraries(pinocchio_double ${testlibs})
target_link_libraries(pinocchio_long_double ${testlibs})
target_link_libraries(pinocchio_mpfr ${testlibs})
target_link_libraries(pinocchio_mpfr2 ${testlibs})

add_executable(test.lqr src/lqr.cpp)
add_executable(test.ur5 src/ur5.cpp)
add_executable(test.ur5.terminal src/ur5_terminal_constraint.cpp)

macro(ddp_test_config name)
  target_compile_options(test.${name} PUBLIC -g -fno-omit-frame-pointer)
  target_link_options(
    test.${name}
    PUBLIC
    -g
    -fno-omit-frame-pointer
  )
  target_link_libraries(
    test.${name}
    ddp-pinocchio
    backward_cpp_main
    pinocchio_double
  )
endmacro()

ddp_test_config(lqr)
ddp_test_config(ur5)
ddp_test_config(ur5.terminal)
