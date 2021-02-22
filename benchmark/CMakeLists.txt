include_directories(include)

add_library(nanobench_main STATIC src/nanobench_main.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)
set(benchlibs ddp-pinocchio backward_cpp_main nanobench_main)
