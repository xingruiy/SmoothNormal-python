cmake_minimum_required(VERSION 3.4...3.18)
project(test_normal CXX)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_libraries(${PCL_LIBRARIES})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(pybind11)
pybind11_add_module(smooth_normal main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(smooth_normal PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})