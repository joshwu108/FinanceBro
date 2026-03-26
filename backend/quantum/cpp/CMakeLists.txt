cmake_minimum_required(VERSION 3.15)
project(qaoa_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(qaoa_cpp src/bindings.cpp)

target_include_directories(qaoa_cpp PRIVATE src)

# Install to the quantum/cpp directory so Python can find it
install(TARGETS qaoa_cpp LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
