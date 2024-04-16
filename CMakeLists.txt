cmake_minimum_required(VERSION 3.8)
project(dupin VERSION 0.0.1 LANGUAGES CXX)

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
# Use modern method for Python binding
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(dupin)
