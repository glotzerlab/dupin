cmake_minimum_required(VERSION 3.8)
project(_dupin VERSION 0.0.1)

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(src)
