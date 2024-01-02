cmake_minimum_required(VERSION 3.8)
project(_dupin VERSION 1.0.0)  # Replace with your actual version

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
add_subdirectory(extern/pybind11)

pybind11_add_module(_dupin src/cpp/dupin.h src/cpp/dupin.cpp src/interface/dupininterface.cpp)

target_include_directories(_dupin PRIVATE 
    ${EIGEN3_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0"
)

target_link_libraries(_dupin PRIVATE TBB::tbb)
target_compile_definitions(_dupin PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_options(_dupin PRIVATE -O2 -march=native)

set_target_properties(_dupin PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
)
