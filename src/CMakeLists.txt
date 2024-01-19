pybind11_add_module(_dupin dupininterface.cpp
    dupin.h dupin.cpp
)

set_target_properties(_dupin PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
)

target_include_directories(_dupin PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
)

target_link_libraries(_dupin PRIVATE TBB::tbb)
target_compile_definitions(_dupin PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_options(_dupin PRIVATE -O2 -march=native)
