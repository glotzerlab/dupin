#include "dupin.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_dupin, m) {
  py::class_<DynamicProgramming>(m, "DynamicProgramming")
      .def(py::init<>())
      .def_property("cost_matrix", &DynamicProgramming::getCostMatrix,
                    &DynamicProgramming::setCostMatrix)
      .def("fit", &DynamicProgramming::fit)
      .def("set_threads", &DynamicProgramming::set_parallelization);
}
