#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "dupin.h"

namespace py = pybind11;

PYBIND11_MODULE(_dupin, m) {
    py::class_<DynamicProgramming>(m, "DynamicProgramming")
        .def_property("datum", &DynamicProgramming::getDatum, &DynamicProgramming::setDatum)
        .def_property("cost_matrix", &DynamicProgramming::getCostMatrix, &DynamicProgramming::setCostMatrix)
        .def_property("num_bkps", &DynamicProgramming::get_num_bkps, &DynamicProgramming::set_num_bkps)
        .def_property("num_timesteps", &DynamicProgramming::get_num_timesteps, &DynamicProgramming::set_num_timesteps)
        .def_property("num_parameters", &DynamicProgramming::get_num_parameters, &DynamicProgramming::set_num_parameters)
        .def(py::init<>()) 
        .def("read_input", &DynamicProgramming::read_input)
        .def("initialize_cost_matrix", &DynamicProgramming::initialize_cost_matrix)
        .def("return_breakpoints", &DynamicProgramming::return_breakpoints)
        .def("set_threads", &DynamicProgramming::set_parallelization); 
}
