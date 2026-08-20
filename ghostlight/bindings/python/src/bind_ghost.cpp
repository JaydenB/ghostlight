#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ghost.h"

namespace py = pybind11;
using namespace py::literals;

void bind_ghost(py::module_& m)
{
    // --------------------------------------------------------------- GhostPair
    py::class_<GhostPair>(m, "GhostPair")
        .def(py::init<>())
        .def(py::init([](int a, int b) { return GhostPair{a, b}; }),
             "surf_a"_a, "surf_b"_a)
        .def_readwrite("surf_a", &GhostPair::surf_a)
        .def_readwrite("surf_b", &GhostPair::surf_b)
        .def("__repr__", [](const GhostPair& p) {
            return "GhostPair(" + std::to_string(p.surf_a)
                + ", " + std::to_string(p.surf_b) + ")";
        });

    // ------------------------------------------------- enumerate_ghost_pairs
    m.def("enumerate_ghost_pairs", &enumerate_ghost_pairs, "lens"_a);

    // --------------------------------------------------- filter_ghost_pairs
    // The C++ API uses two output parameters; here we return a (pairs, boosts) tuple.
    m.def("filter_ghost_pairs",
        [](const OpticalSystem& lens, float hw, float hh, const FlareConfig& cfg) {
            std::vector<GhostPair> pairs;
            std::vector<float>     boosts;
            filter_ghost_pairs(lens, hw, hh, cfg, pairs, boosts);
            return py::make_tuple(std::move(pairs), std::move(boosts));
        },
        "lens"_a, "sensor_half_w"_a, "sensor_half_h"_a, "config"_a);
}
