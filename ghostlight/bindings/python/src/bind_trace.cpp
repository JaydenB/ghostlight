#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "trace.h"
#include "trace_event.h"

namespace py = pybind11;
using namespace py::literals;

void bind_trace(py::module_& m)
{
    // ----------------------------------------------------------- TraceStatus
    py::enum_<TraceStatus>(m, "TraceStatus")
        .value("OK",             TraceStatus::OK)
        .value("VIGNETTED",      TraceStatus::VIGNETTED)
        .value("TIR",            TraceStatus::TIR)
        .value("MISSED_SURFACE", TraceStatus::MISSED_SURFACE)
        .value("INVALID_INPUT",  TraceStatus::INVALID_INPUT)
        .export_values();

    // ----------------------------------------------------------- TraceResult
    py::class_<TraceResult>(m, "TraceResult")
        .def_readonly("position", &TraceResult::position)
        .def_readonly("weight",   &TraceResult::weight)
        .def_readonly("status",   &TraceResult::status)
        .def("__repr__", [](const TraceResult& r) {
            return "TraceResult(status=" + std::to_string((int)r.status)
                + ", weight=" + std::to_string(r.weight) + ")";
        });

    // ----------------------------------------------------------- TraceEvent
    py::class_<TraceEvent>(m, "TraceEvent")
        .def_readonly("surface_index",  &TraceEvent::surface_index)
        .def_readonly("hit_point",      &TraceEvent::hit_point)
        .def_readonly("surface_normal", &TraceEvent::surface_normal)
        .def_readonly("ior_before",     &TraceEvent::ior_before)
        .def_readonly("ior_after",      &TraceEvent::ior_after)
        .def_readonly("fresnel_weight", &TraceEvent::fresnel_weight)
        .def_readonly("status",         &TraceEvent::status)
        .def_readonly("reflected",      &TraceEvent::reflected)
        .def("__repr__", [](const TraceEvent& e) {
            return "TraceEvent(surface=" + std::to_string(e.surface_index)
                + ", reflected=" + (e.reflected ? "True" : "False")
                + ", w=" + std::to_string(e.fresnel_weight) + ")";
        });

    // ------------------------------------------------------------- RayPath
    py::class_<RayPath>(m, "RayPath")
        .def_readonly("events", &RayPath::events)
        .def_readonly("result", &RayPath::result)
        .def("__repr__", [](const RayPath& p) {
            return "RayPath(" + std::to_string(p.events.size()) + " events)";
        });

    // ----------------------------------------- Free tracing functions
    // Fast non-diagnostic overloads
    m.def("trace_ghost_ray",
        [](const Ray& ray, const OpticalSystem& lens, int a, int b) {
            if (a < 0 || b < 0 || a >= b || b >= lens.num_surfaces())
                throw py::value_error("bounce indices must satisfy 0 <= bounce_a < bounce_b < num_surfaces");
            return trace_ghost_ray(ray, lens, a, b);
        },
        "ray"_a, "lens"_a, "bounce_a"_a, "bounce_b"_a);

    m.def("trace_primary_ray",
        py::overload_cast<const Ray&, const OpticalSystem&>(&trace_primary_ray),
        "ray"_a, "lens"_a);

    // Diagnostic overloads — allocate RayPath internally so Python callers
    // don't deal with output-reference idioms.
    m.def("trace_ghost_ray_diagnostic",
        [](const Ray& r, const OpticalSystem& l, int a, int b) {
            if (a < 0 || b < 0 || a >= b || b >= l.num_surfaces())
                throw py::value_error("bounce indices must satisfy 0 <= bounce_a < bounce_b < num_surfaces");
            RayPath path;
            trace_ghost_ray(r, l, a, b, path);
            return path;
        },
        "ray"_a, "lens"_a, "bounce_a"_a, "bounce_b"_a);

    m.def("trace_primary_ray_diagnostic",
        [](const Ray& r, const OpticalSystem& l) {
            RayPath path;
            trace_primary_ray(r, l, path);
            return path;
        },
        "ray"_a, "lens"_a);

    // HURB's per-surface geometry input, exposed so its edge distance can be
    // tested directly rather than inferred from a kicked render. Returns the
    // perpendicular distance (mm) from a world-space point to the surface's
    // nearest clear-aperture edge, plus the world-space edge normal there.
    m.def("_aperture_edge_distance_debug",
        [](const Vec3f& hit, const Surface& s) {
            Vec3f n;
            const float d = aperture_edge_distance(hit, s, n);
            return py::make_tuple(d, n);
        },
        "hit"_a, "surface"_a);
}
