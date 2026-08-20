#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_geometry(py::module_&);
void bind_trace(py::module_&);
void bind_calibration(py::module_&);
void bind_ghost(py::module_&);
void bind_config(py::module_&);
void bind_renderers(py::module_&);

PYBIND11_MODULE(_ghostlight, m)
{
    m.doc() = "Ghostlight Python bindings";

    // Order matters: config depends on spectral enums; ghost depends on config;
    // renderers depend on all of the above.
    bind_geometry(m);
    bind_trace(m);
    bind_calibration(m);
    bind_config(m);
    bind_ghost(m);
    bind_renderers(m);
}
