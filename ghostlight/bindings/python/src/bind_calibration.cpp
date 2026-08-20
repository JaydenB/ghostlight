#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lens_calibration.h"

namespace py = pybind11;
using namespace py::literals;

void bind_calibration(py::module_& m)
{
    // -------------------------------------------------------- LensCalibration
    py::class_<LensCalibration>(m, "LensCalibration")
        .def_readonly("sensor_half_w",    &LensCalibration::sensor_half_w)
        .def_readonly("sensor_half_h",    &LensCalibration::sensor_half_h)
        .def_readonly("max_half_angle_h", &LensCalibration::max_half_angle_h)
        .def_readonly("max_half_angle_v", &LensCalibration::max_half_angle_v)
        // The illuminated edge — a different question from the covered field
        // above, which is the onset of vignetting. Ask for this one when you
        // mean "how much of the frame does the lens cover".
        .def_readonly("image_circle_semi_w", &LensCalibration::image_circle_semi_w)
        .def_readonly("image_circle_semi_h", &LensCalibration::image_circle_semi_h)
        // First-order optics (per axis; 0 = solve failed). Used by the
        // diffraction starburst to scale the pattern physically.
        .def_readonly("focal_length_x",        &LensCalibration::focal_length_x)
        .def_readonly("focal_length_y",        &LensCalibration::focal_length_y)
        .def_readonly("entrance_pupil_semi_x", &LensCalibration::entrance_pupil_semi_x)
        .def_readonly("entrance_pupil_semi_y", &LensCalibration::entrance_pupil_semi_y)
        .def_readonly("f_number_x",            &LensCalibration::f_number_x)
        .def_readonly("f_number_y",            &LensCalibration::f_number_y)
        // How much of the bound the f-number describes the stop actually fills
        // (1.0 unless the stop is bladed) — the starburst's light-collection
        // term multiplies by it.
        .def_readonly("pupil_area_frac",       &LensCalibration::pupil_area_frac)
        .def("__repr__", [](const LensCalibration& c) {
            return "LensCalibration(sensor_half=(" + std::to_string(c.sensor_half_w)
                + "," + std::to_string(c.sensor_half_h) + "))";
        });

    // -------------------------------------------------------- calibrate_lens
    m.def("calibrate_lens", &calibrate_lens,
        "lens"_a, "d_line_nm"_a = 587.56f);

    // --------------------------------------------------- build_spectral_lambdas
    m.def("build_spectral_lambdas", &build_spectral_lambdas,
        "n_samples"_a, "lambda_min"_a = 400.0f, "lambda_max"_a = 700.0f);
}
