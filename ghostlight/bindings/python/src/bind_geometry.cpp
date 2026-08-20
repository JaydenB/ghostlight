#include "vec3.h"
#include "ray.h"
#include "optical_system.h"

#include <pybind11/pybind11.h>

// Declare std::vector<Surface> and std::vector<std::string> opaque BEFORE
// stl.h registers automatic list<->vector conversion casters.  Without this,
// stl.h's casters shadow bind_vector's registered types, causing the
// `surfaces` property getter to return a Python list copy instead of a live
// SurfaceList reference — so mutations (append, __setitem__) wouldn't
// propagate back to the C++ vector.
PYBIND11_MAKE_OPAQUE(std::vector<Surface>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<ApertureImage>);

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;
using namespace py::literals;

void bind_geometry(py::module_& m)
{
    // ------------------------------------------------------------------ Vec3f
    py::class_<Vec3f>(m, "Vec3f")
        .def(py::init<>())
        .def(py::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def(py::init<float>(), "s"_a)
        .def_readwrite("x", &Vec3f::x)
        .def_readwrite("y", &Vec3f::y)
        .def_readwrite("z", &Vec3f::z)
        .def("__getitem__", [](const Vec3f& v, int i) {
            if (i < 0 || i > 2) throw py::index_error();
            return v[i];
        })
        .def("__setitem__", [](Vec3f& v, int i, float val) {
            if (i < 0 || i > 2) throw py::index_error();
            v[i] = val;
        })
        .def("__iter__", [](const Vec3f& v) {
            return py::make_iterator(&v.x, &v.x + 3);
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Vec3f&) { return 3; })
        .def("__repr__", [](const Vec3f& v) {
            return "Vec3f(" + std::to_string(v.x) + ", "
                            + std::to_string(v.y) + ", "
                            + std::to_string(v.z) + ")";
        })
        .def("length",    &Vec3f::length)
        .def("length_sq", &Vec3f::length_sq)
        .def("normalized",&Vec3f::normalized)
        .def("__add__", [](const Vec3f& a, const Vec3f& b) { return a + b; })
        .def("__sub__", [](const Vec3f& a, const Vec3f& b) { return a - b; })
        .def("__mul__", [](const Vec3f& a, float s) { return a * s; })
        .def("__rmul__",[](const Vec3f& a, float s) { return a * s; })
        .def("__truediv__", [](const Vec3f& a, float s) { return a / s; })
        .def("__neg__", [](const Vec3f& a) { return -a; });

    // --------------------------------------------------------------------- Ray
    py::class_<Ray>(m, "Ray")
        .def(py::init<>())
        .def(py::init([](Vec3f o, Vec3f d, float lam) {
            return Ray{o, d, lam};
        }), "origin"_a, "dir"_a, "wavelength"_a = 587.56f)
        .def_readwrite("origin",     &Ray::origin)
        .def_readwrite("dir",        &Ray::dir)
        .def_readwrite("wavelength", &Ray::lambda)  // "lambda" is a Python keyword
        .def("__repr__", [](const Ray& r) {
            return "Ray(origin=Vec3f(" + std::to_string(r.origin.x) + ","
                + std::to_string(r.origin.y) + "," + std::to_string(r.origin.z)
                + "), dir=Vec3f(" + std::to_string(r.dir.x) + ","
                + std::to_string(r.dir.y) + "," + std::to_string(r.dir.z)
                + "), wavelength=" + std::to_string(r.lambda) + ")";
        });

    // --------------------------------------------------------------- Constants
    // Shared format and geometry limits for Python serializers.
    m.attr("LENS_FORMAT_MAJOR") = py::int_(LENS_FORMAT_MAJOR);
    m.attr("LENS_FORMAT_MINOR") = py::int_(LENS_FORMAT_MINOR);
    m.attr("MAX_ASPHERE_TERMS") = py::int_(MAX_ASPHERE_TERMS);

    // ------------------------------------------------------------------- Enums
    py::enum_<CoatingModel>(m, "CoatingModel")
        .value("SIMPLE",           CoatingModel::SIMPLE)
        .value("SPECTRAL",         CoatingModel::SPECTRAL)
        .value("ANGULAR",          CoatingModel::ANGULAR)
        .value("SPECTRAL_ANGULAR", CoatingModel::SPECTRAL_ANGULAR)
        .value("ATTENUATOR_GAUSS", CoatingModel::ATTENUATOR_GAUSS)
        .value("ARTIST",           CoatingModel::ARTIST)
        .export_values();

    py::enum_<SurfaceForm>(m, "SurfaceForm")
        .value("SPHERE",      FORM_SPHERE)
        .value("ASPHERE",     FORM_ASPHERE)
        .value("CYLINDRICAL", FORM_CYLINDRICAL)
        .export_values();

    py::enum_<DispersionModel>(m, "DispersionModel")
        .value("AIR",       DISP_AIR)
        .value("ABBE",      DISP_ABBE)
        .value("SELLMEIER", DISP_SELLMEIER)
        .export_values();

    py::enum_<CylinderAxis>(m, "CylinderAxis")
        .value("AXIS_X", CYL_AXIS_X)
        .value("AXIS_Y", CYL_AXIS_Y)
        .export_values();

    py::enum_<ApertureShape>(m, "ApertureShape")
        .value("CIRCLE",  APERTURE_CIRCLE)
        .value("POLYGON", APERTURE_POLYGON)
        .value("IMAGE",   APERTURE_IMAGE)
        .export_values();

    // ---------------------------------------------------------- ApertureProfile
    // Read-only derived geometry; authored-field setters refresh it.
    py::class_<ApertureProfile>(m, "ApertureProfile")
        .def_readonly("blades",    &ApertureProfile::blades)
        .def_readonly("plain",     &ApertureProfile::plain)
        .def_readonly("facets",    &ApertureProfile::facets)
        .def_readonly("rotation",  &ApertureProfile::rotation)
        .def_readonly("sigma",     &ApertureProfile::sigma)
        .def_readonly("half",      &ApertureProfile::half)
        .def_readonly("r_w",       &ApertureProfile::r_w)
        .def_readonly("phi_w",     &ApertureProfile::phi_w)
        .def_readonly("p",         &ApertureProfile::p)
        .def_readonly("beta",      &ApertureProfile::beta)
        .def_readonly("eA",        &ApertureProfile::eA)
        .def_readonly("eB",        &ApertureProfile::eB)
        .def_readonly("area_frac", &ApertureProfile::area_frac)
        .def("deformed",   &ApertureProfile::deformed)
        .def("radius_at",  &ApertureProfile::radius_at, py::arg("theta"),
             "Boundary radius at absolute angle theta; 1.0 is the blade tip.")
        .def("dr_dtheta",  &ApertureProfile::dr_dtheta, py::arg("theta"),
             "Analytic slope of radius_at, as HURB's edge distance uses it.");

    // ------------------------------------------------------------------ Coating
    py::class_<Coating>(m, "Coating")
        .def(py::init<>())
        .def_readwrite("model",            &Coating::model)
        .def_readwrite("ar_layers",        &Coating::ar_layers)
        .def_readwrite("gauss_sigma",      &Coating::gauss_sigma)
        .def_readwrite("gauss_background", &Coating::gauss_background)
        .def_readwrite("gauss_peak",       &Coating::gauss_peak)
        .def_readwrite("gauss_decenter_x", &Coating::gauss_decenter_x)
        .def_readwrite("gauss_decenter_y", &Coating::gauss_decenter_y)
        .def_readwrite("angle_ref_ior",    &Coating::angle_ref_ior)
        .def_readwrite("tint_r",           &Coating::tint_r)
        .def_readwrite("tint_g",           &Coating::tint_g)
        .def_readwrite("tint_b",           &Coating::tint_b)
        .def_readwrite("tint_strength",    &Coating::tint_strength)
        .def_readwrite("out_of_range_discard", &Coating::out_of_range_discard)
        // Table sizes are readable for introspection; the table CONTENTS are
        // owned by OpticalSystem.coating_tables and edited through the
        // _OpticalSystem.set_coating_* accessors (never via raw pointers).
        .def_readonly("table_count",       &Coating::table_count)
        .def_readonly("sa_n_wavelengths",  &Coating::sa_n_wavelengths)
        .def_readonly("sa_n_angles",       &Coating::sa_n_angles);

    // ------------------------------------------------------------------ Surface
    py::class_<Surface>(m, "Surface")
        .def(py::init([]() {
            Surface s{};
            s.radius    = 0.0f;
            s.thickness = 0.0f;
            s.ior       = 1.0f;
            s.abbe_v    = 0.0f;
            s.semi_aperture = 10.0f;
            s.is_stop   = false;
            s.is_active = true;
            s.z         = 0.0f;
            s.aperture_shape         = APERTURE_CIRCLE;
            s.aperture_blades        = 0;
            s.aperture_rotation_rad  = 0.0f;
            s.aperture_aspect        = 1.0f;
            s.aperture_semi_diameter = 0.0f;
            s.aperture_curvature       = 0.0f;
            s.aperture_twist           = 0.0f;
            s.aperture_notch_rad       = 0.0f;
            s.aperture_notch_angle_rad = 0.0f;
            s.refresh_aperture_profile();
            s.decenter_x = 0.0f;
            s.decenter_y = 0.0f;
            // rot = identity
            s.rot[0]=1.f; s.rot[1]=0.f; s.rot[2]=0.f;
            s.rot[3]=0.f; s.rot[4]=1.f; s.rot[5]=0.f;
            s.rot[6]=0.f; s.rot[7]=0.f; s.rot[8]=1.f;
            s.form      = FORM_SPHERE;
            s.conic_k   = 0.0f;
            s.n_asphere_terms = 0;
            s.cyl_axis  = CYL_AXIS_X;
            s.disp_model = DISP_AIR;
            for (int i = 0; i < MAX_ASPHERE_TERMS; ++i) s.asphere_terms[i] = 0.0f;
            for (int i = 0; i < 3; ++i) { s.sellmeier_B[i] = 0.0f; s.sellmeier_C[i] = 0.0f; }
            return s;
        }))
        .def_readwrite("radius",       &Surface::radius)
        .def_readwrite("thickness",    &Surface::thickness)
        .def_readwrite("ior",          &Surface::ior)
        .def_readwrite("abbe_v",       &Surface::abbe_v)
        .def_readwrite("semi_aperture",&Surface::semi_aperture)
        .def_readwrite("coating",      &Surface::coating)
        .def_readwrite("is_stop",      &Surface::is_stop)
        .def_readwrite("is_active",    &Surface::is_active)
        .def_readwrite("z",            &Surface::z)
        // Rebuild the derived blade profile whenever its authored inputs change.
        .def_property("aperture_shape",
            [](const Surface& s) { return s.aperture_shape; },
            [](Surface& s, int v) { s.aperture_shape = v; s.refresh_aperture_profile(); })
        .def_property("aperture_blades",
            [](const Surface& s) { return s.aperture_blades; },
            [](Surface& s, int v) { s.aperture_blades = v; s.refresh_aperture_profile(); })
        .def_property("aperture_rotation_rad",
            [](const Surface& s) { return s.aperture_rotation_rad; },
            [](Surface& s, float v) { s.aperture_rotation_rad = v; s.refresh_aperture_profile(); })
        .def_property("aperture_curvature",
            [](const Surface& s) { return s.aperture_curvature; },
            [](Surface& s, float v) { s.aperture_curvature = v; s.refresh_aperture_profile(); })
        .def_property("aperture_twist",
            [](const Surface& s) { return s.aperture_twist; },
            [](Surface& s, float v) { s.aperture_twist = v; s.refresh_aperture_profile(); })
        .def_property("aperture_notch_rad",
            [](const Surface& s) { return s.aperture_notch_rad; },
            [](Surface& s, float v) { s.aperture_notch_rad = v; s.refresh_aperture_profile(); })
        .def_property("aperture_notch_angle_rad",
            [](const Surface& s) { return s.aperture_notch_angle_rad; },
            [](Surface& s, float v) { s.aperture_notch_angle_rad = v; s.refresh_aperture_profile(); })
        .def_readonly("aperture_profile",        &Surface::aperture_profile)
        .def_readwrite("aperture_aspect",        &Surface::aperture_aspect)
        .def_readwrite("aperture_semi_diameter", &Surface::aperture_semi_diameter)
        .def_readwrite("decenter_x",   &Surface::decenter_x)
        .def_readwrite("decenter_y",   &Surface::decenter_y)
        // rot: row-major 3×3 local→world rotation, exposed as a 9-element numpy array.
        .def_property("rot",
            [](py::object self) {
                auto& s = py::cast<Surface&>(self);
                return py::array_t<float>({9}, {sizeof(float)}, s.rot, self);
            },
            [](Surface& s, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() != 9)
                    throw py::value_error("rot must have exactly 9 elements (row-major 3x3)");
                std::copy_n(arr.data(), 9, s.rot);
            })
        .def_readwrite("form",         &Surface::form)
        .def_readwrite("conic_k",      &Surface::conic_k)
        .def_property("n_asphere_terms",
            [](const Surface& s) { return s.n_asphere_terms; },
            [](Surface& s, int n) {
                if (n < 0 || n > MAX_ASPHERE_TERMS)
                    throw py::value_error("n_asphere_terms must be between 0 and "
                                          + std::to_string(MAX_ASPHERE_TERMS));
                s.n_asphere_terms = n;
            })
        .def_readwrite("cyl_axis",     &Surface::cyl_axis)
        .def_readwrite("disp_model",   &Surface::disp_model)
        // Fixed arrays exposed via numpy properties
        .def_property("asphere_terms",
            [](py::object self) {
                auto& s = py::cast<Surface&>(self);
                return py::array_t<float>(
                    {(py::ssize_t)std::max(0, std::min(s.n_asphere_terms,
                                                       MAX_ASPHERE_TERMS))},
                    {(py::ssize_t)sizeof(float)},
                    s.asphere_terms, self);
            },
            [](Surface& s, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() > MAX_ASPHERE_TERMS)
                    throw py::value_error("asphere_terms: max " + std::to_string(MAX_ASPHERE_TERMS) + " terms");
                s.n_asphere_terms = (int)arr.size();
                std::copy_n(arr.data(), arr.size(), s.asphere_terms);
            })
        .def_property("sellmeier_B",
            [](py::object self) {
                auto& s = py::cast<Surface&>(self);
                return py::array_t<float>({3}, {sizeof(float)}, s.sellmeier_B, self);
            },
            [](Surface& s, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() != 3) throw py::value_error("sellmeier_B must have exactly 3 elements");
                std::copy_n(arr.data(), 3, s.sellmeier_B);
            })
        .def_property("sellmeier_C",
            [](py::object self) {
                auto& s = py::cast<Surface&>(self);
                return py::array_t<float>({3}, {sizeof(float)}, s.sellmeier_C, self);
            },
            [](Surface& s, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.size() != 3) throw py::value_error("sellmeier_C must have exactly 3 elements");
                std::copy_n(arr.data(), 3, s.sellmeier_C);
            })
        .def("ior_at", &Surface::ior_at, "lambda_nm"_a)
        .def("__repr__", [](const Surface& s) {
            return "Surface(radius=" + std::to_string(s.radius)
                + ", thickness=" + std::to_string(s.thickness)
                + ", ior=" + std::to_string(s.ior)
                + ", semi_aperture=" + std::to_string(s.semi_aperture)
                + ", z=" + std::to_string(s.z) + ")";
        });

    // Register std::vector<Surface> so OpticalSystem.surfaces is a live reference list.
    py::bind_vector<std::vector<Surface>>(m, "SurfaceList");

    // Register std::vector<std::string> for surface_ids
    py::bind_vector<std::vector<std::string>>(m, "StringList");

    // --------------------------------------------------------------- ApertureImage
    py::class_<ApertureImage>(m, "ApertureImage")
        .def(py::init<>())
        .def_readonly("width",          &ApertureImage::width)
        .def_readonly("height",         &ApertureImage::height)
        .def_property("semi_diameter",
            [](const ApertureImage& img) { return img.semi_diameter; },
            [](ApertureImage& img, float value) {
                if (!std::isfinite(value) || value < 0.0f)
                    throw py::value_error("semi_diameter must be finite and non-negative");
                img.semi_diameter = value;
            })
        .def_readwrite("source_path",   &ApertureImage::source_path)
        // Pixels as a 2-D float32 numpy view.  The getter returns a view that
        // keeps the owning ApertureImage alive; the setter copies the
        // user-provided array into the std::vector<float> backing store.
        .def_property("pixels",
            [](py::object self) {
                auto& img = py::cast<ApertureImage&>(self);
                if (img.pixels.empty())
                    return py::array_t<float>(
                        std::vector<py::ssize_t>{0, 0},
                        std::vector<py::ssize_t>{0, 0},
                        nullptr, self);
                return py::array_t<float>(
                    {(py::ssize_t)img.height, (py::ssize_t)img.width},
                    {(py::ssize_t)(img.width * sizeof(float)), (py::ssize_t)sizeof(float)},
                    img.pixels.data(), self);
            },
            [](ApertureImage& img,
               py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 2)
                    throw py::value_error("pixels must be a 2-D array (height, width)");
                if (arr.shape(0) > std::numeric_limits<int>::max()
                    || arr.shape(1) > std::numeric_limits<int>::max())
                    throw py::value_error("pixel dimensions are too large");
                int h = (int)arr.shape(0);
                int w = (int)arr.shape(1);
                img.height = h;
                img.width  = w;
                img.pixels.assign(arr.data(), arr.data() + (size_t)h * w);
            })
        .def("__repr__", [](const ApertureImage& img) {
            return "ApertureImage(width=" + std::to_string(img.width)
                + ", height=" + std::to_string(img.height)
                + ", semi_diameter=" + std::to_string(img.semi_diameter)
                + ", source_path='" + img.source_path + "')";
        });

    py::bind_vector<std::vector<ApertureImage>>(m, "ApertureImageList");

    // --------------------------------------------------------------- OpticalSystem
    py::class_<OpticalSystem>(m, "_OpticalSystem")
        .def(py::init<>())
        .def_readwrite("name",         &OpticalSystem::name)
        .def_readwrite("focal_length", &OpticalSystem::focal_length)
        .def_property("surfaces",
            [](OpticalSystem& self) -> std::vector<Surface>& { return self.surfaces; },
            [](OpticalSystem& self, const std::vector<Surface>& v) {
                self.surfaces = v;
                self.surface_ids.assign(v.size(), std::string());
                self.aperture_images.assign(v.size(), ApertureImage{});
                self.coating_tables.assign(v.size(), CoatingTables{});
                self.sync_coating_pointers();
                self.sync_aperture_profiles();
            },
            py::return_value_policy::reference_internal)
        .def_property("surface_ids",
            [](OpticalSystem& self) -> std::vector<std::string>& { return self.surface_ids; },
            [](OpticalSystem& self, const std::vector<std::string>& v) { self.surface_ids = v; },
            py::return_value_policy::reference_internal)
        .def_property("aperture_images",
            [](OpticalSystem& self) -> std::vector<ApertureImage>& { return self.aperture_images; },
            [](OpticalSystem& self, const std::vector<ApertureImage>& v) { self.aperture_images = v; },
            py::return_value_policy::reference_internal)
        .def("load", [](OpticalSystem& self, const std::string& path) {
            if (!self.load(path.c_str()))
                throw std::runtime_error("OpticalSystem.load() failed: " + path);
        }, "filename"_a)
        .def("finalize", [](OpticalSystem& self) {
            if (!self.finalize())
                throw std::runtime_error("OpticalSystem.finalize() failed: surfaces is empty");
        })
        .def("num_surfaces", &OpticalSystem::num_surfaces)
        .def("ior_before", [](const OpticalSystem& self, int idx) {
            if (idx < 0 || idx >= self.num_surfaces())
                throw py::index_error("surface index out of range");
            return self.ior_before(idx);
        }, "idx"_a)
        .def("ior_before_at", [](const OpticalSystem& self, int idx, float lambda_nm) {
            if (idx < 0 || idx >= self.num_surfaces())
                throw py::index_error("surface index out of range");
            return self.ior_before(idx, lambda_nm);
        }, "idx"_a, "lambda_nm"_a)
        .def("insert_surface", &OpticalSystem::insert_surface,
             "index"_a, "surface"_a, "surface_id"_a = std::string())
        .def("erase_surface", &OpticalSystem::erase_surface, "index"_a)
        // ---- Coating table accessors ------------------------------------
        // Table contents live in OpticalSystem::coating_tables (off the
        // Surface POD); these copy data in/out and re-patch the Surface
        // coating pointers, so callers never touch raw pointers.
        .def("set_coating_spectral_table",
            [](OpticalSystem& self, int i,
               py::array_t<float, py::array::c_style | py::array::forcecast> data,
               bool out_of_range_discard) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                if (data.ndim() != 2 || data.shape(1) != 2 || data.shape(0) < 1)
                    throw py::value_error("spectral table must be an (N, 2) array of (lambda_nm, r)");
                if (self.coating_tables.size() < self.surfaces.size())
                    self.coating_tables.resize(self.surfaces.size());
                CoatingTables t;
                auto d = data.unchecked<2>();
                t.table.resize((size_t)data.shape(0));
                for (py::ssize_t k = 0; k < data.shape(0); ++k)
                    t.table[(size_t)k] = { d(k, 0), d(k, 1) };
                std::sort(t.table.begin(), t.table.end(),
                          [](const CoatingTable1D& a, const CoatingTable1D& b)
                          { return a.key < b.key; });
                self.coating_tables[i] = std::move(t);
                Coating c;
                c.model = CoatingModel::SPECTRAL;
                c.out_of_range_discard = out_of_range_discard;
                self.surfaces[i].coating = c;
                self.sync_coating_pointers();
            }, "index"_a, "data"_a, "out_of_range_discard"_a = false)
        .def("set_coating_angular_table",
            [](OpticalSystem& self, int i,
               py::array_t<float, py::array::c_style | py::array::forcecast> data,
               float angle_ref_ior, bool out_of_range_discard) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                if (data.ndim() != 2 || data.shape(1) != 2 || data.shape(0) < 1)
                    throw py::value_error("angular table must be an (N, 2) array of (angle_deg, r)");
                if (self.coating_tables.size() < self.surfaces.size())
                    self.coating_tables.resize(self.surfaces.size());
                CoatingTables t;
                auto d = data.unchecked<2>();
                t.table.resize((size_t)data.shape(0));
                for (py::ssize_t k = 0; k < data.shape(0); ++k)
                    t.table[(size_t)k] = { d(k, 0), d(k, 1) };
                std::sort(t.table.begin(), t.table.end(),
                          [](const CoatingTable1D& a, const CoatingTable1D& b)
                          { return a.key < b.key; });
                self.coating_tables[i] = std::move(t);
                Coating c;
                c.model = CoatingModel::ANGULAR;
                c.angle_ref_ior = angle_ref_ior;
                c.out_of_range_discard = out_of_range_discard;
                self.surfaces[i].coating = c;
                self.sync_coating_pointers();
            }, "index"_a, "data"_a, "angle_ref_ior"_a = 1.0f,
               "out_of_range_discard"_a = false)
        .def("get_coating_table",
            // Shared 1-D table getter for SPECTRAL and ANGULAR models;
            // returns a copied (N, 2) array of (key, r).
            [](const OpticalSystem& self, int i) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                size_t n = (i < (int)self.coating_tables.size())
                         ? self.coating_tables[i].table.size() : 0;
                py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)n, 2});
                if (n > 0)
                {
                    const auto& t = self.coating_tables[i].table;
                    auto o = out.mutable_unchecked<2>();
                    for (size_t k = 0; k < n; ++k)
                    {
                        o((py::ssize_t)k, 0) = t[k].key;
                        o((py::ssize_t)k, 1) = t[k].r;
                    }
                }
                return out;
            }, "index"_a)
        .def("set_coating_sa_table",
            [](OpticalSystem& self, int i,
               py::array_t<float, py::array::c_style | py::array::forcecast> wl,
               py::array_t<float, py::array::c_style | py::array::forcecast> ang,
               py::array_t<float, py::array::c_style | py::array::forcecast> r,
               float angle_ref_ior, bool out_of_range_discard) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                if (wl.ndim() != 1 || ang.ndim() != 1 || wl.shape(0) < 1 || ang.shape(0) < 1)
                    throw py::value_error("wavelengths and angles must be 1-D non-empty arrays");
                if (r.ndim() != 2 || r.shape(0) != wl.shape(0) || r.shape(1) != ang.shape(0))
                    throw py::value_error("r must be an (n_wavelengths, n_angles) array");
                if (!std::is_sorted(wl.data(), wl.data() + wl.shape(0))
                    || !std::is_sorted(ang.data(), ang.data() + ang.shape(0)))
                    throw py::value_error("wavelengths and angles must be sorted ascending");
                if (self.coating_tables.size() < self.surfaces.size())
                    self.coating_tables.resize(self.surfaces.size());
                CoatingTables t;
                t.sa_wavelengths.assign(wl.data(), wl.data() + wl.shape(0));
                t.sa_angles.assign(ang.data(), ang.data() + ang.shape(0));
                t.sa_r.assign(r.data(), r.data() + (size_t)r.shape(0) * r.shape(1));
                self.coating_tables[i] = std::move(t);
                Coating c;
                c.model = CoatingModel::SPECTRAL_ANGULAR;
                c.angle_ref_ior = angle_ref_ior;
                c.out_of_range_discard = out_of_range_discard;
                self.surfaces[i].coating = c;
                self.sync_coating_pointers();
            }, "index"_a, "wavelengths_nm"_a, "angles_deg"_a, "r"_a,
               "angle_ref_ior"_a = 1.0f, "out_of_range_discard"_a = false)
        .def("get_coating_sa_table",
            // Returns (wavelengths_nm, angles_deg, r) as copied arrays.
            [](const OpticalSystem& self, int i) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                size_t nw = 0, na = 0;
                const CoatingTables* t = nullptr;
                if (i < (int)self.coating_tables.size())
                {
                    const CoatingTables& tt = self.coating_tables[i];
                    if (!tt.sa_r.empty()
                        && tt.sa_r.size() == tt.sa_wavelengths.size() * tt.sa_angles.size())
                    {
                        t = &tt;
                        nw = tt.sa_wavelengths.size();
                        na = tt.sa_angles.size();
                    }
                }
                py::array_t<float> wl(std::vector<py::ssize_t>{(py::ssize_t)nw});
                py::array_t<float> ang(std::vector<py::ssize_t>{(py::ssize_t)na});
                py::array_t<float> r(std::vector<py::ssize_t>{(py::ssize_t)nw,
                                                              (py::ssize_t)na});
                if (t)
                {
                    std::copy_n(t->sa_wavelengths.data(), nw, wl.mutable_data());
                    std::copy_n(t->sa_angles.data(), na, ang.mutable_data());
                    std::copy_n(t->sa_r.data(), nw * na, r.mutable_data());
                }
                return py::make_tuple(wl, ang, r);
            }, "index"_a)
        .def("set_coating_layers",
            // layers: list of dicts {material: str, thickness_nm: float,
            // nk_table: (K, 3) array of (lambda_um, n, k)}.  Stores the specs
            // as the writer's source-of-truth, then bakes them to a
            // SPECTRAL_ANGULAR table via TMM (requires final surface IORs).
            [](OpticalSystem& self, int i, py::list layers) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                if (py::len(layers) == 0)
                    throw py::value_error("layers list must not be empty");
                if (self.coating_tables.size() < self.surfaces.size())
                    self.coating_tables.resize(self.surfaces.size());
                CoatingTables t;
                for (auto item : layers)
                {
                    py::dict d = py::cast<py::dict>(item);
                    CoatingLayerSpec spec;
                    spec.material     = d.contains("material")
                                      ? py::cast<std::string>(d["material"]) : "";
                    spec.thickness_nm = py::cast<float>(d["thickness_nm"]);
                    auto nk = py::cast<py::array_t<float,
                        py::array::c_style | py::array::forcecast>>(d["nk_table"]);
                    if (nk.ndim() != 2 || nk.shape(1) != 3 || nk.shape(0) < 1)
                        throw py::value_error("nk_table must be a (K, 3) array of (lambda_um, n, k)");
                    auto a = nk.unchecked<2>();
                    for (py::ssize_t k = 0; k < nk.shape(0); ++k)
                    {
                        spec.nk_lambda_um.push_back(a(k, 0));
                        spec.nk_n.push_back(a(k, 1));
                        spec.nk_k.push_back(a(k, 2));
                    }
                    t.layers.push_back(std::move(spec));
                }
                CoatingTables old_tables = self.coating_tables[i];
                Coating old_coating = self.surfaces[i].coating;
                self.coating_tables[i] = std::move(t);
                self.surfaces[i].coating = Coating{};
                std::string err;
                if (!self.bake_coating_layers(i, &err))
                {
                    self.coating_tables[i] = std::move(old_tables);
                    self.surfaces[i].coating = old_coating;
                    self.sync_coating_pointers();
                    throw py::value_error("set_coating_layers: " + err);
                }
            }, "index"_a, "layers"_a)
        .def("get_coating_layers",
            [](const OpticalSystem& self, int i) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                py::list out;
                if (i >= (int)self.coating_tables.size())
                    return out;
                for (const CoatingLayerSpec& spec : self.coating_tables[i].layers)
                {
                    py::dict d;
                    d["material"]     = spec.material;
                    d["thickness_nm"] = spec.thickness_nm;
                    py::array_t<float> nk(std::vector<py::ssize_t>{
                        (py::ssize_t)spec.nk_lambda_um.size(), 3});
                    auto a = nk.mutable_unchecked<2>();
                    for (size_t k = 0; k < spec.nk_lambda_um.size(); ++k)
                    {
                        a((py::ssize_t)k, 0) = spec.nk_lambda_um[k];
                        a((py::ssize_t)k, 1) = spec.nk_n[k];
                        a((py::ssize_t)k, 2) = spec.nk_k[k];
                    }
                    d["nk_table"] = nk;
                    out.append(d);
                }
                return out;
            }, "index"_a)
        .def("clear_coating",
            // Reset surface i to uncoated SIMPLE and free its table storage.
            [](OpticalSystem& self, int i) {
                if (i < 0 || i >= (int)self.surfaces.size())
                    throw py::index_error("surface index out of range");
                if (self.coating_tables.size() < self.surfaces.size())
                    self.coating_tables.resize(self.surfaces.size());
                self.coating_tables[i] = CoatingTables{};
                self.surfaces[i].coating = Coating{};
                self.sync_coating_pointers();
            }, "index"_a)
        .def("sync_coating_pointers", &OpticalSystem::sync_coating_pointers)
        .def("coating_state_hash", &OpticalSystem::coating_state_hash)
        .def("aperture_image_state_hash", &OpticalSystem::aperture_image_state_hash)
        .def("print_summary", &OpticalSystem::print_summary)
        .def("__repr__", [](const OpticalSystem& l) {
            return "OpticalSystem(name='" + l.name + "', surfaces="
                + std::to_string(l.num_surfaces()) + ")";
        });

    m.def("dot",   &dot,   "a"_a, "b"_a);
    m.def("cross", &cross, "a"_a, "b"_a);
}
