#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "PointFlareRenderer.h"
#include "SourceFlareRenderer.h"
#include "PSFRenderer.h"
#include "diffraction.h"
#include "veil_render.h"
#include "gate_render.h"
#include "hurb.h"
#include "source_map.h"
#include "aperture_sdf.h"
#include "spectral.h"
#include "buffers.h"

#include <cmath>
#include <limits>
#include <string>

// cudaGetDeviceCount resolved at link time from ghostlight_core's static CUDA runtime.
extern "C" int cudaGetDeviceCount(int*);

namespace py = pybind11;
using namespace py::literals;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

void require_positive_dimensions(const char* function, int w, int h)
{
    if (w <= 0 || h <= 0)
        throw py::value_error(std::string(function)
                              + ": width and height must be positive");
}

// Shared single-source setup for the debug render entry points.
struct DebugSource {
    std::vector<FlareSource> sources;
    std::vector<float>       spx, spy;
    float eff_half_w = 0.0f, eff_half_h = 0.0f;
};

DebugSource make_debug_source(const OpticalSystem& lens,
                              const PointFlareConfig& cfg,
                              const LensCalibration& calib, int w, int h)
{
    DebugSource d;
    d.eff_half_w = (cfg.sensor_half_w > 0.0f) ? cfg.sensor_half_w : calib.sensor_half_w;
    d.eff_half_h = (cfg.sensor_half_h > 0.0f) ? cfg.sensor_half_h : calib.sensor_half_h;

    // Debug renders use the production source-angle mapping.
    const SourceMapSolve smap = solve_source_angle(lens, calib,
                                                   (cfg.source_x - 0.5f) * 2.0f,
                                                   (cfg.source_y - 0.5f) * 2.0f,
                                                   d.eff_half_w, d.eff_half_h);
    const float ax0 = smap.angle_x;
    const float ay0 = smap.angle_y;

    float M_in[3][3], M_out[3][3];
    resolve_input_matrix(cfg.input_cs, cfg.custom_input_to_xyz, M_in);
    resolve_output_matrix(cfg.output_cs, cfg.custom_xyz_to_output, M_out);
    const float xr = M_in[0][0]*cfg.source_r + M_in[0][1]*cfg.source_g + M_in[0][2]*cfg.source_b;
    const float xg = M_in[1][0]*cfg.source_r + M_in[1][1]*cfg.source_g + M_in[1][2]*cfg.source_b;
    const float xb = M_in[2][0]*cfg.source_r + M_in[2][1]*cfg.source_g + M_in[2][2]*cfg.source_b;
    FlareSource s;
    s.angle_x = ax0; s.angle_y = ay0;
    s.r = M_out[0][0]*xr + M_out[0][1]*xg + M_out[0][2]*xb;
    s.g = M_out[1][0]*xr + M_out[1][1]*xg + M_out[1][2]*xb;
    s.b = M_out[2][0]*xr + M_out[2][1]*xg + M_out[2][2]*xb;

    float sx = 0.0f, sy = 0.0f;
    source_map_screen(smap, ax0, ay0, &sx, &sy);
    d.sources = { s };
    d.spx = { sx * w };
    d.spy = { sy * h };
    return d;
}

} // namespace

void bind_renderers(py::module_& m)
{
    m.def("render_point_flare",
        [](int w, int h,
           const OpticalSystem&       lens,
           const LensCalibration&  calib,
           const PointFlareConfig& cfg) -> py::dict
        {
            require_positive_dimensions("render_point_flare", w, h);
            FlareBuffers out;
            bool ok;
            {
                py::gil_scoped_release nogil;   // let the caller's UI thread run during the GPU render
                ok = render_point_flare(w, h, lens, calib, cfg, out);
            }
            if (!ok)
                throw std::runtime_error("render_point_flare: CUDA error");
            return flare_buffers_to_dict(std::move(out));
        },
        "width"_a, "height"_a, "lens"_a, "calibration"_a, "config"_a);

    // Takes a (N, 3) float32 array of [d_angle_x, d_angle_y, weight] rows:
    // angular offsets (radians) around the screen-space center given by
    // config.source_x/source_y, each weighted by its quadrature weight.
    // Weights summing to 1 average an extended source; a subset of those
    // weights renders a partial (progressive) accumulation chunk.
    m.def("render_source_flare",
        [](FloatArray offsets, int w, int h,
           const OpticalSystem&       lens,
           const LensCalibration&  calib,
           const PointFlareConfig& cfg) -> py::dict
        {
            require_positive_dimensions("render_source_flare", w, h);
            if (offsets.ndim() != 2 || offsets.shape(1) != 3 || offsets.shape(0) < 1)
                throw py::value_error("render_source_flare: offsets must be a (N, 3) "
                                      "float32 array of [d_angle_x, d_angle_y, weight] "
                                      "rows with N >= 1");
            if (offsets.shape(0) > std::numeric_limits<int>::max())
                throw py::value_error("render_source_flare: too many source offsets");
            const int n = (int)offsets.shape(0);

            FlareBuffers out;
            bool ok;
            {
                py::gil_scoped_release nogil;   // let the caller's UI thread run during the GPU render
                ok = render_source_flare(w, h, offsets.data(), n, lens, calib, cfg, out);
            }
            if (!ok)
                throw std::runtime_error("render_source_flare: CUDA error");
            return flare_buffers_to_dict(std::move(out));
        },
        "offsets"_a, "width"_a, "height"_a, "lens"_a, "calibration"_a, "config"_a);

    // ----------------------------------------------------------- PSFCellStatus
    py::enum_<PSFCellStatus>(m, "PSFCellStatus")
        .value("OK",              PSF_CELL_OK)
        .value("CHIEF_VIGNETTED", PSF_CELL_CHIEF_VIGNETTED)
        .value("DARK",            PSF_CELL_DARK)
        .export_values();

    // Takes a (N, 5) float32 array of [angle_x, angle_y, r, g, b] rows, or a
    // (N, 7) array of [angle_x, angle_y, r, g, b, target_x_mm, target_y_mm]
    // rows for FIXED_TARGET mode (angle_* is then the aim seed).  Each source
    // maps to one tile, row-major: source i → tile (i % grid_nx, i / grid_nx).
    // N must be <= grid_nx * grid_ny.
    m.def("render_psf",
        [](FloatArray sources_arr,
           const OpticalSystem&    lens,
           const PSFConfig&        cfg) -> py::dict
        {
            if (cfg.grid_nx <= 0 || cfg.grid_ny <= 0
                || cfg.tile_w <= 0 || cfg.tile_h <= 0)
                throw py::value_error("render_psf: grid and tile dimensions must be positive");
            if (!(cfg.tile_extent_mm > 0.0f) || !std::isfinite(cfg.tile_extent_mm))
                throw py::value_error("render_psf: tile_extent_mm must be finite and positive");
            if (sources_arr.ndim() != 2
                || (sources_arr.shape(1) != 5 && sources_arr.shape(1) != 7))
                throw py::value_error("render_psf: sources must be a (N, 5) array of "
                                      "[angle_x, angle_y, r, g, b] rows, or a (N, 7) array "
                                      "with [.., target_x_mm, target_y_mm] appended");
            if (sources_arr.shape(0) > std::numeric_limits<int>::max())
                throw py::value_error("render_psf: too many sources");
            const int   n     = (int)sources_arr.shape(0);
            const int   width = (int)sources_arr.shape(1);
            std::vector<PSFSource> sources(n);
            const float* p = sources_arr.data();
            for (int i = 0; i < n; ++i) {
                sources[i].angle_x = p[i * width + 0];
                sources[i].angle_y = p[i * width + 1];
                sources[i].r       = p[i * width + 2];
                sources[i].g       = p[i * width + 3];
                sources[i].b       = p[i * width + 4];
                if (width == 7) {
                    sources[i].target_x_mm = p[i * width + 5];
                    sources[i].target_y_mm = p[i * width + 6];
                }
            }

            PSFOutput out;
            bool ok;
            {
                py::gil_scoped_release nogil;   // let the caller's UI thread run during the GPU render
                ok = render_psf(lens, sources, cfg, out);
            }
            if (!ok)
                throw std::runtime_error("render_psf: CUDA error");
            return psf_output_to_dict(std::move(out));
        },
        "sources"_a, "lens"_a, "config"_a);

    // Debug entry point exposing the starburst's PSF, sprite, scale, and layer.
    m.def("_render_starburst_debug",
        [](int w, int h,
           const OpticalSystem&    lens,
           const LensCalibration&  calib,
           const PointFlareConfig& cfg) -> py::dict
        {
            const DebugSource ds = make_debug_source(lens, cfg, calib, w, h);
            const float eff_half_w = ds.eff_half_w, eff_half_h = ds.eff_half_h;
            const auto& sources = ds.sources;
            const auto& spx     = ds.spx;
            const auto& spy     = ds.spy;

            std::vector<float> sb_r((size_t)w*h, 0.0f), sb_g((size_t)w*h, 0.0f), sb_b((size_t)w*h, 0.0f);
            StarburstDebug dbg;
            std::string err;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = render_starburst(lens, calib, sources, spx, spy, w, h, w, h,
                                      eff_half_w, eff_half_h, cfg,
                                      sb_r.data(), sb_g.data(), sb_b.data(), &err, &dbg);
            }
            if (!ok)
                throw std::runtime_error("render_starburst: " + err);

            py::dict d;
            d["width"] = w; d["height"] = h;
            d["grid"] = dbg.grid;
            d["starburst_r"] = view_vector_moved(std::move(sb_r), h, w);
            d["starburst_g"] = view_vector_moved(std::move(sb_g), h, w);
            d["starburst_b"] = view_vector_moved(std::move(sb_b), h, w);
            d["psf"]        = view_vector_moved(std::move(dbg.psf), dbg.grid, dbg.grid);
            d["sprite_rgb"] = view_vector_moved_3d(std::move(dbg.sprite_rgb), dbg.grid, dbg.grid, 3);
            d["pupil"]      = view_vector_moved(std::move(dbg.pupil), dbg.grid, dbg.grid);
            d["dx_mm_x"] = dbg.dx_mm_x; d["dx_mm_y"] = dbg.dx_mm_y;
            d["f_number_x"] = dbg.f_number_x; d["f_number_y"] = dbg.f_number_y;
            d["focal_length_x"] = dbg.focal_length_x; d["focal_length_y"] = dbg.focal_length_y;
            d["entrance_pupil_semi_x"] = dbg.entrance_pupil_semi_x;
            d["entrance_pupil_semi_y"] = dbg.entrance_pupil_semi_y;
            d["lambda_ref_nm"] = dbg.lambda_ref_nm;
            d["field_angle_x"] = dbg.field_angle_x; d["field_angle_y"] = dbg.field_angle_y;
            d["throughput"] = dbg.throughput;
            d["chief_offset_x"] = dbg.chief_offset_x; d["chief_offset_y"] = dbg.chief_offset_y;
            d["source_px"] = spx[0]; d["source_py"] = spy[0];
            return d;
        },
        "width"_a, "height"_a, "lens"_a, "calibration"_a, "config"_a);

    // Debug entry point exposing the veil's normalized GSF, RGB sprite, scale,
    // and final layer for one source.
    m.def("_render_veil_debug",
        [](int w, int h,
           const OpticalSystem&    lens,
           const LensCalibration&  calib,
           const PointFlareConfig& cfg) -> py::dict
        {
            const DebugSource ds = make_debug_source(lens, cfg, calib, w, h);
            const float eff_half_w = ds.eff_half_w, eff_half_h = ds.eff_half_h;
            const auto& sources = ds.sources;
            const auto& spx     = ds.spx;
            const auto& spy     = ds.spy;

            std::vector<float> ve_r((size_t)w*h, 0.0f), ve_g((size_t)w*h, 0.0f), ve_b((size_t)w*h, 0.0f);
            VeilDebug dbg;
            std::string err;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = render_veil(lens, calib, sources, spx, spy, w, h, w, h,
                                 eff_half_w, eff_half_h, cfg,
                                 ve_r.data(), ve_g.data(), ve_b.data(), &err, &dbg);
            }
            if (!ok)
                throw std::runtime_error("render_veil: " + err);

            py::dict d;
            d["width"] = w; d["height"] = h;
            d["grid"] = dbg.grid;
            d["veil_r"] = view_vector_moved(std::move(ve_r), h, w);
            d["veil_g"] = view_vector_moved(std::move(ve_g), h, w);
            d["veil_b"] = view_vector_moved(std::move(ve_b), h, w);
            if (dbg.grid > 0) {
                d["reference"]  = view_vector_moved(std::move(dbg.reference), dbg.grid, dbg.grid);
                d["sprite_rgb"] = view_vector_moved_3d(std::move(dbg.sprite_rgb), dbg.grid, dbg.grid, 3);
            }
            d["dx_mm_x"] = dbg.dx_mm_x; d["dx_mm_y"] = dbg.dx_mm_y;
            d["core_mm"] = dbg.core_mm;
            d["falloff"] = dbg.falloff;
            d["core_texels"] = dbg.core_texels;
            d["source_px"] = spx[0]; d["source_py"] = spy[0];
            return d;
        },
        "width"_a, "height"_a, "lens"_a, "calibration"_a, "config"_a);

    // Debug entry point exposing the gate layer, direct-source reference, scrape
    // counts, reach, and predicted t/(2N) band for one source.
    m.def("_render_gate_debug",
        [](int w, int h,
           const OpticalSystem&    lens,
           const LensCalibration&  calib,
           const PointFlareConfig& cfg) -> py::dict
        {
            const DebugSource ds = make_debug_source(lens, cfg, calib, w, h);
            const float eff_half_w = ds.eff_half_w, eff_half_h = ds.eff_half_h;

            std::vector<float> ga_r((size_t)w*h, 0.0f), ga_g((size_t)w*h, 0.0f), ga_b((size_t)w*h, 0.0f);
            GateDebug dbg;
            std::string err;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = render_gate(lens, calib, ds.sources, w, h, w, h, 0, 0,
                                 eff_half_w, eff_half_h, cfg,
                                 ga_r.data(), ga_g.data(), ga_b.data(), &err, &dbg);
            }
            if (!ok)
                throw std::runtime_error("render_gate: " + err);

            py::dict d;
            d["width"] = w; d["height"] = h;
            d["gate_r"] = view_vector_moved(std::move(ga_r), h, w);
            d["gate_g"] = view_vector_moved(std::move(ga_g), h, w);
            d["gate_b"] = view_vector_moved(std::move(ga_b), h, w);
            if (!dbg.direct_r.empty()) {
                d["direct_r"] = view_vector_moved(std::move(dbg.direct_r), h, w);
                d["direct_g"] = view_vector_moved(std::move(dbg.direct_g), h, w);
                d["direct_b"] = view_vector_moved(std::move(dbg.direct_b), h, w);
            }
            d["traces"]    = dbg.traces;
            d["scrapes"]   = dbg.scrapes;
            d["energy"]    = dbg.energy;
            d["reach_mm"]  = dbg.reach_mm;
            d["band_x_mm"] = dbg.band_x_mm;
            d["band_y_mm"] = dbg.band_y_mm;
            d["x_pos"] = dbg.gate.x_pos; d["x_neg"] = dbg.gate.x_neg;
            d["y_pos"] = dbg.gate.y_pos; d["y_neg"] = dbg.gate.y_neg;
            d["t_mm"]  = dbg.gate.t_mm;  d["zs_mm"] = dbg.gate.zs_mm;
            d["sig_wide"] = dbg.gate.sig_wide; d["sig_tight"] = dbg.gate.sig_tight;
            return d;
        },
        "width"_a, "height"_a, "lens"_a, "calibration"_a, "config"_a);

    // The source-position map, exposed so it can be checked against a fresh
    // trace rather than against its own arithmetic. `source_x/y` are screen
    // fractions ([0,1], 0.5 = optical axis), matching PointFlareConfig.
    m.def("_solve_source_map",
        [](const OpticalSystem& lens, const LensCalibration& calib,
           float source_x, float source_y,
           float eff_half_w, float eff_half_h) -> py::dict
        {
            SourceMapSolve sm;
            {
                py::gil_scoped_release nogil;
                sm = solve_source_angle(lens, calib,
                                        (source_x - 0.5f) * 2.0f,
                                        (source_y - 0.5f) * 2.0f,
                                        eff_half_w, eff_half_h);
            }
            const char* st = (sm.status == SourceMapStatus::SOLVED)    ? "solved"
                           : (sm.status == SourceMapStatus::CONTINUED) ? "continued"
                                                                       : "fallback";
            py::dict d;
            d["angle_x"]     = sm.angle_x;
            d["angle_y"]     = sm.angle_y;
            d["status"]      = st;
            d["residual_mm"] = sm.residual_mm;
            d["anchor_ax"]   = sm.anchor_ax;  d["anchor_ay"] = sm.anchor_ay;
            d["anchor_x_mm"] = sm.anchor_x;   d["anchor_y_mm"] = sm.anchor_y;
            // Row-major d(landing_mm)/d(angle_rad): [dx/dax, dx/day, dy/dax, dy/day].
            d["jacobian"]    = py::make_tuple(sm.J[0], sm.J[1], sm.J[2], sm.J[3]);
            d["probe_evals"] = sm.probe_evals;
            // Where the solved angle puts the source on screen — the forward
            // direction every splatted layer uses.
            float sx = 0.0f, sy = 0.0f;
            source_map_screen(sm, sm.angle_x, sm.angle_y, &sx, &sy);
            d["screen_x"] = sx;
            d["screen_y"] = sy;
            return d;
        },
        "lens"_a, "calibration"_a, "source_x"_a, "source_y"_a,
        "eff_half_w"_a, "eff_half_h"_a);

    // The relation _solve_source_map inverts: one traced landing at one angle.
    // Returns None past the image circle, where the lens delivers no landing.
    m.def("_source_map_landing",
        [](const OpticalSystem& lens, const LensCalibration& calib,
           float angle_x, float angle_y) -> py::object
        {
            float x = 0.0f, y = 0.0f;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = source_map_landing(lens, calib, angle_x, angle_y, &x, &y);
            }
            if (!ok) return py::none();
            return py::make_tuple(x, y);
        },
        "lens"_a, "calibration"_a, "angle_x"_a, "angle_y"_a);

    // The forward map at an arbitrary angle, given a base solve — the pure
    // arithmetic an extended source's offsets go through, with no tracing.
    m.def("_source_map_screen",
        [](const OpticalSystem& lens, const LensCalibration& calib,
           float source_x, float source_y, float eff_half_w, float eff_half_h,
           float angle_x, float angle_y) -> py::tuple
        {
            float sx = 0.0f, sy = 0.0f;
            {
                py::gil_scoped_release nogil;
                const SourceMapSolve sm =
                    solve_source_angle(lens, calib,
                                       (source_x - 0.5f) * 2.0f,
                                       (source_y - 0.5f) * 2.0f,
                                       eff_half_w, eff_half_h);
                source_map_screen(sm, angle_x, angle_y, &sx, &sy);
            }
            return py::make_tuple(sx, sy);
        },
        "lens"_a, "calibration"_a, "source_x"_a, "source_y"_a,
        "eff_half_w"_a, "eff_half_h"_a, "angle_x"_a, "angle_y"_a);

    // Host-only gate parameter resolution.
    m.def("_gate_params_debug",
        [](const GateConfig& gc, float half_w, float half_h) -> py::dict
        {
            const GpuGate g = build_gpu_gate(gc, half_w, half_h);
            py::dict d;
            d["x_pos"] = g.x_pos; d["x_neg"] = g.x_neg;
            d["y_pos"] = g.y_pos; d["y_neg"] = g.y_neg;
            d["t_mm"] = g.t_mm;   d["zs_mm"] = g.zs_mm;
            d["sig_wide"] = g.sig_wide; d["sig_tight"] = g.sig_tight;
            d["cos_g"] = g.cos_g; d["sin_g"] = g.sin_g;
            d["r0"] = g.r0;       d["max_kick"] = g.max_kick;
            d["gain"] = g.gain;   d["n_scatter"] = g.n_scatter;
            d["inv_scatter"] = g.inv_scatter;
            d["lobe"] = g.lobe;
            return d;
        },
        "config"_a, "sensor_half_w"_a, "sensor_half_h"_a);

    // Host-only gate-lobe sampler.
    m.def("_gate_sample_debug",
        [](const GateConfig& gc, int n, unsigned int seed) -> py::dict
        {
            const GpuGate g = build_gpu_gate(gc, 1.0f, 1.0f);
            const int count = (n > 0) ? n : 0;
            std::vector<float> ax(count), ot(count), az(count);
            for (int i = 0; i < count; ++i) {
                unsigned int state = gate_wang_hash(seed ^ (2654435761u * (unsigned int)i + 1u));
                if (state == 0u) state = 1u;
                gate_sample_lobe(g, state, ax[i], ot[i], az[i]);
            }
            py::dict d;
            d["axis"]  = view_vector_moved_1d(std::move(ax), count);
            d["other"] = view_vector_moved_1d(std::move(ot), count);
            d["z"]     = view_vector_moved_1d(std::move(az), count);
            d["sig_wide"]  = g.sig_wide;
            d["sig_tight"] = g.sig_tight;
            return d;
        },
        "config"_a, "n"_a, "seed"_a);

    // Host-only HURB sampler using an independent hashed seed per entry.
    m.def("_hurb_sample_debug",
        [](float lambda_nm, FloatArray d_in, int dist, float max_kick,
           unsigned int seed) -> py::dict
        {
            auto d = d_in.unchecked<1>();
            const int n = (int)d.shape(0);
            std::vector<float> kicks(n), sigmas(n);
            for (int i = 0; i < n; ++i) {
                const float dd = d(i);
                const float sg = hurb_sigma(lambda_nm, dd, dist);
                unsigned int state = hurb_wang_hash(seed ^ (2654435761u * (unsigned int)i + 1u));
                if (state == 0u) state = 1u;
                kicks[i]  = (sg > 0.0f) ? hurb_sample_kick(dist, sg, max_kick, state) : 0.0f;
                sigmas[i] = sg;
            }
            py::dict out;
            out["kicks"]     = view_vector_moved_1d(std::move(kicks), n);
            out["sigma"]     = view_vector_moved_1d(std::move(sigmas), n);
            out["gauss_k"]   = HURB_GAUSS_K;
            out["lorentz_k"] = HURB_LORENTZ_K;
            return out;
        },
        "lambda_nm"_a, "d_mm"_a, "dist"_a, "max_kick_rad"_a, "seed"_a);

    // Host-only image-stop SDF bake. A negative stop_index returns empty arrays.
    m.def("_aperture_sdf_debug",
        [](const OpticalSystem& lens) -> py::dict
        {
            py::dict out;
            const int target = find_sdf_target_surface(lens);
            out["stop_index"] = target;
            if (target < 0) return out;

            const Surface&       s   = lens.surfaces[target];
            const ApertureImage& img = lens.aperture_images[target];
            float sd = (s.aperture_semi_diameter > 0.0f) ? s.aperture_semi_diameter
                                                         : img.semi_diameter;

            ApertureSdfBake bake;
            if (!bake_aperture_sdf(img, sd, s.aperture_aspect, bake))
                return out;

            const int W = bake.width, H = bake.height;
            std::vector<float> sdf((size_t)W * H), nx((size_t)W * H), ny((size_t)W * H);
            for (size_t p = 0; p < (size_t)W * H; ++p) {
                sdf[p] = bake.texels[p * 4 + 0];
                nx[p]  = bake.texels[p * 4 + 1];
                ny[p]  = bake.texels[p * 4 + 2];
            }
            out["sdf"]    = view_vector_moved(std::move(sdf), H, W);
            out["nx"]     = view_vector_moved(std::move(nx),  H, W);
            out["ny"]     = view_vector_moved(std::move(ny),  H, W);
            out["sx"]     = bake.sx;
            out["sy"]     = bake.sy;
            out["width"]  = W;
            out["height"] = H;
            return out;
        },
        "lens"_a);

    // Expose CUDA device availability.
    m.def("_cuda_available", []() -> bool {
        int count = 0;
        return cudaGetDeviceCount(&count) == 0 && count > 0;
    });
}
