#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "render_config.h"
#include "spectral.h"

namespace py = pybind11;
using namespace py::literals;

void bind_config(py::module_& m)
{
    // --------------------------------------------------------- GhostAovMode enum
    py::enum_<GhostAovMode>(m, "GhostAovMode")
        .value("NONE",     GhostAovMode::NONE)
        .value("PER_PAIR", GhostAovMode::PER_PAIR)
        .export_values();

    // --------------------------------------------------------- GhostFilter
    // Applied after intensity and IOR-contrast filtering. `pairs` uses canonical
    // (surf_a, surf_b) tuples with surf_a < surf_b.
    py::class_<GhostFilter> filter_cls(m, "GhostFilter");
    py::enum_<GhostFilter::Mode>(filter_cls, "Mode")
        .value("ALL",     GhostFilter::Mode::ALL)
        .value("INCLUDE", GhostFilter::Mode::INCLUDE)
        .value("EXCLUDE", GhostFilter::Mode::EXCLUDE)
        .export_values();
    filter_cls
        .def(py::init<>())
        .def_readwrite("mode", &GhostFilter::mode)
        .def_readwrite("pairs", &GhostFilter::pairs)
        .def("__repr__", [](const GhostFilter& f) {
            const char* m = (f.mode == GhostFilter::Mode::ALL)     ? "ALL"
                          : (f.mode == GhostFilter::Mode::INCLUDE) ? "INCLUDE"
                          : "EXCLUDE";
            return "GhostFilter(mode=" + std::string(m)
                 + ", pairs=" + std::to_string(f.pairs.size()) + ")";
        });

    // --------------------------------------------------------- Color space enums
    py::enum_<SensorModel>(m, "SensorModel")
        .value("CIE_1931", SENSOR_CIE_1931)
        .export_values();

    py::enum_<InputColorSpace>(m, "InputColorSpace")
        .value("ACESCG",      CS_INPUT_ACESCG)
        .value("SRGB_LINEAR", CS_INPUT_SRGB_LINEAR)
        .value("CUSTOM",      CS_INPUT_CUSTOM)
        .export_values();

    py::enum_<OutputColorSpace>(m, "OutputColorSpace")
        .value("ACESCG",      CS_ACESCG)
        .value("SRGB_LINEAR", CS_SRGB_LINEAR)
        .value("P3_D65",      CS_P3_D65)
        .value("P3_D60",      CS_P3_D60)
        .value("XYZ",         CS_XYZ)
        .value("CUSTOM",      CS_CUSTOM)
        .export_values();

    py::enum_<HurbKickDistribution>(m, "HurbKickDistribution")
        .value("GAUSSIAN",   HurbKickDistribution::GAUSSIAN)
        .value("LORENTZIAN", HurbKickDistribution::LORENTZIAN);

    py::enum_<StarburstEngine>(m, "StarburstEngine")
        .value("SPRITE", StarburstEngine::SPRITE)
        .value("MDFT",   StarburstEngine::MDFT)
        .export_values();

    // ----------------------------------------------------------- RenderConfig
    py::class_<RenderConfig>(m, "RenderConfig")
        .def(py::init<>())
        .def_readwrite("ray_grid",          &RenderConfig::ray_grid)
        .def_readwrite("spectral_samples",  &RenderConfig::spectral_samples)
        .def_readwrite("pupil_jitter",      &RenderConfig::pupil_jitter)
        .def_readwrite("jitter_seed",       &RenderConfig::jitter_seed)
        .def_readwrite("sensor_model",      &RenderConfig::sensor_model)
        .def_readwrite("input_cs",          &RenderConfig::input_cs)
        .def_readwrite("output_cs",         &RenderConfig::output_cs)
        .def_readwrite("sensor_half_w",     &RenderConfig::sensor_half_w)
        .def_readwrite("sensor_half_h",     &RenderConfig::sensor_half_h)
        .def_readwrite("verbose",           &RenderConfig::verbose)
        .def_readwrite("collect_stats",     &RenderConfig::collect_stats,
            "Diagnostic: fill the output dict's 'stats' entry with stage "
            "timing + per-pair survivor counters. Adds a second instrumented kernel "
            "launch; off by default. Render output is unchanged either way.")
        .def_readwrite("spectral_correction", &RenderConfig::spectral_correction,
            "When true (default), apply a per-render output-space 3x3 to the ghost "
            "buffers so any spectral_samples count matches the converged colour and "
            "a fixed reference brightness (the High preset) - fixes the Low-greenish/"
            "dim vs High-bluer/brighter sample-count drift. Corrects only the CMF "
            "integration bias; physical dispersion tint is kept. Off = raw drift.")
        .def_property("custom_input_to_xyz",
            [](const RenderConfig& c) {
                return py::array_t<float>({3, 3}, {3 * sizeof(float), sizeof(float)},
                                          &c.custom_input_to_xyz[0][0]);
            },
            [](RenderConfig& c, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3)
                    throw py::value_error("expected a (3, 3) array");
                std::copy_n(arr.data(), 9, &c.custom_input_to_xyz[0][0]);
            })
        .def_property("custom_xyz_to_output",
            [](const RenderConfig& c) {
                return py::array_t<float>({3, 3}, {3 * sizeof(float), sizeof(float)},
                                          &c.custom_xyz_to_output[0][0]);
            },
            [](RenderConfig& c, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3)
                    throw py::value_error("expected a (3, 3) array");
                std::copy_n(arr.data(), 9, &c.custom_xyz_to_output[0][0]);
            });

    // ----------------------------------------------------------- MatteBox
    py::class_<MatteBox>(m, "MatteBox")
        .def(py::init<>())
        .def_readwrite("enabled",    &MatteBox::enabled)
        .def_readwrite("z_front_mm", &MatteBox::z_front_mm,
            "Flag plane distance ahead of the front vertex (mm, toward the scene).")
        .def_readwrite("top",    &MatteBox::top,    "+y half-extent (mm); MATTE_BOX_OPEN = unflagged")
        .def_readwrite("bottom", &MatteBox::bottom, "-y half-extent (mm)")
        .def_readwrite("left",   &MatteBox::left,   "-x half-extent (mm)")
        .def_readwrite("right",  &MatteBox::right,  "+x half-extent (mm)");

    // ----------------------------------------------------------- Baffle
    py::enum_<BaffleShape>(m, "BaffleShape")
        .value("RECT",   BaffleShape::RECT)
        .value("CIRCLE", BaffleShape::CIRCLE);

    py::class_<Baffle>(m, "Baffle")
        .def(py::init<>())
        .def_readwrite("shape",      &Baffle::shape,
            "BaffleShape.RECT (top/bottom/left/right half-extents) or "
            "BaffleShape.CIRCLE (cx/cy/radius/aspect ellipse).")
        .def_readwrite("z_front_mm", &Baffle::z_front_mm,
            "Plane distance ahead of the front vertex (mm).")
        .def_readwrite("top",    &Baffle::top,    "RECT +y half-extent (mm); MATTE_BOX_OPEN = unflagged")
        .def_readwrite("bottom", &Baffle::bottom, "RECT -y half-extent (mm)")
        .def_readwrite("left",   &Baffle::left,   "RECT -x half-extent (mm)")
        .def_readwrite("right",  &Baffle::right,  "RECT +x half-extent (mm)")
        .def_readwrite("cx",     &Baffle::cx,     "CIRCLE centre x offset (mm)")
        .def_readwrite("cy",     &Baffle::cy,     "CIRCLE centre y offset (mm)")
        .def_readwrite("radius", &Baffle::radius, "CIRCLE radius (mm)")
        .def_readwrite("aspect", &Baffle::aspect, "CIRCLE ellipse y-scale (1 = circle)");
    m.attr("MATTE_BOX_OPEN") = MATTE_BOX_OPEN;

    // ---------------------------------------------------------- DiffractionConfig
    py::class_<DiffractionConfig>(m, "DiffractionConfig")
        .def(py::init<>())
        .def_readwrite("starburst",      &DiffractionConfig::starburst,
            "Master switch for the aperture-diffraction starburst pass. Off by "
            "default; with it off no starburst is drawn.")
        .def_readwrite("starburst_engine", &DiffractionConfig::starburst_engine,
            "Which engine draws the starburst. StarburstEngine.SPRITE (default) "
            "is the fast FFT-sprite path; StarburstEngine.MDFT evaluates the "
            "diffraction integral exactly at the sensor pixels (resample-free, "
            "flux-matched, no sub-pixel-motion artefacts).")
        .def_readwrite("starburst_grid", &DiffractionConfig::starburst_grid,
            "FFT grid side; power of two >= 256. 1024 is the quality/speed default.")
        .def_readwrite("starburst_grid_cap", &DiffractionConfig::starburst_grid_cap,
            "Ceiling for auto-extent grid growth when one FFT period does not "
            "cover the sensor. Rounded down to a power of two and clamped to "
            "[starburst_grid, 16384].")
        .def_readwrite("pupil_fill",     &DiffractionConfig::pupil_fill,
            "Entrance-pupil radius as a fraction of the FFT half-grid; padding "
            "factor Q = 1/pupil_fill (Q >= 2 required). Smaller = finer detail, "
            "smaller sensor window.")
        .def_readwrite("starburst_gain", &DiffractionConfig::starburst_gain,
            "Artistic brightness multiplier on the physically-scaled starburst.")
        .def_readwrite("scale_trim",     &DiffractionConfig::scale_trim,
            "Artistic multiplier on the physical sensor pitch (1.0 = physically "
            "scaled, often sub-pixel at preview resolution; >1 enlarges without "
            "changing structure).")
        .def_readwrite("spectral_samples", &DiffractionConfig::spectral_samples,
            "Wavelengths integrated for dispersion; 0 = inherit "
            "RenderConfig.spectral_samples. 16-32 gives smooth rainbow spikes.")
        .def_readwrite("use_survivor_mask", &DiffractionConfig::use_survivor_mask,
            "Clip the pupil by the ray-traced survivor set so the starburst morphs "
            "into the off-axis cat's-eye and dims by the physical vignetting "
            "throughput. On axis it is a no-op (the whole pupil survives).")
        .def_readwrite("survivor_grid",  &DiffractionConfig::survivor_grid,
            "Side of the CPU-traced cat's-eye envelope grid (per axis). Only needs "
            "to resolve a smooth vignette shape; 192 (~a few ms) is ample.")
        .def_readwrite("use_surface_textures", &DiffractionConfig::use_surface_textures,
            "Fold the front-most APERTURE_IMAGE surface's bitmap into the pupil as "
            "a graded amplitude (dust / smudges diffract), matching the tracer's UV.")
        .def_readwrite("veil",           &DiffractionConfig::veil,
            "Render the physical veiling-glare halo as a separate additive "
            "energy-conserving spectral glare-spread "
            "function. Off (default) = no veil pass runs; the veil buffer stays empty.")
        .def_readwrite("veil_gain",      &DiffractionConfig::veil_gain,
            "Total veil energy as a fraction of the source's own flux (the veil is "
            "the energy-normalised GSF times this). 0.03 puts ~3% of the source "
            "into the halo. NOT scaled by flare_gain.")
        .def_readwrite("veil_spread",    &DiffractionConfig::veil_spread,
            "GSF core radius on the sensor as a fraction of the sensor half-height, "
            "so the halo scales with the frame. Larger washes more of the frame. "
            "Clamped to [1e-3, 4].")
        .def_readwrite("veil_falloff",   &DiffractionConfig::veil_falloff,
            "Radial falloff power p of the generalized Lorentzian g=(a^2/(r^2+a^2))^p. "
            "p=1 is a pure Lorentzian (broadest tails); 1.5 gives a ~1/r^3 glare "
            "tail. Clamped to [1, 3].")
        .def_readwrite("matte_box",      &DiffractionConfig::matte_box)
        .def_readwrite("baffles",        &DiffractionConfig::baffles,
            "Front-of-lens baffle stack (hoods / mount rims / casing) as a list of "
            "Baffle. Empty by default; the matte box is merged in as one RECT baffle "
            "at trace time, so an empty list + disabled matte box is a total no-op. "
            "Assign a whole list (in-place append won't persist).")
        .def_readwrite("hurb",           &DiffractionConfig::hurb,
            "HURB edge diffraction: kick each ghost ray that passes an edge "
            "by a random angle ~ lambda/d, reproducing the diffraction glow around "
            "the stop, rims, and baffles. "
            "Off (default) is a total no-op (compile-time templated -> 0 cost off).")
        .def_readwrite("hurb_kick",      &DiffractionConfig::hurb_kick,
            "HURB kick distribution: HurbKickDistribution.LORENTZIAN (default; heavy "
            "1/theta^2 glare tails) or .GAUSSIAN (conservative sinc^2-core match).")
        .def_readwrite("hurb_min_sigma_rad", &DiffractionConfig::hurb_min_sigma_rad,
            "Skip the kick when a far edge's sigma falls below this (radians).")
        .def_readwrite("hurb_max_kick_rad",  &DiffractionConfig::hurb_max_kick_rad,
            "Hard clamp on the kick magnitude (radians); bounds the Lorentzian tail.");

    // ------------------------------------------------------------- GateConfig
    py::enum_<GateLobe>(m, "GateLobe")
        .value("CAUCHY_ISO", GateLobe::CAUCHY_ISO)
        .value("GROOVED",    GateLobe::GROOVED)
        .export_values();

    py::class_<GateConfig>(m, "GateConfig")
        .def(py::init<>())
        .def_readwrite("enabled",         &GateConfig::enabled,
            "Render the film-gate scatter layer (a separate additive buffer, "
            "render_gate). Off (default) = no gate pass runs; the gate buffer "
            "stays empty and the output dict omits its keys.")
        .def_readwrite("thickness_mm",    &GateConfig::thickness_mm,
            "Depth of the cut through the aperture plate (mm). Sets the capture "
            "band t/(2N) at f-number N, so it drives the layer's ENERGY and how "
            "wide a source-position window fires at all. Not the reach knob.")
        .def_readwrite("standoff_mm",     &GateConfig::standoff_mm,
            "Gap from the wall's rear edge to the sensor plane (mm). The lever "
            "arm the scattered ray travels, so this sets REACH into frame — but "
            "the pupil band narrows with it, so reach trades against energy.")
        .def_readwrite("offset_left_mm",   &GateConfig::offset_left_mm,
            "Offset (mm) on the -x side of the opening, which otherwise follows "
            "the render's sensor extent. Positive opens outward, negative crops "
            "inward; each side is floored so the opening cannot invert.")
        .def_readwrite("offset_right_mm",  &GateConfig::offset_right_mm,
            "Offset (mm) on the +x side of the opening. See offset_left_mm.")
        .def_readwrite("offset_bottom_mm", &GateConfig::offset_bottom_mm,
            "Offset (mm) on the -y side of the opening. See offset_left_mm.")
        .def_readwrite("offset_top_mm",    &GateConfig::offset_top_mm,
            "Offset (mm) on the +y side of the opening. See offset_left_mm.")
        .def_readwrite("lobe",            &GateConfig::lobe,
            "Scatter distribution off the wall: GateLobe.GROOVED (default; "
            "Cauchy across the machining marks, Gaussian along them, so the "
            "streak runs into frame) or .CAUCHY_ISO (isotropic, reads as a blob).")
        .def_readwrite("roughness_rad",   &GateConfig::roughness_rad,
            "Cauchy HWHM (radians) across the marks. Direction only, so it is "
            "energy-conserving: it sets SPREAD, never brightness. Clamped to "
            "[0, 0.5]; 0 makes the wall a perfect mirror, which folds a point "
            "source back to a point.")
        .def_readwrite("groove_aniso",    &GateConfig::groove_aniso,
            "Width along the marks as a fraction of roughness_rad. 0.12 (default) "
            "is the ~8:1 anisotropy that reads as a machined cut; 1.0 is round. "
            "Clamped to [1e-3, 1].")
        .def_readwrite("groove_angle_deg", &GateConfig::groove_angle_deg,
            "Orientation of the machining marks (degrees), measured in each "
            "wall's own frame from that wall's frame-edge tangent, so all four "
            "walls rotate coherently. 0 = marks along the edge, fan into frame.")
        .def_readwrite("max_kick_rad",    &GateConfig::max_kick_rad,
            "Hard clamp on one sampled kick (radians); bounds the Cauchy tail so "
            "a single draw cannot cross the frame. Clamped to [1e-4, 1.5].")
        .def_readwrite("reflectance_r0",  &GateConfig::reflectance_r0,
            "Schlick reflectance at NORMAL incidence: 0.04 blackened metal, 0.6 "
            "bare aluminium. Grazing Fresnel takes R toward 1 regardless, so this "
            "moves brightness far less than it looks like it should.")
        .def_readwrite("gain",            &GateConfig::gain,
            "Artistic multiplier on the physical brightness. At 1.0 the layer "
            "carries its traced fraction of the source's flux, which measures "
            "0.02-0.33% of the peak source. NOT scaled by flare_gain.")
        .def_readwrite("scatter_samples", &GateConfig::scatter_samples,
            "Lobe draws per scraping (ray, wavelength, wall). Each carries 1/n of "
            "the weight, so this changes smoothness, not energy. Clamped [1, 64].")
        .def_readwrite("spectral_samples", &GateConfig::spectral_samples,
            "Wavelengths integrated; 0 (default) inherits RenderConfig."
            "spectral_samples. The wall is achromatic but the trace that decides "
            "the scrape is dispersive, so each wavelength crosses the edge at its "
            "own source position and the flare walks through hue as light moves.");

    // ----------------------------------------------------------- FlareConfig
    py::class_<FlareConfig, RenderConfig>(m, "FlareConfig")
        .def(py::init<>())
        .def_readwrite("aperture_blades",     &FlareConfig::aperture_blades,
            "override; if 0, the entrance-pupil sampling mask is derived from "
            "the is_stop surface's aperture_shape")
        .def_readwrite("aperture_rotation",   &FlareConfig::aperture_rotation,
            "degrees; only meaningful when aperture_blades >= 3 forces a polygon mask")
        .def_readwrite("flare_gain",          &FlareConfig::flare_gain)
        .def_readwrite("min_ghost_intensity", &FlareConfig::min_ghost_intensity)
        .def_readwrite("ghost_normalize",     &FlareConfig::ghost_normalize)
        .def_readwrite("max_area_boost",      &FlareConfig::max_area_boost)
        .def_readwrite("cull_dead_pairs",     &FlareConfig::cull_dead_pairs,
            "Skip ghost pairs whose coarse GPU probe finds zero rays "
            "reaching the sensor for any source/wavelength (the big off-axis "
            "saving). Default true; provably safe (dead pairs contribute nothing). "
            "false = trace every pair.")
        .def_readwrite("concentrate_samples", &FlareConfig::concentrate_samples,
            "Lay each (pair, source)'s sample budget over its probe-"
            "measured pupil survivor box instead of the full pupil. Same ray "
            "count and the same estimator in expectation (weights scale by box "
            "area; mask-rejected samples count as zeros), but variance drops by "
            "the pupil/box area ratio - dramatically cleaner off-axis ghosts. "
            "Thin probe evidence falls back to full-pupil sampling (never a hard "
            "cull). Default true; clear to sample the full pupil uniformly.")
        .def_readwrite("adaptive_sample_budgets", &FlareConfig::adaptive_sample_budgets,
            "Scale each concentrated (pair, source)'s sample COUNT "
            "to its survivor-rect area (harvesting concentration's over-sampling as "
            "speed), reweighting so the estimator mean is unchanged. Requires "
            "concentrate_samples. Default TRUE; clear to fall back to the full "
            "concentration sample count.")
        .def_readwrite("adaptive_density_boost", &FlareConfig::adaptive_density_boost,
            "Pupil sample density relative to the full-grid path when "
            "adaptive_sample_budgets is on (quality margin; higher trades speed "
            "for lower variance). Default 8.0.")
        .def_readwrite("adaptive_min_samples", &FlareConfig::adaptive_min_samples,
            "Absolute per-(pair, source) sample floor under adaptive_sample_budgets "
            "so a dim ghost is never starved below visibility.")
        .def_readwrite("aov_mode",            &FlareConfig::aov_mode)
        .def_readwrite("aov_max_pairs",       &FlareConfig::aov_max_pairs)
        .def_readwrite("ghost_filter",        &FlareConfig::ghost_filter,
            "Ghost-pair selection filter. Default mode ALL is a "
            "no-op; INCLUDE / EXCLUDE consult `pairs` (list of (surf_a, "
            "surf_b) tuples).")
        .def_readwrite("diffraction",         &FlareConfig::diffraction,
            "Aperture-diffraction settings (DiffractionConfig). Default "
            "(starburst off) draws no diffraction pass.")
        .def_readwrite("gate",                &FlareConfig::gate,
            "Film-gate flare settings (GateConfig) — mechanical scatter off the "
            "aperture plate's cut edge. A sibling of `diffraction`, not part of "
            "it. Default (enabled off) runs no gate pass and emits no gate layer.");

    // ------------------------------------------------------ PointFlareConfig
    py::class_<PointFlareConfig, FlareConfig>(m, "PointFlareConfig")
        .def(py::init<>())
        .def_readwrite("source_x", &PointFlareConfig::source_x)
        .def_readwrite("source_y", &PointFlareConfig::source_y)
        .def_readwrite("source_r", &PointFlareConfig::source_r)
        .def_readwrite("source_g", &PointFlareConfig::source_g)
        .def_readwrite("source_b", &PointFlareConfig::source_b);

    // ------------------------------------------------------- PSFCenterMode
    py::enum_<PSFCenterMode>(m, "PSFCenterMode")
        .value("CHIEF_CENTROID", PSF_CENTER_CHIEF_CENTROID)
        .value("FIXED_TARGET",   PSF_CENTER_FIXED_TARGET)
        .export_values();

    // ----------------------------------------------------------- PSFConfig
    py::class_<PSFConfig, RenderConfig>(m, "PSFConfig")
        .def(py::init<>())
        .def_readwrite("grid_nx",          &PSFConfig::grid_nx)
        .def_readwrite("grid_ny",          &PSFConfig::grid_ny)
        .def_readwrite("tile_w",           &PSFConfig::tile_w)
        .def_readwrite("tile_h",           &PSFConfig::tile_h)
        .def_readwrite("tile_extent_mm",   &PSFConfig::tile_extent_mm)
        .def_readwrite("monochromatic",    &PSFConfig::monochromatic)
        .def_readwrite("center_mode",      &PSFConfig::center_mode,
            "PSFCenterMode: CHIEF_CENTROID (pupil-mean anchor) or "
            "FIXED_TARGET (aim the chief ray at each source's target_x/y_mm). "
            "In FIXED_TARGET mode the sensor is carried by sensor_half_w/h.")
        .def_readwrite("splat_sigma_um",   &PSFConfig::splat_sigma_um,
            "Gaussian per-ray spot size in MICROMETRES on the sensor (physical, "
            "so the rendered spot is independent of tile resolution); "
            "<=0 = single-pixel bilinear (raw).")
        .def_readwrite("aperture_blades",  &PSFConfig::aperture_blades,
            "override; if 0, the entrance-pupil sampling mask is derived from "
            "the is_stop surface's aperture_shape")
        .def_readwrite("aperture_rotation",&PSFConfig::aperture_rotation,
            "degrees; only meaningful when aperture_blades >= 3 forces a polygon mask");
}
