// ============================================================================
// aperture_sdf.h — Signed-distance-field bake for APERTURE_IMAGE stops.
//
// HURB edge diffraction (hurb.h) needs a perpendicular edge distance (mm) and
// a transverse edge normal at each surface. Analytic apertures get these from
// aperture_edge_distance() (trace_core.h); a bitmap aperture has no analytic
// edge, so this module bakes a signed distance field + gradient normal from the
// >0.5-thresholded matte. The float4 result [d_mm, nx, ny, 0] is uploaded as a
// texture and sampled on the GPU (see upload_aperture_sdf_textures / trace_cuda.h).
//
// Pure host C++ (no CUDA) so the bake is unit-testable via the _aperture_sdf_debug
// pybind entry point. Scope: ONLY the aperture stop when it is an image aperture
// (find_sdf_target_surface) — front-element "dirt" textures diffract via the
// starburst pupil fold, not HURB.
// ============================================================================
#pragma once

#include "optical_system.h"

#include <vector>

// One baked SDF for a single image-aperture surface.
struct ApertureSdfBake
{
    int   width  = 0;
    int   height = 0;
    // Interleaved float4 texels [d_mm, nx, ny, 0], row-major (W*H*4 floats).
    // d_mm is signed (positive inside the pass region); (nx,ny) is the unit
    // transverse edge normal (world frame; sign is irrelevant to the kick).
    std::vector<float> texels;
    float sx = 0.0f;   // mm per texel along X (world) = 2*aspect*semi_d/width
    float sy = 0.0f;   // mm per texel along Y (world) = 2*semi_d/height
};

// Index of the surface eligible for SDF edge kicks: the is_stop surface iff it
// is APERTURE_IMAGE with loaded pixels. Returns -1 otherwise. Single source of
// truth for the stop-only scoping decision.
int find_sdf_target_surface(const OpticalSystem& lens);

// Bake a signed distance field + unit edge normal from `img` (thresholded at
// luminance > 0.5, matching the trace matte). semi_diameter_mm / aspect set the
// world mm-per-texel scale to match the trace UV convention. Returns false on
// degenerate input (empty/zero-size image); `out` is left cleared.
bool bake_aperture_sdf(const ApertureImage& img, float semi_diameter_mm,
                       float aspect, ApertureSdfBake& out);
