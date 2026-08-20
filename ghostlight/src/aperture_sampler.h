// ============================================================================
// aperture_sampler.h — Entrance-pupil sample grid generation.
//
// Shared by ghost_render.cu, psf_render.cu, and starburst_render.cu so all
// renderers see the same Halton sequence, polygon mask, and anamorphic stretch.
// Header-inline so it can be #included from .cu and .cpp without extra link rules.
// ============================================================================
#pragma once

#include "optical_system.h"

#include <cmath>
#include <cstdint>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// One sample in normalised entrance-pupil coordinates; both axes in [-1, 1].
struct ApertureSample { float u, v; };

// Inputs to the sampler that the caller pulls from its config struct(s).
// Kept neutral (not RenderConfig / FlareConfig) so renderers that don't
// inherit FlareConfig can still drive it.
struct ApertureSamplerParams
{
    // RenderConfig fields
    int   ray_grid          = 64;   // grid resolution (samples = ray_grid * ray_grid before clip)
    int   pupil_jitter      = 0;    // 0 = regular, 1 = wang-hash jitter, 2 = Halton
    int   jitter_seed       = 0;    // seed offset for pupil_jitter == 1

    // Optional overrides for the lens-derived stop shape.
    // blades_override == 0 → derive from is_stop surface.
    // blades_override >= 3 → force polygon of N blades, with rotation_deg
    //                        controlling orientation.  Aspect is always taken
    //                        from the stop surface (config has no aspect knob).
    int   blades_override   = 0;
    float rotation_deg      = 0.0f;
};

// Entrance-pupil acceptance mask in normalised [-1,1]^2 coords: the anamorphic
// unit disk, optionally clipped to an inscribed regular polygon. This is the
// SINGLE source of truth for the pupil silhouette — build_aperture_samples()
// (host grid build), the cull probe, and the main scatter kernel all test the
// SAME predicate, so a concentrated / remapped grid can never disagree with the
// full grid about the silhouette (avoiding mask-drift). POD +
// __host__ __device__ so it can be passed by value into a kernel.
struct PupilMask
{
    float aspect     = 1.0f;   // anamorphic stretch on U (stop aperture_aspect)
    int   n_blades   = 0;      // >= 3 → polygon; else circular
    float rot_rad    = 0.0f;   // polygon rotation
    float apothem    = 1.0f;   // cos(pi/n_blades)   (precomputed; 1 if circular)
    float sector_ang = 1.0f;   // 2pi/n_blades       (precomputed; 1 if circular)

    // Copied from the stop when its blade shape is authored; inactive
    // (blades = 0) otherwise, including under a blades_override, which forces a
    // plain n-gon by definition.
    ApertureProfile profile;

    // Uses the same silhouette test as build_aperture_samples().
    __host__ __device__ bool contains(float u, float v) const
    {
        const float ux = u / aspect;
        const float r2 = ux * ux + v * v;
        if (r2 > 1.0f) return false;
        if (n_blades >= 3)
        {
            // Deformed blades: the same normalised r(theta) check_aperture()
            // runs, so the pre-filter and trace agree on the silhouette.
            if (profile.deformed())
                return sqrtf(r2) <= profile.radius_at(atan2f(v, ux));

            float angle  = atan2f(v, ux) - rot_rad;
            float sector = fmodf(angle, sector_ang);
            if (sector < 0.0f) sector += sector_ang;
            if (sqrtf(r2) * cosf(sector - sector_ang * 0.5f) > apothem)
                return false;
        }
        return true;
    }
};

// Resolve the pupil mask from the lens stop shape + the sampler's optional
// blades/rotation override. Host-only (reads lens.surfaces).
inline PupilMask resolve_pupil_mask(const OpticalSystem& lens,
                                    const ApertureSamplerParams& p)
{
    int   n_blades = 0;
    float rot_rad  = 0.0f;
    float aspect   = 1.0f;
    ApertureProfile profile;
    {
        const Surface* stop = nullptr;
        for (const auto& s : lens.surfaces)
            if (s.is_stop) { stop = &s; break; }

        if (stop)
        {
            aspect = stop->aperture_aspect;
            if (stop->aperture_shape == APERTURE_POLYGON
                && stop->aperture_blades >= 3)
            {
                n_blades = stop->aperture_blades;
                rot_rad  = stop->aperture_rotation_rad;
                profile  = stop->aperture_profile;
            }
            // APERTURE_IMAGE / APERTURE_CIRCLE / no stop: leave the mask
            // circular.  Image-aperture masks would require bitmap sampling
            // here; the trace-time check is left to do the real test.
        }
    }
    if (p.blades_override > 0)
    {
        n_blades = p.blades_override;
        rot_rad  = p.rotation_deg * ((float)M_PI / 180.0f);
        // Aspect is intentionally still the lens-derived value. The override
        // describes a plain n-gon and has no blade-shape knobs, so it also
        // drops any profile the stop carried.
        profile  = ApertureProfile{};
    }

    PupilMask m;
    m.aspect     = aspect;
    m.n_blades   = n_blades;
    m.rot_rad    = rot_rad;
    m.profile    = profile;
    const bool polygonal = (n_blades >= 3);
    m.apothem    = polygonal ? cosf((float)M_PI / n_blades) : 1.0f;
    m.sector_ang = polygonal ? (2.0f * (float)M_PI / n_blades) : 1.0f;
    return m;
}

namespace aperture_sampler_detail {

// Wang hash — used for per-sample jitter when pupil_jitter == 1.
// __host__ __device__ so the concentrated kernel path can generate the sample
// stream on the GPU (macros are empty in host-only translation units).
__host__ __device__ inline uint32_t wang_hash(uint32_t s)
{
    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s ^= s >> 4u;
    s *= 0x27d4eb2du;
    s ^= s >> 15u;
    return s;
}

// Radical-inverse base 2 (bit reversal) — Halton low-discrepancy sequence.
__host__ __device__ inline float halton2(uint32_t n)
{
    n = (n << 16u) | (n >> 16u);
    n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
    n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
    n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
    n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
    return (float)n * (1.0f / 4294967296.0f);
}

// Radical-inverse base 3 — Halton companion to halton2().
__host__ __device__ inline float halton3(uint32_t n)
{
    float r = 0.0f, f = 1.0f / 3.0f;
    while (n > 0) { r += (n % 3u) * f; n /= 3u; f /= 3.0f; }
    return r;
}

} // namespace aperture_sampler_detail

// Build the entrance-pupil sample grid for one render.
//
// Resolution: out.size() <= p.ray_grid * p.ray_grid.  Samples outside the
// circular / polygon mask (after anamorphic stretch) are rejected, so the
// returned count is typically ~π/4 * N² for a round stop.
//
// The downstream check_aperture() during the trace is the authoritative
// world-space aperture test; this sampler is a pre-filter to avoid wasting
// rays that obviously won't survive.
inline std::vector<ApertureSample>
build_aperture_samples(const OpticalSystem& lens,
                       const ApertureSamplerParams& p)
{
    using namespace aperture_sampler_detail;

    const PupilMask mask = resolve_pupil_mask(lens, p);

    const int      N           = p.ray_grid;
    const int      jitter      = p.pupil_jitter;
    const uint32_t seed_offset = (uint32_t)p.jitter_seed * 1000003u;

    std::vector<ApertureSample> grid_samples;
    if (N <= 0) return grid_samples;
    grid_samples.reserve((size_t)N * N);

    for (int k = 0; k < N * N; ++k)
    {
        const int gx = k % N;
        const int gy = k / N;

        float u, v;
        if (jitter == 2)
        {
            u = halton2((uint32_t)k) * 2.0f - 1.0f;
            v = halton3((uint32_t)k) * 2.0f - 1.0f;
        }
        else
        {
            float ju = (jitter == 1) ? wang_hash((uint32_t)k + seed_offset)
                                           / 4294967296.0f : 0.5f;
            float jv = (jitter == 1) ? wang_hash((uint32_t)k + (uint32_t)(N * N) + seed_offset)
                                           / 4294967296.0f : 0.5f;
            u = ((gx + ju) / N) * 2.0f - 1.0f;
            v = ((gy + jv) / N) * 2.0f - 1.0f;
        }

        // Same silhouette predicate the probe + scatter kernel use (the shared
        // anamorphic-disk / polygon test).
        if (mask.contains(u, v))
            grid_samples.push_back({u, v});
    }
    return grid_samples;
}
