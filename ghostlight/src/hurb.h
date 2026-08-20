// ============================================================================
// hurb.h — Heisenberg Uncertainty Ray Bending edge diffraction.
//
// Physically models a lens's soft chromatic edge-diffraction glow. A ghost ray
// that survives passing an edge at perpendicular distance d gets a random
// angular kick, perpendicular to that edge, of scale
//
//     sigma ~ lambda * K / d          (radians, small-angle)
//
// with a distribution-specific constant K (HURB_GAUSS_K / HURB_LORENTZ_K). The
// kick changes direction only, so energy is conserved; the aggregate of many
// kicks reproduces the diffraction ENVELOPE around each edge (no fringes). It is
// chromatic (sigma scales with lambda), so it throws a coloured glow.
//
// These free functions carry the trace_core.h-style __host__/__device__ guard,
// so they compile both in the ghost kernel (.cu) and in the host-side
// _hurb_sample_debug validation entry point.
// ============================================================================
#pragma once

#include "render_config.h"   // HurbKickDistribution, DiffractionConfig
#include "ray.h"             // Vec3f, dot (for hurb_apply_kick)

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__CUDACC__)
  #define GHL_HURB_HD __host__ __device__
#else
  #define GHL_HURB_HD
#endif

// Distribution-specific calibration constants for sigma = lambda * K / d.
//   Gaussian   : 1 / (2*sqrt(2)*pi)   (Freniere/Groot/Gregory 1999)
//   Lorentzian : 1 / (2*pi)
constexpr float HURB_GAUSS_K   = 0.112539539f;   // 1/(2*sqrt(2)*pi)
constexpr float HURB_LORENTZ_K = 0.159154943f;   // 1/(2*pi)

GHL_HURB_HD inline float hurb_k_for(int dist)
{
    return (dist == (int)HurbKickDistribution::LORENTZIAN) ? HURB_LORENTZ_K
                                                           : HURB_GAUSS_K;
}

// Angular kick scale (radians) at perpendicular edge distance d_mm and this
// wavelength. Small-angle, so this is directly the Gaussian std / Lorentzian
// HWHM. d_mm <= 0 (on or past the edge) returns 0 — the caller guards those.
GHL_HURB_HD inline float hurb_sigma(float lambda_nm, float d_mm, int dist)
{
    if (d_mm <= 0.0f) return 0.0f;
    const float lam_mm = lambda_nm * 1.0e-6f;    // nm -> mm
    return lam_mm * hurb_k_for(dist) / d_mm;
}

// Deterministic RNG: a wang-hash to seed, then a small xorshift stream. Matches
// the wang-hash idiom the aperture sampler / ghost seeds use, so a HURB stream
// is reproducible from a per-ray seed (source-angle bits ^ grid ^ pair ^ jitter).
GHL_HURB_HD inline unsigned int hurb_wang_hash(unsigned int s)
{
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15);
    return s;
}

// Next uniform in [0, 1) from a 32-bit xorshift state (advances the state).
GHL_HURB_HD inline float hurb_next_uniform(unsigned int& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Sample a 1-D kick angle (radians) from the chosen distribution with scale
// sigma, hard-clamped to +/- max_kick. Advances the RNG state.
//   LORENTZIAN : Cauchy(scale = sigma) via the inverse CDF, tan(pi*(u-0.5)).
//   GAUSSIAN   : N(0, sigma^2) via Box-Muller.
// Clamping bounds the heavy Cauchy tail (a near-grazing edge ray). Direction
// only, so no energy is created.
GHL_HURB_HD inline float hurb_sample_kick(int dist, float sigma, float max_kick,
                                         unsigned int& state)
{
    float k;
    if (dist == (int)HurbKickDistribution::LORENTZIAN) {
        const float u = hurb_next_uniform(state);
        k = sigma * tanf((float)M_PI * (u - 0.5f));
    } else {
        const float u1 = fmaxf(hurb_next_uniform(state), 1.0e-7f);
        const float u2 = hurb_next_uniform(state);
        k = sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
    if (k >  max_kick) k =  max_kick;
    if (k < -max_kick) k = -max_kick;
    return k;
}

// Runtime HURB parameters — the by-value kernel param used when HURB is compiled
// in (ghost_kernel<STATS, true>). OFF (on == 0) never reaches the kernel path.
struct GpuHurb {
    int   on            = 0;
    int   dist          = (int)HurbKickDistribution::LORENTZIAN;
    float min_sigma_rad = 1.0e-5f;
    float max_kick_rad  = 0.35f;
};

// Apply one HURB edge kick to `dir` in place, given the (world-frame) outward
// edge normal `n_edge` and perpendicular edge distance `d_mm` at the hit (as
// returned by aperture_edge_distance in trace_core.h). A far edge — where the
// kick scale sigma falls at/below h.min_sigma_rad — is skipped WITHOUT drawing
// from the RNG (the cheap common case; and geometry is identical across chunks,
// so this keeps the stream reproducible). The kick rotates the direction inside
// the plane spanned by `dir` and `n_edge`, so |dir| is preserved and no energy
// is created. A ray travelling parallel to the edge normal (no transverse
// component) is left unchanged.
GHL_HURB_HD inline void hurb_apply_kick(const GpuHurb& h, Vec3f& dir,
                                       const Vec3f& n_edge, float d_mm,
                                       float lambda_nm, unsigned int& state)
{
    const float sigma = hurb_sigma(lambda_nm, d_mm, h.dist);
    if (sigma <= h.min_sigma_rad) return;                 // far edge -> no kick

    const float theta = hurb_sample_kick(h.dist, sigma, h.max_kick_rad, state);

    // Transverse kick direction: component of the edge normal perpendicular to
    // the ray. Degenerates only when dir is parallel to n_edge (skip).
    Vec3f t = n_edge - dir * dot(n_edge, dir);
    const float tl = sqrtf(dot(t, t));
    if (tl < 1.0e-8f) return;
    t = t * (1.0f / tl);

    Vec3f nd = dir + t * tanf(theta);
    const float nl = sqrtf(dot(nd, nd));
    if (nl < 1.0e-12f) return;
    dir = nd * (1.0f / nl);
}

// Host: resolve the runtime HURB params from the render config.
inline GpuHurb build_gpu_hurb(const DiffractionConfig& dc)
{
    GpuHurb h;
    h.on            = dc.hurb ? 1 : 0;
    h.dist          = (int)dc.hurb_kick;
    h.min_sigma_rad = dc.hurb_min_sigma_rad;
    h.max_kick_rad  = dc.hurb_max_kick_rad;
    return h;
}
