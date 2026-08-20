// ============================================================================
// gate.h — Film-gate scatter geometry and lobe sampling.
//
// The aperture plate's opening is cut through metal, so its edge is a WALL
// parallel to the optical axis, standing `standoff` ahead of the sensor and
// `thickness` deep. A primary ray that would land just outside the opening
// strikes that wall at near-grazing incidence, reflects, and lands back inside
// the frame at the mirror fold. See GateConfig in render_config.h for the full
// model and for what each knob controls.
//
// These free functions carry the trace_core.h-style __host__/__device__ guard so
// the same code runs in the gate kernel (.cu) and in the host-side
// _gate_sample_debug validation entry point, exactly as hurb.h does.
//
// The local RNG keeps gate sampling independent of ghost-kernel headers.
// ============================================================================
#pragma once

#include "render_config.h"   // GateConfig, GateLobe
#include "vec3.h"            // Vec3f

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__CUDACC__)
  #define GHL_GATE_HD __host__ __device__
#else
  #define GHL_GATE_HD
#endif

// Runtime gate parameters — the by-value kernel param. All geometry is already
// resolved to sensor millimetres and every clamp applied, so the kernel does no
// config interpretation. Compare GpuHurb (hurb.h) at 16 bytes.
struct GpuGate {
    // Opening edges in sensor mm, per-side offsets already folded in. Signed
    // plane coordinates: x_neg < 0 < x_pos.
    float x_pos = 0.0f, x_neg = 0.0f;
    float y_pos = 0.0f, y_neg = 0.0f;
    float t_mm  = 0.8f;          // wall depth
    float zs_mm = 5.0f;          // standoff: wall rear edge -> sensor plane
    float sig_wide  = 0.08f;     // Cauchy HWHM across the machining marks (rad)
    float sig_tight = 0.0096f;   // width along them (rad)
    float cos_g = 1.0f;          // groove rotation in each wall's own frame
    float sin_g = 0.0f;
    float r0 = 0.04f;            // Schlick reflectance at normal incidence
    float max_kick = 0.35f;      // clamp on one sampled kick (rad)
    float inv_scatter = 0.25f;   // 1 / scatter_samples, pre-divided
    float gain = 1.0f;
    int   n_scatter = 4;
    int   lobe = (int)GateLobe::GROOVED;
};

// Deterministic RNG. See the duplication note in the file header.
GHL_GATE_HD inline unsigned int gate_wang_hash(unsigned int s)
{
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15);
    return s;
}

// Next uniform in [0, 1) from a 32-bit xorshift state (advances the state).
GHL_GATE_HD inline float gate_next_uniform(unsigned int& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// Cauchy(scale = sigma) via the inverse CDF, hard-clamped. The clamp bounds a
// tail that is otherwise unbounded — one draw must not fling a ray across the
// frame — and is why the deposited energy stays finite.
GHL_GATE_HD inline float gate_sample_cauchy(float sigma, float max_kick,
                                           unsigned int& state)
{
    const float u = gate_next_uniform(state);
    float k = sigma * tanf((float)M_PI * (u - 0.5f));
    if (k >  max_kick) k =  max_kick;
    if (k < -max_kick) k = -max_kick;
    return k;
}

// N(0, sigma^2) via Box-Muller, hard-clamped to the same bound.
GHL_GATE_HD inline float gate_sample_gaussian(float sigma, float max_kick,
                                             unsigned int& state)
{
    const float u1 = fmaxf(gate_next_uniform(state), 1.0e-7f);
    const float u2 = gate_next_uniform(state);
    float k = sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    if (k >  max_kick) k =  max_kick;
    if (k < -max_kick) k = -max_kick;
    return k;
}

// One scatter draw, in the wall's own transverse frame: `d_axis` runs along the
// wall normal (into frame), `d_other` along the wall, `d_z` down the optical
// axis. The caller adds these to the specular direction.
//
//   GROOVED    — Cauchy across the marks, Gaussian along them, the pair rotated
//                by the groove angle. The heavy Cauchy tail is what sets reach;
//                the Gaussian keeps the streak thin.
//   CAUCHY_ISO — Cauchy on both transverse axes, no rotation applied (it would
//                be a no-op in distribution). NOT reachable from GROOVED by
//                setting groove_aniso = 1: that leaves the tight axis Gaussian.
//
// The axial term reuses half the wide draw to correlate forward tilt with the fan.
GHL_GATE_HD inline void gate_sample_lobe(const GpuGate& g, unsigned int& state,
                                        float& d_axis, float& d_other, float& d_z)
{
    if (g.lobe == (int)GateLobe::CAUCHY_ISO) {
        d_axis  = gate_sample_cauchy(g.sig_wide, g.max_kick, state);
        d_other = gate_sample_cauchy(g.sig_wide, g.max_kick, state);
        d_z     = gate_sample_cauchy(g.sig_wide, g.max_kick, state);
        return;
    }
    const float wide  = gate_sample_cauchy(g.sig_wide,  g.max_kick, state);
    const float tight = gate_sample_gaussian(g.sig_tight, g.max_kick, state);
    d_axis  = wide * g.cos_g - tight * g.sin_g;
    d_other = wide * g.sin_g + tight * g.cos_g;
    d_z     = 0.5f * wide;
}

// Schlick reflectance. cos_i is measured to the wall NORMAL, so a grazing ray
// has cos_i -> 0 and R -> 1 whatever r0 is. That is why a blackened gate still
// flares, and why milling the land matters more than painting it.
GHL_GATE_HD inline float gate_schlick(float cos_i, float r0)
{
    float c = 1.0f - fminf(fmaxf(cos_i, 0.0f), 1.0f);
    const float c2 = c * c;
    return r0 + (1.0f - r0) * c2 * c2 * c;
}

// Does a ray landing at `p_axis` with slope `s_axis` strike this wall?
//
// `edge` is the wall's signed plane coordinate and `sgn` is +1 for the wall on
// the positive side of the axis, -1 for the negative side, so the test is
// written once and used four times. On a hit, `back` is the distance from the
// wall hit point to the sensor plane (in the range [zs, zs+t]) and `fold` is
// the specular landing coordinate 2*edge - p_axis.
GHL_GATE_HD inline bool gate_wall_scrape(const GpuGate& g,
                                        float p_axis, float s_axis,
                                        float edge, float sgn,
                                        float& back, float& fold)
{
    const float slope_out = sgn * s_axis;         // outward component
    if (!(slope_out > 1.0e-12f)) return false;    // heading inward, or parallel
    const float over = sgn * (p_axis - edge);     // how far past the edge it lands
    if (!(over > 0.0f)) return false;             // lands inside the opening
    if (over >= (g.zs_mm + g.t_mm) * slope_out) return false;  // crossed ahead of the wall
    if (over <= g.zs_mm * slope_out)            return false;  // crossed behind it
    back = over / slope_out;                      // == |z_hit|, in [zs, zs+t]
    fold = 2.0f * edge - p_axis;
    return true;
}

// Host: resolve the runtime gate params from the render config and the render's
// resolved sensor extent. Unlike build_gpu_hurb this needs the sensor half-
// extents, because the opening DERIVES from the rendered frame — callers must
// pass the same eff_half_w/h they hand the ghost splat or the wall will not sit
// on the rendered frame edge.
inline GpuGate build_gpu_gate(const GateConfig& gc,
                              float sensor_half_w, float sensor_half_h)
{
    auto clampf = [](float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };

    GpuGate g;

    // A negative offset larger than the half-extent would invert the opening
    // and make every ray "outside" it, so each side is floored at 0.1 mm from
    // the axis. Positive offsets are unbounded — an oversized hard matte is a
    // legitimate thing to author.
    const float hw = (sensor_half_w > 0.0f) ? sensor_half_w : 1.0f;
    const float hh = (sensor_half_h > 0.0f) ? sensor_half_h : 1.0f;
    g.x_pos =  fmaxf(hw + gc.offset_right_mm,  0.1f);
    g.x_neg = -fmaxf(hw + gc.offset_left_mm,   0.1f);
    g.y_pos =  fmaxf(hh + gc.offset_top_mm,    0.1f);
    g.y_neg = -fmaxf(hh + gc.offset_bottom_mm, 0.1f);

    g.t_mm  = fmaxf(gc.thickness_mm, 1.0e-4f);
    g.zs_mm = fmaxf(gc.standoff_mm,  0.0f);

    g.sig_wide = clampf(gc.roughness_rad, 0.0f, 0.5f);
    const float aniso = clampf(gc.groove_aniso, 1.0e-3f, 1.0f);
    g.sig_tight = (gc.lobe == GateLobe::CAUCHY_ISO) ? g.sig_wide
                                                    : g.sig_wide * aniso;

    const float th = gc.groove_angle_deg * (float)(M_PI / 180.0);
    g.cos_g = std::cos(th);
    g.sin_g = std::sin(th);

    g.r0       = clampf(gc.reflectance_r0, 0.0f, 1.0f);
    g.max_kick = clampf(gc.max_kick_rad, 1.0e-4f, 1.5f);
    g.gain     = fmaxf(gc.gain, 0.0f);

    int ns = gc.scatter_samples;
    if (ns < 1)  ns = 1;
    if (ns > 64) ns = 64;
    g.n_scatter   = ns;
    g.inv_scatter = 1.0f / (float)ns;

    g.lobe = (int)gc.lobe;
    return g;
}
