// ============================================================================
// spawn_plane.h — Where the renderers launch entrance rays.
//
// Every renderer spawns its pupil rays on a plane SPAWN_OFFSET mm ahead of the
// front vertex (surfaces[0].z - SPAWN_OFFSET) and samples a disc of
// surfaces[0].semi_aperture across it. Baffle planes are positioned relative to
// the same plane (baffle.h resolves dz = SPAWN_OFFSET - z_front_mm), so the two
// must agree on the distance.
//
// Off axis the disc has to follow the beam. A ray launched at height h and
// heading (bx, by) arrives at the front vertex at h + SPAWN_OFFSET*b, so an
// axis-centred disc samples a set that has slid off the front element by
// SPAWN_OFFSET*b — on a lens whose front element sits close to the marginal ray
// that discards most of the beam at 20 deg. spawn_shift() gives the correction
// that lands the sampled disc back on the element: it IS the front aperture's
// back-projection onto the spawn plane, so it captures everything a planar disc
// can (the small residual is surface sag, which no plane can follow).
//
// The tracer itself has no opinion about where rays start; this is a renderer
// convention, which is why it lives here rather than in trace_core.h.
// ============================================================================
#pragma once

#if defined(__CUDACC__)
  #define GHL_SPAWN_HD __host__ __device__
#else
  #define GHL_SPAWN_HD
#endif

// Spawn-plane distance ahead of the front vertex (mm).
constexpr float SPAWN_OFFSET = 20.0f;

// Lateral offset to add to a spawn position so a ray heading (bx, by) — the
// direction's x/z and y/z ratios — crosses the front vertex plane where an
// unshifted launch of the same height would have crossed on axis.
//
// On axis bx and by are exactly zero, so the launch shift is zero.
GHL_SPAWN_HD inline void spawn_shift(float bx, float by, float& out_dx, float& out_dy)
{
    out_dx = -SPAWN_OFFSET * bx;
    out_dy = -SPAWN_OFFSET * by;
}
