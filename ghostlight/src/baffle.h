// ============================================================================
// baffle.h — Front-of-lens occluder stack.
//
// A GPU-side array of clipping planes ahead of the front element — matte-box
// flags, lens-hood rims, mount casings — that both clip entrance rays (the
// tracer + the starburst pupil) and provide the edges HURB bends rays around
// (baffle_stack_edge_distance). build_gpu_baffles() merges the render-config
// MatteBox (as one RECT baffle) plus any explicit DiffractionConfig::baffles.
//
// A pupil ray at spawn-plane position (px, py) heading (tan_ax, tan_ay) reaches
// a plane `dz` mm downstream at (px + tan_ax*dz, py + tan_ay*dz); it is blocked
// when that point falls outside the plane's shape.
//
// The device helpers carry the same __host__/__device__ guard as trace_core.h,
// so they compile in plain C++ (host tests) and in .cu kernels alike.
// ============================================================================
#pragma once

#include "render_config.h"   // Baffle, BaffleShape, MatteBox, DiffractionConfig
#include "spawn_plane.h"     // SPAWN_OFFSET (baffle dz is relative to that plane)
#include "vec3.h"            // Vec3f (baffle_stack_edge_distance normal, HURB path)

#if defined(__CUDACC__)
  #define GHL_BAFFLE_HD __host__ __device__
#else
  #define GHL_BAFFLE_HD
#endif

// Max baffles in the GPU stack (by-value kernel param; keep it modest).
constexpr int GHL_MAX_BAFFLES = 8;

// POD mirror of Baffle for the device. `shape` stores a BaffleShape int; `dz` is
// the plane offset from the SPAWN plane (mm), already resolved by the host.
struct GpuBaffle {
    int   shape  = 0;                 // (int)BaffleShape
    float dz     = 0.0f;              // plane offset from the spawn plane (mm)
    // RECT half-extents (mm); a side at MATTE_BOX_OPEN never clips.
    float top    = 1e30f, bottom = 1e30f, left = 1e30f, right = 1e30f;
    // CIRCLE: centre offset (mm), radius (mm), ellipse aspect.
    float cx     = 0.0f, cy = 0.0f, radius = 1e30f, aspect = 1.0f;
};

struct GpuBaffleStack {
    int       n = 0;
    GpuBaffle b[GHL_MAX_BAFFLES];
};

// True if a ray reaching plane-point (mx, my) is blocked by this baffle.
GHL_BAFFLE_HD inline bool baffle_blocks(const GpuBaffle& b, float mx, float my)
{
    if (b.shape == (int)BaffleShape::RECT)
        return (mx > b.right || mx < -b.left || my > b.top || my < -b.bottom);
    // CIRCLE / ellipse.
    float dx = (mx - b.cx) / b.radius;
    float dy = (my - b.cy) / (b.radius * b.aspect);
    return (dx * dx + dy * dy > 1.0f);
}

// True if a pupil ray at spawn-plane (px, py) heading (tan_ax, tan_ay) is blocked
// by ANY baffle in the stack. n == 0 -> a free zero-trip loop (never blocked).
GHL_BAFFLE_HD inline bool baffle_stack_blocks(const GpuBaffleStack& s,
                                             float px, float py,
                                             float tan_ax, float tan_ay)
{
    for (int k = 0; k < s.n; ++k) {
        const GpuBaffle& b = s.b[k];
        float mx = px + tan_ax * b.dz;
        float my = py + tan_ay * b.dz;
        if (baffle_blocks(b, mx, my)) return true;
    }
    return false;
}

// Perpendicular distance (mm) to the NEAREST baffle edge for a pupil ray at spawn
// (px, py) heading (tan_ax, tan_ay), evaluated at each baffle's plane, plus the
// transverse edge normal (nx, ny, 0). Only clearances INSIDE each shape count
// (the hard clip is baffle_stack_blocks upstream); the smallest across the whole
// stack wins. An empty stack, or a ray far from every edge, returns a large
// sentinel + zero normal, so HURB (hurb.h) applies no kick. HURB path only.
//   RECT   : min clearance to the four sides (an open side at MATTE_BOX_OPEN never
//            wins); normal points along that side's outward axis.
//   CIRCLE : first-order distance to the ellipse boundary
//            ((mx-cx)/radius)^2 + ((my-cy)/(radius*aspect))^2 = 1, normal along
//            the boundary gradient (matches aperture_edge_distance's ellipse form).
GHL_BAFFLE_HD inline float baffle_stack_edge_distance(const GpuBaffleStack& s,
                                                     float px, float py,
                                                     float tan_ax, float tan_ay,
                                                     Vec3f& edge_normal_out)
{
    float best = 1e30f;
    Vec3f best_n = Vec3f(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < s.n; ++k) {
        const GpuBaffle& b = s.b[k];
        const float mx = px + tan_ax * b.dz;
        const float my = py + tan_ay * b.dz;
        float d; Vec3f n;
        if (b.shape == (int)BaffleShape::RECT) {
            d = b.right - mx;                 n = Vec3f( 1.0f, 0.0f, 0.0f);  // +x side
            const float dl = mx + b.left;     // -x side
            const float dt = b.top - my;      // +y side
            const float db = my + b.bottom;   // -y side
            if (dl < d) { d = dl; n = Vec3f(-1.0f, 0.0f, 0.0f); }
            if (dt < d) { d = dt; n = Vec3f( 0.0f, 1.0f, 0.0f); }
            if (db < d) { d = db; n = Vec3f( 0.0f,-1.0f, 0.0f); }
        } else {
            const float rr = (b.radius > 1e-12f) ? b.radius : 1e-12f;
            const float ra = rr * ((b.aspect > 0.0f) ? b.aspect : 1.0f);
            const float ex = mx - b.cx;
            const float ey = my - b.cy;
            const float g  = sqrtf((ex/rr)*(ex/rr) + (ey/ra)*(ey/ra));
            const float gx = ex/(rr*rr);
            const float gy = ey/(ra*ra);
            const float gl = sqrtf(gx*gx + gy*gy);
            if (gl < 1e-20f) { d = rr; n = Vec3f(1.0f, 0.0f, 0.0f); }
            else             { d = (1.0f - g) / gl; n = Vec3f(gx/gl, gy/gl, 0.0f); }
        }
        if (d < best) { best = d; best_n = n; }
    }
    if (best < 0.0f) best = 0.0f;
    edge_normal_out = best_n;
    return best;
}

// Host: merge the render config's front-of-lens geometry into one GPU stack.
//   (1) the MatteBox -> one RECT baffle (dz = SPAWN_OFFSET - z_front_mm);
//   (2) explicit DiffractionConfig::baffles.
// SPAWN_OFFSET (spawn_plane.h) is the plane both the tracer and the starburst
// launch from; a baffle's dz is measured from there.
inline GpuBaffleStack build_gpu_baffles(const DiffractionConfig& dc)
{
    GpuBaffleStack s;

    if (dc.matte_box.enabled && s.n < GHL_MAX_BAFFLES) {
        const MatteBox& m = dc.matte_box;
        GpuBaffle& b = s.b[s.n++];
        b.shape  = (int)BaffleShape::RECT;
        b.dz     = SPAWN_OFFSET - m.z_front_mm;
        b.top    = m.top;   b.bottom = m.bottom;
        b.left   = m.left;  b.right  = m.right;
    }

    for (const Baffle& cb : dc.baffles) {
        if (s.n >= GHL_MAX_BAFFLES) break;
        GpuBaffle& b = s.b[s.n++];
        b.shape  = (int)cb.shape;
        b.dz     = SPAWN_OFFSET - cb.z_front_mm;
        b.top    = cb.top;  b.bottom = cb.bottom;
        b.left   = cb.left; b.right  = cb.right;
        b.cx     = cb.cx;   b.cy     = cb.cy;
        b.radius = cb.radius; b.aspect = (cb.aspect > 0.0f) ? cb.aspect : 1.0f;
    }

    return s;
}
