// ============================================================================
// spawn_probe.cpp — Probing the disc the renderers actually spawn.
// ============================================================================

#include "spawn_probe.h"

#include "spawn_plane.h"        // SPAWN_OFFSET, spawn_shift

#include <algorithm>
#include <cmath>
#include <vector>

// Default probe resolution and boundary supersampling balance throughput-area
// accuracy against trace cost.
static const int PROBE_GRID    = 17;
static const int PROBE_EDGE_SS = 2;

// Below this entrance-pupil / front-element ratio the grid is restricted to the
// window that passes light rather than spread over the whole front disc.
static const float DEEP_STOP_FILL = 0.45f;

// How much bigger than the pupil the sampling window starts. Big enough that
// the pupil's walk between two probe angles stays inside it — a window that
// trails the walk clips the beam, under-reads the throughput and drags the
// resolved crossing inward — and small enough to retain probe resolution.
static const float WINDOW_DILATION = 5.0f;

// ---------------------------------------------------------------------------
bool trace_from_ok(const OpticalSystem& lens, float hx, float hy,
                   float angle_h, float angle_v, float lambda_nm,
                   TraceResult* result_out)
{
    Ray ray;
    ray.origin = Vec3f(hx, hy, lens.surfaces[0].z - SPAWN_OFFSET);
    ray.lambda = lambda_nm;
    ray.dir    = Vec3f(std::tan(angle_h), std::tan(angle_v), 1.0f).normalized();

    TraceResult res = trace_primary_ray(ray, lens);
    if (result_out) *result_out = res;
    return res.status == TraceStatus::OK;
}

// ---------------------------------------------------------------------------
Window initial_probe_window(float pupil_fill)
{
    Window w{0.0f, 0.0f, 1.0f};
    if (pupil_fill > 0.0f && pupil_fill < DEEP_STOP_FILL)
        w.r = std::min(1.0f, pupil_fill * WINDOW_DILATION);
    return w;
}

// ---------------------------------------------------------------------------
DiscArea probe_window(const OpticalSystem& lens, const PupilMask& mask,
                      float front_R, float angle_h, float angle_v,
                      float lambda_nm, const Window& win, int grid, int edge_ss)
{
    const int   n    = (grid > 0) ? grid : PROBE_GRID;
    const int   ess  = (edge_ss > 0) ? edge_ss : PROBE_EDGE_SS;
    const float step = 2.0f * win.r / (float)n;   // cell size in disc units
    const float bx   = std::tan(angle_h);
    const float by   = std::tan(angle_v);

    float sdx = 0.0f, sdy = 0.0f;
    spawn_shift(bx, by, sdx, sdy);   // the disc follows the beam

    // Pass 1: cell centres. `alive` doubles as the boundary test below.
    std::vector<unsigned char> alive((size_t)n * n, 0);
    std::vector<float>         land_x((size_t)n * n, 0.0f);
    std::vector<float>         land_y((size_t)n * n, 0.0f);

    auto sample = [&](float u, float v, float* out_x, float* out_y) -> bool {
        if (!mask.contains(u, v)) return false;
        TraceResult res;
        if (!trace_from_ok(lens, u * front_R + sdx, v * front_R + sdy,
                           angle_h, angle_v, lambda_nm, &res))
            return false;
        if (!std::isfinite(res.position.x) || !std::isfinite(res.position.y))
            return false;
        *out_x = res.position.x;
        *out_y = res.position.y;
        return true;
    };

    const float u_org = win.cu - win.r;   // lower corner of the sampled square
    const float v_org = win.cv - win.r;

    for (int j = 0; j < n; ++j) {
        const float v = v_org + ((float)j + 0.5f) * step;
        for (int i = 0; i < n; ++i) {
            const float u   = u_org + ((float)i + 0.5f) * step;
            const size_t k  = (size_t)j * n + i;
            float x = 0.0f, y = 0.0f;
            if (sample(u, v, &x, &y)) {
                alive[k]  = 1;
                land_x[k] = x;
                land_y[k] = y;
            }
        }
    }

    // Pass 2: a cell whose 4-neighbourhood disagrees about survival straddles a
    // boundary (the mask rim or the vignetting edge) and is resampled for
    // fractional coverage; interior cells take their whole weight.
    DiscArea out;
    const float cell_area = step * step;      // absolute, so windows compare
    const float sub_w     = 1.0f / (float)(ess * ess);
    float lo_u = 1e30f, hi_u = -1e30f, lo_v = 1e30f, hi_v = -1e30f;

    auto note = [&](float u, float v, float w, float x, float y) {
        out.area   += w * cell_area;
        out.mean_x += w * cell_area * x;
        out.mean_y += w * cell_area * y;
        lo_u = std::min(lo_u, u); hi_u = std::max(hi_u, u);
        lo_v = std::min(lo_v, v); hi_v = std::max(hi_v, v);
    };

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const size_t k    = (size_t)j * n + i;
            const bool   here = alive[k] != 0;
            const float  uc   = u_org + ((float)i + 0.5f) * step;
            const float  vc   = v_org + ((float)j + 0.5f) * step;

            // A survivor on the sampled rim means the beam runs past the
            // window. Tested directly rather than as part of the neighbour
            // scan below: that scan stops at the first disagreement it finds,
            // so an in-grid dead neighbour would mask the rim on three sides
            // out of four and the caller would never widen.
            const bool on_rim = (i == 0 || i == n - 1 || j == 0 || j == n - 1);
            if (here && on_rim) out.at_edge = true;

            bool edge = here && on_rim;   // off-grid neighbours count as dead
            const int di[4] = {1, -1, 0, 0};
            const int dj[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4 && !edge; ++d) {
                const int ni = i + di[d], nj = j + dj[d];
                if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                if ((alive[(size_t)nj * n + ni] != 0) != here) edge = true;
            }

            if (!edge) {
                if (here) note(uc, vc, 1.0f, land_x[k], land_y[k]);
                continue;
            }

            const float u0  = u_org + (float)i * step;
            const float v0  = v_org + (float)j * step;
            const float sub = step / (float)ess;
            for (int sj = 0; sj < ess; ++sj) {
                const float v = v0 + ((float)sj + 0.5f) * sub;
                for (int si = 0; si < ess; ++si) {
                    const float u = u0 + ((float)si + 0.5f) * sub;
                    float x = 0.0f, y = 0.0f;
                    if (!sample(u, v, &x, &y)) continue;
                    note(u, v, sub_w, x, y);
                }
            }
        }
    }

    if (out.area > 0.0f) {
        out.mean_x /= out.area;
        out.mean_y /= out.area;
        // Survivor bounding box, padded by a cell, as the next angle's seed.
        out.fit.cu = 0.5f * (lo_u + hi_u);
        out.fit.cv = 0.5f * (lo_v + hi_v);
        out.fit.r  = std::max(std::max(hi_u - out.fit.cu, hi_v - out.fit.cv), step) + step;
    }
    return out;
}

// ---------------------------------------------------------------------------
DiscArea probe_disc(const OpticalSystem& lens, const PupilMask& mask,
                    float front_R, float angle_h, float angle_v,
                    float lambda_nm, Window* track, int grid, int edge_ss)
{
    Window win = *track;
    if (win.r >= 1.0f) { win.cu = 0.0f; win.cv = 0.0f; win.r = 1.0f; }

    DiscArea best;
    for (int attempt = 0; attempt < 6; ++attempt) {
        const DiscArea d = probe_window(lens, mask, front_R,
                                        angle_h, angle_v, lambda_nm, win, grid, edge_ss);
        if (d.area > 0.0f) best = d;
        if (d.area > 0.0f && !d.at_edge) {
            // Clean fit: re-centre on the survivors, keep the size. The pupil
            // walks smoothly, so its own box is the best predictor of where it
            // will be one probe step later.
            *track    = d.fit;
            track->r  = win.r;
            return d;
        }
        if (win.r >= 1.0f) break;                 // the whole disc already

        // Re-centre before widening. The pupil's area barely changes with field
        // angle — it is its POSITION that moves — so a beam on the rim usually
        // means the window is trailing the walk, not that it is too small. Slide
        // onto whatever was found (hill-climbing toward the beam) and keep the
        // resolution; widening is the last resort, because on a deep-stop lens
        // a wider window is a coarser one and the answer degrades with it.
        if (d.area > 0.0f && attempt < 3) {
            win.cu = d.fit.cu;
            win.cv = d.fit.cv;
            continue;
        }
        win.r  = std::min(1.0f, win.r * 2.5f);
        if (win.r >= 1.0f) { win.cu = 0.0f; win.cv = 0.0f; }
    }
    track->cu = win.cu; track->cv = win.cv; track->r = win.r;
    return best;
}
