// ============================================================================
// newton_aim.h — 2x2 finite-difference Newton on a ray-landing function.
//
// Shared by two callers with different tolerances:
//   * aim_chief_ray()       (PSFRenderer.cpp) — clean chief ray at 0.1 um, then
//                           a survivor-mean fallback at 5 um.
//   * solve_source_angle()  (source_map.cpp)  — spawn-disc survivor mean at
//                           1 um, with a coarser finite-difference step because
//                           the area-weighted mean is coverage-quantised.
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>

// 2×2 finite-difference Newton on a landing function `landing(ax, ay) -> (x, y)`
// (returns false when the probe vignettes).  Refines (ax, ay) so the landing
// meets `target`. Reverts to the best iterate and reports its residual; returns
// true if that residual is < tol. Steps are clamped and singular Jacobians abort.
template <typename LandingFn>
static bool newton_aim(LandingFn&& landing,
                       float target_x, float target_y,
                       float& ax, float& ay, float& out_residual,
                       int max_iter, float fd_step, float step_cap, float tol)
{
    float best_ax = ax, best_ay = ay, best_res = 1e30f;
    auto record = [&](float x, float y) {
        const float rx = x - target_x, ry = y - target_y;
        const float res = std::sqrt(rx*rx + ry*ry);
        if (res < best_res) { best_res = res; best_ax = ax; best_ay = ay; }
        return res;
    };

    float cx, cy;
    if (!landing(ax, ay, cx, cy)) { out_residual = 1e30f; return false; }

    for (int it = 0; it < max_iter; ++it)
    {
        const float rx = cx - target_x, ry = cy - target_y;
        if (record(cx, cy) < tol) break;

        float xx, xy, yx, yy;
        if (!landing(ax + fd_step, ay, xx, xy)) break;
        if (!landing(ax, ay + fd_step, yx, yy)) break;
        // Columns of J: d(landing)/d(ax) and d(landing)/d(ay).
        const float Jxx = (xx - cx) / fd_step, Jyx = (xy - cy) / fd_step;
        const float Jxy = (yx - cx) / fd_step, Jyy = (yy - cy) / fd_step;
        const float det = Jxx * Jyy - Jxy * Jyx;
        if (std::fabs(det) < 1e-12f) break;

        float dax = (-rx * Jyy + ry * Jxy) / det;
        float day = ( rx * Jyx - ry * Jxx) / det;
        dax = std::max(-step_cap, std::min(step_cap, dax));
        day = std::max(-step_cap, std::min(step_cap, day));
        ax += dax; ay += day;
        if (!landing(ax, ay, cx, cy)) break;
    }
    // Fold in the final landing, then revert to the best iterate.
    if (landing(ax, ay, cx, cy)) record(cx, cy);
    ax = best_ax; ay = best_ay;
    out_residual = best_res;
    return best_res < tol;
}
