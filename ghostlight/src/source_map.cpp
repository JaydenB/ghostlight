// ============================================================================
// source_map.cpp — Screen position <-> field angle, solved by tracing.
// See source_map.h for why this exists rather than a closed form.
// ============================================================================

#include "source_map.h"

#include "aperture_sampler.h"   // PupilMask, resolve_pupil_mask
#include "newton_aim.h"         // newton_aim (shared with PSFRenderer)
#include "spawn_probe.h"        // Window, DiscArea, probe_disc, initial_probe_window

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// The probe wavelength. The d-line, because that is what calibrate_lens
// resolves sensor_half at: the solve inverts that measurement, so it has to be
// the same measurement.
static constexpr float kLambdaNm = 587.56f;

// Match calibration's probe grid so the solve inverts the same measurement.
static constexpr int kSolveGrid = 17;

// Newton settings.
//
// The finite-difference step must exceed survivor-grid quantization while
// remaining local relative to field curvature.
static constexpr int   kMaxIter = 8;
static constexpr float kFdStep  = 5e-3f;   // rad
static constexpr float kTolMm   = 1e-3f;   // 1 um — what Newton aims for

// What counts as solved, as a fraction of the frame's smaller half-extent.
//
// Acceptance tolerance as a fraction of frame half-extent. This accommodates
// the survivor grid's quantized landing measurement.
static constexpr float kAcceptFrac = 0.01f;

// Continuation: how many bisection steps look for the last convergent target
// (in mm along the ray), and how far the arctangent extension is allowed to
// reach. tan() is clamped at 1.55 rad (88.8 deg) so an extreme extrapolation
// saturates instead of blowing up through the pole.
static constexpr int   kBisectSteps  = 6;
static constexpr float kBisectTolMm  = 0.02f;
static constexpr float kMaxExtendRad = 1.55f;

// Match calibration's boundary-cell subsampling because this solve inverts its
// measured survivor mean.
static const int kSolveEdgeSS = 2;

// Angular step of the warm-up walk that carries the sampling window out from
// the axial pupil to the seed angle. Stepped by ANGLE rather than into a fixed
// number of pieces: a fixed count is a step size that grows with the field, so
// the far-off-axis case the walk exists for is the one it would serve worst.
static const float kWarmStepRad = 0.10f;   // ~5.7 deg

namespace {

// Landing function with a sampling window that follows the pupil. The window
// advances only after a successful probe.
struct SolveProbe {
    const OpticalSystem& lens;
    PupilMask            mask;
    float                front_R;
    Window               win;
    int                  evals = 0;   // probes actually traced
    int                  calls = 0;   // probes asked for

    // A handful of recent evaluations, so a repeated one is free.
    //
    // Worth having because the search asks for the same angle more than once by
    // construction: newton_aim evaluates its final iterate, then evaluates it
    // again to fold into the best-iterate record, and the Jacobian measurement
    // that follows evaluates it a third time. On an 8-evaluation solve that is
    // two evaluations of pure repetition — a quarter of the cost.
    //
    // The key includes the WINDOW, not just the angle, because the window is an
    // input to the probe: the same angle sampled through a differently-placed
    // window is a different measurement. Storing the resulting window too means
    // a hit reproduces the state transition as well as the value, so caching
    // cannot change any answer.
    struct Entry { float ax, ay, cu, cv, r, x, y; Window out; bool ok; };
    static constexpr int kCache = 8;
    Entry cache[kCache]{};
    int   n_cached = 0, next_slot = 0;

    bool at(float ax, float ay, float& x, float& y)
    {
        ++calls;
        for (int i = 0; i < n_cached; ++i) {
            const Entry& e = cache[i];
            if (e.ax == ax && e.ay == ay
                && e.cu == win.cu && e.cv == win.cv && e.r == win.r) {
                if (!e.ok) return false;
                win = e.out; x = e.x; y = e.y;
                return true;
            }
        }

        ++evals;
        const Window in = win;
        Window w = in;                        // probe_disc mutates its tracker
        const DiscArea d = probe_disc(lens, mask, front_R, ax, ay,
                                      kLambdaNm, &w, kSolveGrid, kSolveEdgeSS);
        const bool ok = (d.area > 0.0f)
                        && std::isfinite(d.mean_x) && std::isfinite(d.mean_y);

        Entry& e = cache[next_slot];
        e.ax = ax; e.ay = ay;
        e.cu = in.cu; e.cv = in.cv; e.r = in.r;
        e.x = d.mean_x; e.y = d.mean_y; e.out = w; e.ok = ok;
        next_slot = (next_slot + 1) % kCache;
        if (n_cached < kCache) ++n_cached;

        if (!ok) return false;
        win = w;
        x = d.mean_x;
        y = d.mean_y;
        return true;
    }
};

// d(landing)/d(angle) by one-sided differences, plus the landing itself.
// J is row-major [dx/dax, dx/day, dy/dax, dy/day].
//
// Use a backward difference when the forward probe has no landing. This is
// required at a continuation anchor, which is the last valid field angle.
bool measure_jacobian(SolveProbe& P, float ax, float ay,
                      float* out_land, float* out_J)
{
    float cx, cy;
    if (!P.at(ax, ay, cx, cy)) return false;

    // Partial derivative along one axis, from whichever side has light.
    auto slope = [&](float dax, float day, float* dx, float* dy) -> bool {
        float px, py;
        if (P.at(ax + dax, ay + day, px, py)) {
            *dx = (px - cx) / kFdStep;
            *dy = (py - cy) / kFdStep;
            return true;
        }
        if (P.at(ax - dax, ay - day, px, py)) {
            *dx = (cx - px) / kFdStep;
            *dy = (cy - py) / kFdStep;
            return true;
        }
        return false;
    };

    float dxdax, dydax, dxday, dyday;
    if (!slope(kFdStep, 0.0f, &dxdax, &dydax)) return false;
    if (!slope(0.0f, kFdStep, &dxday, &dyday)) return false;

    out_land[0] = cx;
    out_land[1] = cy;
    out_J[0] = dxdax;  out_J[1] = dxday;
    out_J[2] = dydax;  out_J[3] = dyday;
    return true;
}

// One-entry memo shared by the chunks of an extended-source render. Hash the
// complete Surface representation so new fields participate in invalidation.
// Padding can cause a missed cache hit, but cannot equate different byte strings.
uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ULL)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

std::mutex     g_memo_mutex;
bool           g_memo_valid = false;
uint64_t       g_memo_key   = 0;
SourceMapSolve g_memo_val;
bool           g_warned_fallback = false;

} // namespace

// ---------------------------------------------------------------------------
SourceMapSolve solve_source_angle(const OpticalSystem&   lens,
                                  const LensCalibration& calib,
                                  float ndc_x, float ndc_y,
                                  float eff_half_w, float eff_half_h)
{
    uint64_t key = 0;
    if (!lens.surfaces.empty())
        key = fnv1a(lens.surfaces.data(), lens.surfaces.size() * sizeof(Surface));
    key = fnv1a(&calib, sizeof(LensCalibration), key);
    {
        const float req[4] = {ndc_x, ndc_y, eff_half_w, eff_half_h};
        key = fnv1a(req, sizeof(req), key);
    }
    {
        std::lock_guard<std::mutex> lk(g_memo_mutex);
        if (g_memo_valid && g_memo_key == key) return g_memo_val;
    }

    SourceMapSolve sm;
    sm.eff_half_w = eff_half_w;
    sm.eff_half_h = eff_half_h;

    // Constant-focal-length seed and fallback.
    const float scale_w = (calib.sensor_half_w > 0.0f) ? eff_half_w / calib.sensor_half_w : 1.0f;
    const float scale_h = (calib.sensor_half_h > 0.0f) ? eff_half_h / calib.sensor_half_h : 1.0f;
    const float seed_ax = std::atan(ndc_x * scale_w * std::tan(calib.max_half_angle_h));
    const float seed_ay = std::atan(ndc_y * scale_h * std::tan(calib.max_half_angle_v));
    sm.inv_tan_w = 1.0f / (scale_w * std::tan(calib.max_half_angle_h));
    sm.inv_tan_h = 1.0f / (scale_h * std::tan(calib.max_half_angle_v));
    sm.angle_x   = seed_ax;
    sm.angle_y   = seed_ay;
    sm.status    = SourceMapStatus::FALLBACK;

    // Set once the probe exists, so a give_up still reports the work it did.
    const int* P_evals_for_report = nullptr;

    auto give_up = [&](const char* why) -> SourceMapSolve {
        {
            std::lock_guard<std::mutex> lk(g_memo_mutex);
            if (!g_warned_fallback) {
                g_warned_fallback = true;
                std::fprintf(stderr,
                             "[ghostlight] warning: source map could not be solved by "
                             "tracing (%s); falling back to the assumed-focal-length "
                             "form, which is wrong on distorting lenses. Further "
                             "occurrences are not reported.\n", why);
            }
            sm.probe_evals = P_evals_for_report ? *P_evals_for_report : 0;
            g_memo_valid = true; g_memo_key = key; g_memo_val = sm;
        }
        return sm;
    };

    if (lens.surfaces.empty()) return give_up("lens has no surfaces");

    // What the map is being asked for, in millimetres on the sensor plane. The
    // target is SIGNED: an inverting lens is handled by the solve landing on a
    // negative angle, not by a magnitude convention that would silently fold
    // the two quadrants together.
    const float target_x = ndc_x * eff_half_w;
    const float target_y = ndc_y * eff_half_h;

    const float front_R = lens.surfaces[0].semi_aperture;
    const float fill_x  = (front_R > 0.0f) ? calib.entrance_pupil_semi_x / front_R : 1.0f;
    const float fill_y  = (front_R > 0.0f) ? calib.entrance_pupil_semi_y / front_R : 1.0f;

    SolveProbe P{lens,
                 resolve_pupil_mask(lens, ApertureSamplerParams{}),
                 front_R,
                 initial_probe_window(std::min(fill_x, fill_y))};
    P_evals_for_report = &P.evals;

    auto landing = [&](float ax, float ay, float& x, float& y) -> bool {
        return P.at(ax, ay, x, y);
    };

    // Reset to the axial pupil and walk outward in fixed angular steps. Resetting
    // makes every solve attempt independent; fixed steps retain resolution at
    // large field angles.
    const Window win0 = initial_probe_window(std::min(fill_x, fill_y));
    auto reset_and_walk = [&](float ax, float ay) {
        P.win = win0;
        const float mag = std::sqrt(ax * ax + ay * ay);
        if (!(P.win.r < 1.0f) || !(mag > 0.0f)) return;
        const int steps = std::max(3, std::min(16,
                              (int)std::ceil(mag / kWarmStepRad)));
        float tx, ty;
        for (int k = 1; k <= steps; ++k) {
            const float f = (float)k / (float)steps;
            P.at(ax * f, ay * f, tx, ty);
        }
    };

    const float seed_mag = std::sqrt(seed_ax * seed_ax + seed_ay * seed_ay);

    // Newton aims for kTolMm and reports the best iterate it reached whether or
    // not it got there; this is the bar that iterate has to clear to count.
    const float accept_mm =
        std::max(kTolMm, kAcceptFrac * std::min(std::fabs(eff_half_w),
                                                std::fabs(eff_half_h)));

    // Fill in a SOLVED result at (ax, ay), measuring its Jacobian there.
    auto commit_solved = [&](float ax, float ay, float res) -> bool {
        float land[2], J[4];
        if (!measure_jacobian(P, ax, ay, land, J)) return false;
        sm.angle_x     = ax;
        sm.angle_y     = ay;
        sm.status      = SourceMapStatus::SOLVED;
        sm.residual_mm = res;
        sm.anchor_ax   = ax;      sm.anchor_ay = ay;
        sm.anchor_x    = land[0]; sm.anchor_y  = land[1];
        for (int i = 0; i < 4; ++i) sm.J[i] = J[i];
        sm.probe_evals = P.evals;
        return true;
    };

    // Keep the best iterate when probe quantization prevents meeting the
    // acceptance threshold for an otherwise reachable target.
    float best_ax = seed_ax, best_ay = seed_ay, best_res = 1e30f;

    // Run one attempt at the real target; true means it met the bar.
    auto attempt = [&](float from_ax, float from_ay, float cap) -> bool {
        reset_and_walk(from_ax, from_ay);
        float ax = from_ax, ay = from_ay, res = 1e30f;
        newton_aim(landing, target_x, target_y, ax, ay, res,
                   kMaxIter, kFdStep, cap, kTolMm);
        if (res < best_res) { best_res = res; best_ax = ax; best_ay = ay; }
        if (!(res < accept_mm)) return false;
        return commit_solved(ax, ay, res);
    };

    auto publish = [&]() -> SourceMapSolve {
        std::lock_guard<std::mutex> lk(g_memo_mutex);
        g_memo_valid = true; g_memo_key = key; g_memo_val = sm;
        return sm;
    };

    // First attempt: allow a large correction from a poorly scaled closed-form seed.
    if (attempt(seed_ax, seed_ay, std::max(0.02f, 0.5f * seed_mag)))
        return publish();

    // Second attempt: cap corrections when discrete survivor rows make the
    // finite-difference Jacobian locally noisy.
    if (attempt(seed_ax, seed_ay, std::max(0.005f, 0.08f * seed_mag)))
        return publish();

    // --- Past the last landing the lens delivers: continue instead. ---
    //
    // Find the furthest point along the ray to the target that DOES solve, and
    // join an arctangent to it.
    //
    // Search absolute radius within the lens's image circle. The anchor must
    // depend only on lens and direction so the continuation remains monotonic.
    float ax_lo = 0.0f, ay_lo = 0.0f;
    bool  have_anchor = false;
    float anchor_radius = 0.0f;   // how far out the lens was still measurable
    // Preserve the sampling window at the accepted anchor; its tracked path is
    // part of the probe state near the illuminated boundary.
    Window anchor_win;

    const float tmag = std::sqrt(target_x * target_x + target_y * target_y);
    if (tmag > 0.0f) {
        const float ux = target_x / tmag, uy = target_y / tmag;
        const float rc = std::sqrt(ux * ux * calib.image_circle_semi_w * calib.image_circle_semi_w
                                 + uy * uy * calib.image_circle_semi_h * calib.image_circle_semi_h);
        // Keep the bracket near the image circle; probing too far into darkness
        // widens the tracked window and can admit unrelated survivor slivers.
        float r_lo = 0.0f;
        float r_hi = 1.05f * rc;

        // Bracket, Newton cap, and probe window must depend on lens and direction,
        // not request radius, so collinear requests share one continuation anchor.
        const float anchor_seed_ax = std::atan(0.98f * rc * ux * std::tan(calib.max_half_angle_h)
                                               / (calib.sensor_half_w > 0.0f ? calib.sensor_half_w : 1.0f));
        const float anchor_seed_ay = std::atan(0.98f * rc * uy * std::tan(calib.max_half_angle_v)
                                               / (calib.sensor_half_h > 0.0f ? calib.sensor_half_h : 1.0f));
        const float anchor_mag = std::sqrt(anchor_seed_ax * anchor_seed_ax
                                         + anchor_seed_ay * anchor_seed_ay);
        const float anchor_cap = std::max(0.02f, 0.5f * anchor_mag);

        reset_and_walk(anchor_seed_ax, anchor_seed_ay);

        auto try_target = [&](float tx, float ty, float& io_ax, float& io_ay) -> bool {
            float tax = io_ax, tay = io_ay, r = 1e30f;
            newton_aim(landing, tx, ty, tax, tay, r,
                       kMaxIter, kFdStep, anchor_cap, kTolMm);
            if (!(r < accept_mm)) return false;
            io_ax = tax; io_ay = tay;
            return true;
        };


        // Start just inside the image circle, then refine the bracket.
        {
            const float g = 0.98f * rc;
            float a = ax_lo, b = ay_lo;
            if (g > 0.0f && try_target(g * ux, g * uy, a, b)) {
                r_lo = g; ax_lo = a; ay_lo = b; have_anchor = true;
                anchor_radius = g; anchor_win = P.win;
            } else {
                r_hi = g;
            }
        }
        for (int i = 0; i < kBisectSteps && (r_hi - r_lo) > kBisectTolMm; ++i) {
            const float mid = 0.5f * (r_lo + r_hi);
            float a = ax_lo, b = ay_lo;
            if (try_target(mid * ux, mid * uy, a, b)) {
                r_lo = mid; ax_lo = a; ay_lo = b; have_anchor = true;
                anchor_radius = mid; anchor_win = P.win;
            } else {
                r_hi = mid;
            }
        }
    }
    if (!have_anchor && !(seed_mag > 0.0f))
        return give_up("nothing solves anywhere along the ray");

    // Third attempt: restart from the converged anchor with a reset window.
    if (have_anchor && attempt(ax_lo, ay_lo, std::max(0.02f, 0.5f * seed_mag)))
        return publish();

    // A target within the measured anchor radius is reachable; retain the best
    // quantized iterate rather than applying out-of-range continuation.
    if (have_anchor && anchor_radius >= tmag && best_res < 1e29f) {
        reset_and_walk(best_ax, best_ay);
        if (commit_solved(best_ax, best_ay, best_res))
            return publish();
    }

    // Restore the accepted anchor's tracked window before measuring its Jacobian.
    if (have_anchor) P.win = anchor_win;

    float land[2], J[4];
    if (!measure_jacobian(P, ax_lo, ay_lo, land, J))
        return give_up("no convergent anchor to continue from");

    const float det = J[0] * J[3] - J[1] * J[2];
    if (!(std::fabs(det) > 1e-12f))
        return give_up("landing is singular in angle at the anchor");

    // theta(T) = theta_b + u_hat * atan(|u|),  u = J^-1 (T - T_b).
    // Value and slope both match the solved map at the handover (atan(x) -> x
    // as x -> 0), it is monotonic in |u|, and it is bounded: however far the
    // source is dragged, the angle approaches anchor + 90 deg rather than
    // running away.
    const float dx = target_x - land[0], dy = target_y - land[1];
    const float ux = ( J[3] * dx - J[1] * dy) / det;
    const float uy = (-J[2] * dx + J[0] * dy) / det;
    const float um = std::sqrt(ux * ux + uy * uy);

    if (um > 1e-12f) {
        const float th = std::atan(um);
        sm.angle_x = ax_lo + (ux / um) * th;
        sm.angle_y = ay_lo + (uy / um) * th;
    } else {
        sm.angle_x = ax_lo;
        sm.angle_y = ay_lo;
    }
    sm.status      = SourceMapStatus::CONTINUED;
    // For continuation, residual_mm is the distance past the measured anchor.
    sm.residual_mm = std::sqrt(dx * dx + dy * dy);
    sm.anchor_ax   = ax_lo;   sm.anchor_ay = ay_lo;
    sm.anchor_x    = land[0]; sm.anchor_y  = land[1];
    for (int i = 0; i < 4; ++i) sm.J[i] = J[i];
    sm.probe_evals = P.evals;

    {
        std::lock_guard<std::mutex> lk(g_memo_mutex);
        g_memo_valid = true; g_memo_key = key; g_memo_val = sm;
    }
    return sm;
}

// ---------------------------------------------------------------------------
void source_map_screen(const SourceMapSolve& base,
                       float angle_x, float angle_y,
                       float* out_sx, float* out_sy)
{
    if (base.status == SourceMapStatus::FALLBACK) {
        // Constant-focal-length forward map.
        *out_sx = 0.5f + 0.5f * std::tan(angle_x) * base.inv_tan_w;
        *out_sy = 0.5f + 0.5f * std::tan(angle_y) * base.inv_tan_h;
        return;
    }

    const float dax = angle_x - base.anchor_ax;
    const float day = angle_y - base.anchor_ay;
    const float d   = std::sqrt(dax * dax + day * day);

    float ux = dax, uy = day;
    if (d > 1e-12f) {
        const float t = std::tan(std::min(d, kMaxExtendRad));
        ux = (dax / d) * t;
        uy = (day / d) * t;
    }
    const float x = base.anchor_x + base.J[0] * ux + base.J[1] * uy;
    const float y = base.anchor_y + base.J[2] * ux + base.J[3] * uy;

    *out_sx = (base.eff_half_w > 0.0f) ? 0.5f + 0.5f * x / base.eff_half_w : 0.5f;
    *out_sy = (base.eff_half_h > 0.0f) ? 0.5f + 0.5f * y / base.eff_half_h : 0.5f;
}

// ---------------------------------------------------------------------------
bool source_map_landing(const OpticalSystem&   lens,
                        const LensCalibration& calib,
                        float angle_x, float angle_y,
                        float* out_x_mm, float* out_y_mm)
{
    if (lens.surfaces.empty()) return false;

    const float front_R = lens.surfaces[0].semi_aperture;
    const float fill_x  = (front_R > 0.0f) ? calib.entrance_pupil_semi_x / front_R : 1.0f;
    const float fill_y  = (front_R > 0.0f) ? calib.entrance_pupil_semi_y / front_R : 1.0f;

    const PupilMask mask = resolve_pupil_mask(lens, ApertureSamplerParams{});
    Window win = initial_probe_window(std::min(fill_x, fill_y));

    // The calibrated grid, not the solve's cheaper one. This is the reference a
    // test checks the solve against, so it must not inherit the solve's own
    // numerical shortcut — agreeing with itself would prove nothing.
    //
    // Walk out to the angle rather than jumping, for the same reason the solve
    // does: on a deep-stop lens the window has to track the pupil to find it.
    const float mag = std::sqrt(angle_x * angle_x + angle_y * angle_y);
    if (win.r < 1.0f && mag > 0.0f) {
        for (int k = 1; k <= 3; ++k) {
            const float f = (float)k / 4.0f;
            probe_disc(lens, mask, front_R, angle_x * f, angle_y * f, kLambdaNm, &win);
        }
    }

    const DiscArea d = probe_disc(lens, mask, front_R, angle_x, angle_y, kLambdaNm, &win);
    if (!(d.area > 0.0f) || !std::isfinite(d.mean_x) || !std::isfinite(d.mean_y))
        return false;
    *out_x_mm = d.mean_x;
    *out_y_mm = d.mean_y;
    return true;
}
