// ============================================================================
// lens_calibration.cpp — Ray-traced covered field of view and first-order optics
// ============================================================================

#include "lens_calibration.h"
#include "aperture_sampler.h"   // PupilMask, resolve_pupil_mask
#include "spawn_probe.h"        // trace_from_ok, Window, DiscArea, probe_disc/window
#include "trace.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Paraxial effective focal length on one axis, from the image-height / field-
// angle relation y' = f * tan(theta) evaluated at a small field angle (1 deg),
// where distortion is negligible. Returns 0 if the probe ray fails to trace.
// ---------------------------------------------------------------------------
static float paraxial_efl(const OpticalSystem& lens, float lambda_nm, bool horizontal)
{
    const float theta = (float)(1.0 * M_PI / 180.0);
    TraceResult res;
    const bool ok = horizontal
        ? trace_from_ok(lens, 0.0f, 0.0f, theta, 0.0f, lambda_nm, &res)
        : trace_from_ok(lens, 0.0f, 0.0f, 0.0f, theta, lambda_nm, &res);
    if (!ok) return 0.0f;

    const float img_h = horizontal ? res.position.x : res.position.y;
    if (!std::isfinite(img_h)) return 0.0f;
    return std::abs(img_h) / std::tan(theta);
}

// ---------------------------------------------------------------------------
// On-axis marginal-ray half-height along one entrance-plane direction: the
// largest height at which an axial (angle 0) ray still reaches the sensor. This
// is the radius of the beam the system admits — the entrance pupil semi-
// diameter in the renderers' front-element coordinates. Returns 0 if even the
// axial ray fails (degenerate lens); returns the full front semi-aperture when
// the front element itself is the limiting stop.
//
// (ux, uy) must be a unit direction.
// ---------------------------------------------------------------------------
static float marginal_semi_along(const OpticalSystem& lens, float lambda_nm,
                                 float ux, float uy)
{
    const float front_R = lens.surfaces[0].semi_aperture;
    if (front_R <= 0.0f) return 0.0f;
    if (!trace_from_ok(lens, 0.0f, 0.0f, 0.0f, 0.0f, lambda_nm)) return 0.0f;

    const float hx_full = front_R * ux;
    const float hy_full = front_R * uy;
    if (trace_from_ok(lens, hx_full, hy_full, 0.0f, 0.0f, lambda_nm))
        return front_R;

    float lo = 0.0f, hi = front_R;
    for (int it = 0; it < 40; ++it)
    {
        const float mid = 0.5f * (lo + hi);
        const float mx  = mid * ux;
        const float my  = mid * uy;
        if (trace_from_ok(lens, mx, my, 0.0f, 0.0f, lambda_nm)) lo = mid;
        else                                                    hi = mid;
    }
    return lo;
}

static float marginal_semi(const OpticalSystem& lens, float lambda_nm, bool horizontal)
{
    return horizontal ? marginal_semi_along(lens, lambda_nm, 1.0f, 0.0f)
                      : marginal_semi_along(lens, lambda_nm, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// The stop, or nullptr. Only the first is honoured, matching the samplers.
// ---------------------------------------------------------------------------
static const Surface* stop_surface(const OpticalSystem& lens)
{
    for (const auto& s : lens.surfaces)
        if (s.is_stop) return &s;
    return nullptr;
}

// The stop when it is bladed, else nullptr — the gate for everything below.
static const Surface* bladed_stop(const OpticalSystem& lens)
{
    const Surface* s = stop_surface(lens);
    return (s && s->aperture_shape == APERTURE_POLYGON
            && s->aperture_blades >= 3) ? s : nullptr;
}

// ---------------------------------------------------------------------------
// Per-axis support radii of a non-round entrance pupil:
//
//     semi_x = max over theta of h(theta)*|cos theta|
//     semi_y = max over theta of h(theta)*|sin theta|
//
// Azimuths cover every blade tip and waist plus both principal axes.
// ---------------------------------------------------------------------------
static void marginal_support(const OpticalSystem& lens, const Surface& stop,
                             float lambda_nm, float* semi_x, float* semi_y)
{
    const int   PER_SECTOR = 8;                 // tips, waists, and between
    const int   n   = stop.aperture_blades * PER_SECTOR;
    const float rot = stop.aperture_rotation_rad;

    float best_x = 0.0f, best_y = 0.0f;
    auto consider = [&](float theta) {
        const float c = std::cos(theta), s = std::sin(theta);
        const float h = marginal_semi_along(lens, lambda_nm, c, s);
        best_x = std::max(best_x, h * std::fabs(c));
        best_y = std::max(best_y, h * std::fabs(s));
    };
    for (int i = 0; i < n; ++i)
        consider(rot + (float)(2.0 * M_PI) * i / (float)n);
    consider(0.0f);
    consider((float)(M_PI / 2.0));

    *semi_x = best_x;
    *semi_y = best_y;
}

// ===========================================================================
// Covered field is where the spawn disc's surviving area crosses a fraction of
// its axial value. Calibration and source mapping share spawn_probe.h so they
// use one measurement. A launch at u*R - S*b with slope b crosses the front
// vertex at u*R, making the result independent of SPAWN_OFFSET.
// ===========================================================================

// Geometric ladder to bracket both narrow and wide field crossings.
static const float LADDER_DEG[] = {0.0f, 4.0f, 8.0f, 14.0f, 22.0f, 32.0f, 45.0f, 60.0f};
static const int   N_LADDER     = 8;
static const float REFINE_TOL_DEG = 0.15f;
static const int   MAX_REFINE     = 7;

// The image circle feeds a coarse "is the frame bigger than the lens" test with
// a 50% margin, so refining its crossing as hard as the map's buys nothing and
// costs a third of the probe. Its curve is also at its steepest there, which
// makes the interpolated first step unusually good.
static const float CIRCLE_TOL_DEG = 1.0f;
static const int   MAX_REFINE_CIRCLE = 2;

// Where the map is thresholded, and where the illuminated edge is.
//   MAP: the 90%-throughput angle/landing pair used by source mapping.
//   IMAGE CIRCLE: the 5%-throughput illuminated edge.
static const float MAP_THRESHOLD          = 0.90f;
static const float IMAGE_CIRCLE_THRESHOLD = 0.05f;

// One resolved field: the angle where surviving area crosses a threshold, and
// the survivor-mean landing there (which is where the flare actually appears).
struct FieldSolve {
    float angle_rad = 0.0f;
    float half_mm   = 0.0f;
    bool  crossed   = false;   // false: throughput never fell this far
};

// Resolve BOTH thresholds from a single walk out along the field angle: the
// ladder is the expensive part, and the image-circle crossing always lies
// beyond the map crossing, so one pass brackets both.
static void resolve_axis(const OpticalSystem& lens, const PupilMask& mask,
                         float front_R, float pupil_fill, float lambda_nm,
                         bool horizontal,
                         FieldSolve* map_out, FieldSolve* circle_out)
{
    // The seeding rule lives with the probe (spawn_probe.h), so the source map
    // starts its own search from exactly the window calibration does.
    Window track = initial_probe_window(pupil_fill);

    const float DEG = (float)(M_PI / 180.0);

    *map_out    = FieldSolve{};
    *circle_out = FieldSolve{};

    // The axial reference the whole search divides by, and the window size it
    // was measured at. probe_disc() only ever widens, so whenever it does, the
    // reference is re-measured to match before anything is compared to it.
    DiscArea axial;
    float    axial_r = -1.0f;

    auto at = [&](float angle) {
        const DiscArea d = probe_disc(lens, mask, front_R,
                                      horizontal ? angle : 0.0f,
                                      horizontal ? 0.0f  : angle, lambda_nm, &track);
        if (track.r != axial_r) {
            Window ax{0.0f, 0.0f, track.r};
            axial   = probe_window(lens, mask, front_R, 0.0f, 0.0f, lambda_nm, ax);
            axial_r = track.r;
        }
        return d;
    };

    at(0.0f);                          // establishes `axial` at the seed size
    if (axial.area <= 0.0f) return;    // nothing gets through at all

    // Walk the ladder once, recording where each threshold is first crossed.
    // Area weighting makes the curve monotone in practice, so the first
    // crossing is the vignetting edge.
    //
    // Each bracket also remembers where the window was centred at its LOW end.
    // Refinement jumps back inside a bracket the ladder has already walked past,
    // and on a deep-stop lens a centre carried forward from further out no
    // longer sits on the beam. Re-seeding each refinement from its own bracket
    // puts the window back where the beam was.
    int   map_cross = -1, circle_cross = -1;
    float f_prev = 1.0f;
    float f_at_map[2] = {1.0f, 0.0f}, f_at_circle[2] = {1.0f, 0.0f};
    Window seed_map = track, seed_circle = track;
    for (int i = 1; i < N_LADDER; ++i) {
        const Window before = track;
        const float f = at(LADDER_DEG[i] * DEG).area / axial.area;
        if (map_cross < 0 && f < MAP_THRESHOLD) {
            map_cross   = i - 1;
            f_at_map[0] = f_prev; f_at_map[1] = f;
            seed_map    = before;
        }
        if (circle_cross < 0 && f < IMAGE_CIRCLE_THRESHOLD) {
            circle_cross   = i - 1;
            f_at_circle[0] = f_prev; f_at_circle[1] = f;
            seed_circle    = before;
            break;   // the image circle is the outer crossing: nothing beyond
        }
        f_prev = f;
    }

    // Refine a bracket with the Illinois variant of regula falsi. Each step is
    // a whole grid evaluation, and the area curve is smooth once area-weighted,
    // so interpolating converges in a fraction of the steps bisection needs —
    // which is most of what keeps this inside the calibration budget.
    auto refine = [&](int cross, float thr, const float f_ends[2],
                      const Window& seed, int max_steps, float tol_deg) -> FieldSolve {
        FieldSolve r;
        // Every evaluation starts from where the beam was at the bracket's low
        // end. Only the centre is restored — the size is left at whatever the
        // search has grown to, because shrinking it back would re-trigger the
        // widening (and with it a re-measured axial reference) on every step.
        auto probe_at = [&](float angle) {
            track.cu = seed.cu;
            track.cv = seed.cv;
            return at(angle);
        };
        if (cross < 0) {
            // Never crossed inside the ladder. Report the ceiling and say so:
            // the number is a floor, not a measurement.
            const DiscArea d = probe_at(LADDER_DEG[N_LADDER - 1] * DEG);
            r.angle_rad = LADDER_DEG[N_LADDER - 1] * DEG;
            r.half_mm   = std::abs(horizontal ? d.mean_x : d.mean_y);
            r.crossed   = false;
            return r;
        }
        float lo = LADDER_DEG[cross] * DEG, hi = LADDER_DEG[cross + 1] * DEG;
        float f_lo = f_ends[0] - thr, f_hi = f_ends[1] - thr;
        // Illinois halves the stagnant end's value to force convergence, which
        // makes f_lo/f_hi useless for interpolating the answer. Keep the real
        // measurements at the bracket ends alongside them.
        float t_lo = f_lo, t_hi = f_hi;
        const float tol = tol_deg * DEG;

        for (int it = 0; it < max_steps && (hi - lo) > tol; ++it) {
            const float denom = f_lo - f_hi;
            float mid = (std::abs(denom) < 1e-12f)
                            ? 0.5f * (lo + hi)
                            : lo + (hi - lo) * f_lo / denom;
            const float edge = 0.01f * (hi - lo);          // stay inside
            mid = std::min(std::max(mid, lo + edge), hi - edge);

            const float f_mid = probe_at(mid).area / axial.area - thr;
            if (f_mid >= 0.0f) { lo = mid; f_lo = f_mid; t_lo = f_mid; f_hi *= 0.5f; }
            else               { hi = mid; f_hi = f_mid; t_hi = f_mid; f_lo *= 0.5f; }
        }

        // Interpolate the crossing across the final bracket rather than
        // returning `lo`. Convergence is usually one-sided — every trial can
        // land above the threshold and leave `lo` sitting on the ladder step it
        // started from, which would report a covered field a whole ladder step
        // too small.
        const float span = t_lo - t_hi;
        float root = (std::abs(span) < 1e-12f) ? 0.5f * (lo + hi)
                                               : lo + (hi - lo) * t_lo / span;
        root = std::min(std::max(root, lo), hi);

        DiscArea d = probe_at(root);
        if (d.area <= 0.0f) {
            // Interpolating a low threshold can overshoot into the dead zone
            // past the last light — the image-circle crossing sits on a steep
            // tail, and a hair beyond it nothing survives at all. There is no
            // landing to average there, so fall back to `lo`, which crossed the
            // threshold from above and therefore has survivors by construction.
            root = lo;
            d    = probe_at(lo);
        }
        r.angle_rad = root;
        r.half_mm   = std::abs(horizontal ? d.mean_x : d.mean_y);
        r.crossed   = true;
        return r;
    };

    *map_out    = refine(map_cross,    MAP_THRESHOLD,          f_at_map,    seed_map,
                         MAX_REFINE,        REFINE_TOL_DEG);
    *circle_out = refine(circle_cross, IMAGE_CIRCLE_THRESHOLD, f_at_circle, seed_circle,
                         MAX_REFINE_CIRCLE, CIRCLE_TOL_DEG);

    // The lower threshold cannot cross first. It can still come back smaller
    // when a lens falls from above the map threshold to below the image-circle
    // one inside a single ladder step: both then refine the same bracket, and
    // the circle gets the cheaper refinement. Hold it to the covered field
    // rather than publishing an image circle inside it.
    if (circle_out->angle_rad < map_out->angle_rad) *circle_out = *map_out;
    if (circle_out->half_mm   < map_out->half_mm)   circle_out->half_mm = map_out->half_mm;
}

// Is the system rotationally symmetric, so the two axes must resolve alike?
// When it is, one search answers both — which is exact, not an approximation,
// and halves the probe's cost on the majority of lenses.
static bool axes_are_symmetric(const OpticalSystem& lens)
{
    static const float IDENTITY[9] = {1.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f};
    for (const auto& s : lens.surfaces) {
        if (s.form == FORM_CYLINDRICAL)  return false;   // anamorphic by design
        if (s.aperture_aspect != 1.0f)   return false;   // elliptical clear bound
        if (s.decenter_x != 0.0f || s.decenter_y != 0.0f) return false;
        for (int k = 0; k < 9; ++k)
            if (s.rot[k] != IDENTITY[k]) return false;   // tilted
        // The aperture shape has to be round too. A polygon reaches its full
        // radius at a vertex and only its apothem at a flat, so unless the
        // blade count is a multiple of four and unrotated the two axes clip
        // differently; an image matte can differ arbitrarily.
        if (s.aperture_shape == APERTURE_POLYGON) {
            if (s.aperture_blades % 4 != 0)          return false;
            if (s.aperture_rotation_rad != 0.0f)     return false;
        } else if (s.aperture_shape != APERTURE_CIRCLE) {
            return false;                            // image matte, or unknown
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
LensCalibration calibrate_lens(const OpticalSystem&         lens,
                                float                     d_line_nm)
{
    LensCalibration calib{};

    if (lens.surfaces.empty())
        return calib;

    // First-order diffraction-pupil scale, solved independently per axis.
    calib.focal_length_x        = paraxial_efl(lens, d_line_nm, true);
    calib.focal_length_y        = paraxial_efl(lens, d_line_nm, false);

    // Bladed stops require a support sweep; round stops are exact on the two
    // principal meridians.
    if (const Surface* stop = bladed_stop(lens))
    {
        marginal_support(lens, *stop, d_line_nm,
                         &calib.entrance_pupil_semi_x,
                         &calib.entrance_pupil_semi_y);
        calib.pupil_area_frac = stop->aperture_profile.area_frac;
    }
    else
    {
        calib.entrance_pupil_semi_x = marginal_semi(lens, d_line_nm, true);
        calib.entrance_pupil_semi_y = marginal_semi(lens, d_line_nm, false);
    }

    // Covered field + image circle, probed over the disc the renderers spawn.
    {
        const PupilMask mask    = resolve_pupil_mask(lens, ApertureSamplerParams{});
        const float     front_R = lens.surfaces[0].semi_aperture;

        // How much of the front element admits light, per axis: decides
        // whether the probe can spread its grid over the whole disc.
        const float fill_x = (front_R > 0.0f) ? calib.entrance_pupil_semi_x / front_R : 1.0f;
        const float fill_y = (front_R > 0.0f) ? calib.entrance_pupil_semi_y / front_R : 1.0f;

        FieldSolve map_h, map_v, circle_h, circle_v;
        resolve_axis(lens, mask, front_R, std::min(fill_x, fill_y), d_line_nm,
                     true, &map_h, &circle_h);
        if (axes_are_symmetric(lens)) {
            map_v    = map_h;
            circle_v = circle_h;
            // The vertical landing of a horizontal probe is 0 by symmetry, so
            // carry the magnitude across rather than the measured component.
        } else {
            resolve_axis(lens, mask, front_R, std::min(fill_x, fill_y), d_line_nm,
                         false, &map_v, &circle_v);
        }

        calib.max_half_angle_h    = map_h.angle_rad;
        calib.max_half_angle_v    = map_v.angle_rad;
        calib.sensor_half_w       = map_h.half_mm;
        calib.sensor_half_h       = map_v.half_mm;
        calib.image_circle_semi_w = circle_h.half_mm;
        calib.image_circle_semi_h = circle_v.half_mm;

        if (!map_h.crossed || !map_v.crossed)
            std::fprintf(stderr,
                         "[ghostlight] warning: covered-field probe never fell to "
                         "%.0f%% of axial within %.0f deg; the calibrated field "
                         "is the search ceiling, so field-based sizing is "
                         "truncated.\n",
                         (double)(MAP_THRESHOLD * 100.0f),
                         (double)LADDER_DEG[N_LADDER - 1]);

        // Downstream divides by both of these — the source-position map forms
        // sensor_half / tan(max_half_angle) with no guard of its own — so a
        // lens that resolves to nothing must not leave a zero here. Two ways to
        // get one: nothing traces at all (the probe returns a zero solve), or
        // the lens vignettes so hard that the crossing lands on the ladder's
        // own zero. Fall back to the lens file's focal length, and say so.
        const float MIN_FIELD_RAD = (float)(0.05 * M_PI / 180.0);
        auto floor_axis = [&](float* angle, float* half, const char* axis) {
            if (*angle >= MIN_FIELD_RAD && *half > 0.0f) return;
            std::fprintf(stderr,
                         "[ghostlight] warning: %s covered field resolved to "
                         "nothing (%.4f deg, %.4f mm); falling back to the lens "
                         "file's focal length. Check the lens is focused at "
                         "z = 0.\n",
                         axis, (double)(*angle * 180.0 / M_PI), (double)*half);
            *angle = MIN_FIELD_RAD;
            *half  = (lens.focal_length > 0.0f)
                         ? lens.focal_length * std::tan(MIN_FIELD_RAD)
                         : std::tan(MIN_FIELD_RAD);   // 1 mm of "focal length"
        };
        floor_axis(&calib.max_half_angle_h, &calib.sensor_half_w, "horizontal");
        floor_axis(&calib.max_half_angle_v, &calib.sensor_half_h, "vertical");

        if (calib.image_circle_semi_w <= 0.0f) calib.image_circle_semi_w = calib.sensor_half_w;
        if (calib.image_circle_semi_h <= 0.0f) calib.image_circle_semi_h = calib.sensor_half_h;
    }

    if (calib.entrance_pupil_semi_x > 0.0f)
        calib.f_number_x = calib.focal_length_x / (2.0f * calib.entrance_pupil_semi_x);
    if (calib.entrance_pupil_semi_y > 0.0f)
        calib.f_number_y = calib.focal_length_y / (2.0f * calib.entrance_pupil_semi_y);

    return calib;
}
