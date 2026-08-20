// ============================================================================
// PSFRenderer.cpp — CPU orchestration for the PSF grid renderer.
//
// Validates inputs, runs a CPU chief-ray pre-pass for tile centring, and
// hands off to launch_psf_render().
// ============================================================================

#include "PSFRenderer.h"

#include "aperture_sampler.h" // PupilMask, resolve_pupil_mask
#include "newton_aim.h"
#include "spawn_probe.h"  // Window, DiscArea, probe_window/probe_disc
#include "psf_render.h"
#include "spawn_plane.h"    // SPAWN_OFFSET, spawn_shift
#include "trace.h"          // trace_primary_ray (CPU)
#include "trace_event.h"    // TraceStatus

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Reference wavelength for chief-ray pre-pass — the "primary lambda" (d-line,
// 587.56 nm).
static constexpr float kChiefRayLambdaNm = 587.56f;

// Trace one pupil-sample for tile centring.  Returns the landing at z=0.
static bool trace_chief_candidate(const OpticalSystem& lens,
                                  float start_z, float front_R,
                                  float angle_x, float angle_y,
                                  float u, float v,
                                  float& out_x_mm, float& out_y_mm)
{
    const float bx = std::tan(angle_x);
    const float by = std::tan(angle_y);

    // Track the beam off axis (spawn_plane.h), matching the GPU kernel's spawn.
    // Every caller — the survivor-mean probe and each Newton aim iterate — comes
    // through here, so the shift follows the aimed angle as it is refined.
    float sdx, sdy;
    spawn_shift(bx, by, sdx, sdy);

    Ray r;
    r.origin = Vec3f(u * front_R + sdx, v * front_R + sdy, start_z);
    r.dir    = Vec3f(bx, by, 1.0f).normalized();
    r.lambda = kChiefRayLambdaNm;

    TraceResult tr = trace_primary_ray(r, lens);
    if (tr.status != TraceStatus::OK
        || !std::isfinite(tr.position.x) || !std::isfinite(tr.position.y))
        return false;
    out_x_mm = tr.position.x;
    out_y_mm = tr.position.y;
    return true;
}

// ---------------------------------------------------------------------------
// Pupil-mean and aiming helpers.
//
// CHIEF_CENTROID mode centres each tile on the unweighted mean of a 7×7 unit-
// disk pupil probe (the centroid of the geometric ray cloud).  FIXED_TARGET
// mode instead aims the chief ray at a sensor-plane cell centre — see
// aim_chief_ray.  Both build on pupil_probe below.
// ---------------------------------------------------------------------------

// Result of tracing an N×N unit-disk pupil probe at one field direction.
struct PupilProbe {
    int   hits   = 0;      // transmitting samples
    float frac   = 0.0f;   // hits / disk-sample count  (0 when nothing transmits)
    float mean_x = 0.0f;   // survivor-mean landing (mm); 0 when hits == 0
    float mean_y = 0.0f;
};

static PupilProbe pupil_probe(const OpticalSystem& lens,
                              float start_z, float front_R,
                              float angle_x, float angle_y, int N)
{
    float sx = 0.0f, sy = 0.0f;
    int hits = 0, total = 0;
    for (int iy = 0; iy < N; ++iy)
    {
        const float v = -1.0f + (iy + 0.5f) * (2.0f / (float)N);
        for (int ix = 0; ix < N; ++ix)
        {
            const float u = -1.0f + (ix + 0.5f) * (2.0f / (float)N);
            if (u*u + v*v > 1.0f) continue;
            ++total;
            float x, y;
            if (trace_chief_candidate(lens, start_z, front_R, angle_x, angle_y, u, v, x, y))
            {
                sx += x; sy += y; ++hits;
            }
        }
    }
    PupilProbe p;
    p.hits   = hits;
    p.frac   = (total > 0) ? (float)hits / (float)total : 0.0f;
    p.mean_x = (hits > 0) ? sx / (float)hits : 0.0f;
    p.mean_y = (hits > 0) ? sy / (float)hits : 0.0f;
    return p;
}

// Where the pupil is on axis, measured once per render and reused as the seed
// for the deep-stop probe below. Empty (r = 0) when nothing transmits at all.
struct AxialPupil {
    PupilMask mask;
    Window    win;
    bool      usable = false;
};

static AxialPupil measure_axial_pupil(const OpticalSystem& lens, float front_R)
{
    AxialPupil a;
    a.mask = resolve_pupil_mask(lens, ApertureSamplerParams{});
    Window full;                                   // the whole front disc
    const DiscArea d = probe_window(lens, a.mask, front_R, 0.0f, 0.0f,
                                    kChiefRayLambdaNm, full);
    if (d.area > 0.0f) { a.win = d.fit; a.usable = true; }
    return a;
}

// Aim the chief ray of one cell at its sensor-plane target.  Reports the aimed
// landing, a PSFCellStatus, the residual, and the pupil transmission fraction.
// out_ax/out_ay receive the refined field direction — the kernel MUST trace the
// pupil cloud at this aimed angle (not the seed), so the cloud lands on the
// target the tile is anchored to.
static void aim_chief_ray(const OpticalSystem& lens,
                          float start_z, float front_R,
                          const AxialPupil& axial,
                          float seed_ax, float seed_ay,
                          float target_x, float target_y,
                          float& out_ax, float& out_ay,
                          float& out_x, float& out_y,
                          uint8_t& out_status, float& out_residual,
                          float& out_pupil_fraction)
{
    static constexpr int   kMaxIter    = 8;
    static constexpr float kFdStep     = 1e-4f;   // rad
    static constexpr float kTolMm      = 1e-4f;   // 0.1 µm — clean-chief convergence
    static constexpr float kTolSoftMm  = 5e-3f;   // 5 µm — vignetted fallback
    static constexpr int   kAimProbeN  = 9;       // fallback survivor-mean probe (fast)
    static constexpr int   kFracProbeN = 17;      // pupil_fraction probe (~227 disk samples)
    static constexpr int   kMinHits    = 4;

    // The tracked-window finite difference must exceed survivor-mean quantization.
    static constexpr float kDeepFdStep   = 1e-3f;  // rad
    static constexpr float kDeepWarmStep = 0.10f;  // rad between warm-up steps

    const float seed_mag = std::sqrt(seed_ax*seed_ax + seed_ay*seed_ay);
    const float step_cap = std::max(0.02f, 0.5f * seed_mag);

    // Aim the u=v=0 primary that centers the GPU pupil cloud. If it clips,
    // use the survivor mean below.

    // --- Clean chief: refine the u=v=0 primary ray onto the target. ---
    // A transmitting u=v=0 ray is reported OK with the best Newton iterate.
    {
        auto landing = [&](float ax, float ay, float& x, float& y) -> bool {
            return trace_chief_candidate(lens, start_z, front_R, ax, ay, 0.0f, 0.0f, x, y);
        };
        float ax = seed_ax, ay = seed_ay, tmp_x, tmp_y;
        if (landing(ax, ay, tmp_x, tmp_y))
        {
            float res = 0.0f;
            newton_aim(landing, target_x, target_y, ax, ay, res,
                       kMaxIter, kFdStep, step_cap, kTolMm);
            landing(ax, ay, out_x, out_y);
            out_ax = ax; out_ay = ay;
            out_status         = PSF_CELL_OK;
            out_residual       = res;
            out_pupil_fraction = pupil_probe(lens, start_z, front_R, ax, ay, kFracProbeN).frac;
            return;
        }
    }

    // Fallback: aim the partial pupil's survivor mean at the target.
    {
        auto landing = [&](float ax, float ay, float& x, float& y) -> bool {
            const PupilProbe p = pupil_probe(lens, start_z, front_R, ax, ay, kAimProbeN);
            if (p.hits < kMinHits) return false;
            x = p.mean_x; y = p.mean_y;
            return true;
        };
        float ax = seed_ax, ay = seed_ay, tmp_x, tmp_y;
        if (landing(ax, ay, tmp_x, tmp_y))
        {
            float res = 0.0f;
            newton_aim(landing, target_x, target_y, ax, ay, res,
                       kMaxIter, kFdStep, step_cap, kTolSoftMm);
            landing(ax, ay, out_x, out_y);
            out_ax = ax; out_ay = ay;
            out_status         = PSF_CELL_CHIEF_VIGNETTED;
            out_residual       = res;
            out_pupil_fraction = pupil_probe(lens, start_z, front_R, ax, ay, kFracProbeN).frac;
            return;
        }
    }

    // A deep pupil can fall between samples of the full front-disc probes. Track
    // the calibrated transmitting window across the field as a final attempt.
    if (axial.usable)
    {
        Window track = axial.win;
        auto landing = [&](float ax, float ay, float& x, float& y) -> bool {
            Window w = track;                 // probe_disc mutates its tracker
            const DiscArea d = probe_disc(lens, axial.mask, front_R,
                                          ax, ay, kChiefRayLambdaNm, &w);
            if (!(d.area > 0.0f)) return false;
            if (!std::isfinite(d.mean_x) || !std::isfinite(d.mean_y)) return false;
            track = w;
            x = d.mean_x; y = d.mean_y;
            return true;
        };

        // Walk the window out to the seed rather than jumping: the pupil moves
        // across the front element with field angle, and a window that arrives
        // where the beam no longer is finds nothing and widens back to the disc
        // this fallback exists to avoid.
        if (track.r < 1.0f && seed_mag > 0.0f) {
            const int steps = std::max(3, std::min(16,
                                  (int)std::ceil(seed_mag / kDeepWarmStep)));
            float tx, ty;
            for (int k = 1; k <= steps; ++k) {
                const float f = (float)k / (float)steps;
                landing(seed_ax * f, seed_ay * f, tx, ty);
            }
        }

        float ax = seed_ax, ay = seed_ay, tmp_x, tmp_y;
        if (landing(ax, ay, tmp_x, tmp_y))
        {
            float res = 0.0f;
            newton_aim(landing, target_x, target_y, ax, ay, res,
                       kMaxIter, kDeepFdStep, step_cap, kTolSoftMm);
            if (landing(ax, ay, out_x, out_y))
            {
                out_ax = ax; out_ay = ay;
                out_status         = PSF_CELL_CHIEF_VIGNETTED;
                out_residual       = res;
                // Report transmission relative to the full unit disc.
                out_pupil_fraction = pupil_probe(lens, start_z, front_R, ax, ay,
                                                 kFracProbeN).frac;
                return;
            }
        }
    }

    // --- No transmission near the seed: anchor the (empty) tile on the cell. ---
    out_ax = seed_ax; out_ay = seed_ay;
    out_x = target_x; out_y = target_y;
    out_status         = PSF_CELL_DARK;
    out_residual       = 1e30f;
    out_pupil_fraction = 0.0f;
}

bool render_psf(const OpticalSystem&         lens,
                const std::vector<PSFSource>& sources,
                const PSFConfig&             cfg,
                PSFOutput&                   out)
{
    // ---- Input validation ----
    if (cfg.grid_nx <= 0 || cfg.grid_ny <= 0 || cfg.tile_w <= 0 || cfg.tile_h <= 0) {
        fprintf(stderr, "PSFRenderer: grid_nx/ny and tile_w/h must be positive "
                        "(got %d, %d, %d, %d)\n",
                cfg.grid_nx, cfg.grid_ny, cfg.tile_w, cfg.tile_h);
        return false;
    }
    if (!(cfg.tile_extent_mm > 0.0f) || !std::isfinite(cfg.tile_extent_mm)) {
        fprintf(stderr, "PSFRenderer: tile_extent_mm must be positive (got %g)\n",
                cfg.tile_extent_mm);
        return false;
    }
    const int64_t n_tiles64 = (int64_t)cfg.grid_nx * cfg.grid_ny;
    if (n_tiles64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "PSFRenderer: tile grid dimensions overflow\n");
        return false;
    }
    const int n_tiles = (int)n_tiles64;
    if (sources.size() > (size_t)n_tiles) {
        fprintf(stderr, "PSFRenderer: %zu sources exceeds tile capacity %d "
                        "(grid_nx * grid_ny)\n",
                sources.size(), n_tiles);
        return false;
    }
    if (lens.num_surfaces() <= 0) {
        fprintf(stderr, "PSFRenderer: lens has no surfaces\n");
        return false;
    }
    for (const PSFSource& source : sources) {
        if (!std::isfinite(source.angle_x) || !std::isfinite(source.angle_y)
            || !std::isfinite(source.r) || !std::isfinite(source.g)
            || !std::isfinite(source.b)
            || !std::isfinite(source.target_x_mm)
            || !std::isfinite(source.target_y_mm)) {
            fprintf(stderr, "PSFRenderer: source values must be finite\n");
            return false;
        }
    }

    const int n_sources    = (int)sources.size();
    const int64_t composite_w64 = (int64_t)cfg.grid_nx * cfg.tile_w;
    const int64_t composite_h64 = (int64_t)cfg.grid_ny * cfg.tile_h;
    if (composite_w64 > std::numeric_limits<int>::max()
        || composite_h64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "PSFRenderer: composite dimensions overflow\n");
        return false;
    }
    const int composite_w  = (int)composite_w64;
    const int composite_h  = (int)composite_h64;
    if ((size_t)composite_w > std::numeric_limits<size_t>::max()
                              / (size_t)composite_h) {
        fprintf(stderr, "PSFRenderer: composite pixel count overflow\n");
        return false;
    }
    const size_t n_px      = (size_t)composite_w * composite_h;
    if (n_px > (size_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "PSFRenderer: composite has too many pixels\n");
        return false;
    }
    PSFOutput next;
    const float mm_per_pixel = cfg.tile_extent_mm / (float)cfg.tile_w;
    // Note: tile_extent_mm is the per-tile extent on *both* axes; tile_w/h may
    // differ, in which case the vertical mm-per-pixel matches the horizontal
    // (square pixels), and the visible mm-extent in y becomes
    // tile_extent_mm * (tile_h / tile_w).

    // ---- Chief-ray pre-pass (CPU) ----
    // CHIEF_CENTROID: each tile is centred on the unweighted pupil-mean landing.
    // FIXED_TARGET:   each tile is centred on the cell target, reached by aiming
    //                 the chief ray at it (see aim_chief_ray).
    // Both modes populate the diagnostic status, pupil fraction, and residual.
    static constexpr int kCentroidProbeN = 7;
    const float start_z = lens.surfaces[0].z - SPAWN_OFFSET;
    const float front_R = lens.surfaces[0].semi_aperture;

    // Where the pupil sits on axis. One probe for the whole render, shared by
    // the deep-stop aim and by the centroid path below —
    // both of which otherwise sample the whole front disc and starve on a lens
    // whose stop is much smaller than its front element.
    const AxialPupil axial = measure_axial_pupil(lens, front_R);
    next.chief_x_mm.assign(n_sources, 0.0f);
    next.chief_y_mm.assign(n_sources, 0.0f);
    next.status.assign(n_sources, PSF_CELL_OK);
    next.pupil_fraction.assign(n_sources, 0.0f);
    next.aim_residual_mm.assign(n_sources, 0.0f);

    const bool target_mode = (cfg.center_mode == PSF_CENTER_FIXED_TARGET);

    std::vector<PSFGpuSource> gpu_sources(n_sources);
    for (int i = 0; i < n_sources; ++i)
    {
        const PSFSource& s = sources[i];

        float   chief_x = 0.0f, chief_y = 0.0f, residual = 0.0f, frac = 0.0f;
        float   beam_ax = s.angle_x, beam_ay = s.angle_y;  // direction the kernel traces
        uint8_t status  = PSF_CELL_OK;

        if (target_mode)
        {
            // Aim refines the field direction so the chief lands on the cell
            // target; the kernel must trace the pupil cloud at that aimed angle.
            aim_chief_ray(lens, start_z, front_R, axial,
                          s.angle_x, s.angle_y,        // aim seed
                          s.target_x_mm, s.target_y_mm,
                          beam_ax, beam_ay,            // aimed direction (out)
                          chief_x, chief_y, status, residual, frac);
        }
        else
        {
            // Pupil-mean landing (0,0 when nothing transmits). Reuse the
            // probe to report a coarse status/fraction; residual is meaningless.
            const PupilProbe p = pupil_probe(lens, start_z, front_R,
                                             s.angle_x, s.angle_y, kCentroidProbeN);
            chief_x = p.mean_x;
            chief_y = p.mean_y;
            status  = (p.hits > 0) ? PSF_CELL_OK : PSF_CELL_DARK;
            frac    = p.frac;

            // Retry within the tracked axial window when the full-disc probe
            // misses a deep pupil.
            if (p.hits == 0 && axial.usable)
            {
                Window w = axial.win;
                const float mag = std::sqrt(s.angle_x * s.angle_x
                                          + s.angle_y * s.angle_y);
                if (w.r < 1.0f && mag > 0.0f) {
                    const int steps = std::max(3, std::min(16,
                                          (int)std::ceil(mag / 0.10f)));
                    for (int k = 1; k <= steps; ++k) {
                        const float f = (float)k / (float)steps;
                        probe_disc(lens, axial.mask, front_R,
                                   s.angle_x * f, s.angle_y * f,
                                   kChiefRayLambdaNm, &w);
                    }
                }
                const DiscArea d = probe_disc(lens, axial.mask, front_R,
                                              s.angle_x, s.angle_y,
                                              kChiefRayLambdaNm, &w);
                if (d.area > 0.0f
                    && std::isfinite(d.mean_x) && std::isfinite(d.mean_y))
                {
                    chief_x = d.mean_x;
                    chief_y = d.mean_y;
                    status  = PSF_CELL_OK;
                    // Keep frac as the full-disc area measurement.
                }
            }
        }

        next.chief_x_mm[i]      = chief_x;
        next.chief_y_mm[i]      = chief_y;
        next.status[i]          = status;
        next.pupil_fraction[i]  = frac;
        next.aim_residual_mm[i] = std::isfinite(residual) ? residual : 0.0f;

        // Tile origin in composite-buffer pixel space (row-major layout).
        const int gx = i % cfg.grid_nx;
        const int gy = i / cfg.grid_nx;

        PSFGpuSource gs;
        gs.angle_x    = beam_ax;
        gs.angle_y    = beam_ay;
        gs.r          = s.r;
        gs.g          = s.g;
        gs.b          = s.b;
        gs.chief_x_mm = chief_x;
        gs.chief_y_mm = chief_y;
        gs.tile_x0    = gx * cfg.tile_w;
        gs.tile_y0    = gy * cfg.tile_h;
        gpu_sources[i] = gs;
    }

    // ---- GPU render ----
    std::vector<float> out_r(n_px, 0.0f), out_g(n_px, 0.0f), out_b(n_px, 0.0f);

    std::string cuda_error;
    launch_psf_render(lens, gpu_sources,
                      cfg.tile_w, cfg.tile_h,
                      composite_w, composite_h,
                      mm_per_pixel,
                      cfg.monochromatic,
                      out_r.data(), out_g.data(), out_b.data(),
                      cfg, &cuda_error);
    if (!cuda_error.empty()) {
        fprintf(stderr, "PSFRenderer: %s\n", cuda_error.c_str());
        return false;
    }

    next.out_r          = std::move(out_r);
    next.out_g          = std::move(out_g);
    next.out_b          = std::move(out_b);
    next.composite_w    = composite_w;
    next.composite_h    = composite_h;
    next.tile_w         = cfg.tile_w;
    next.tile_h         = cfg.tile_h;
    next.grid_nx        = cfg.grid_nx;
    next.grid_ny        = cfg.grid_ny;
    next.tile_extent_mm = cfg.tile_extent_mm;
    out = std::move(next);
    return true;
}
