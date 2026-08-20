// ============================================================================
// flare_buffers.h — Shared output type for PointFlareRenderer + SourceFlareRenderer
// ============================================================================
#pragma once

#include <vector>

// Per-ghost-pair AOV layer — one triplet per active GhostPair when
// aov_mode == GhostAovMode::PER_PAIR.  Identified by physical surface indices.
struct FlareAovLayer {
    int surf_a, surf_b;              // identifies the GhostPair
    std::vector<float> r, g, b;      // width×height, post-blur
};

// Per-render diagnostics (instrumentation).
//
// Populated only when RenderConfig::collect_stats is true; otherwise FlareBuffers
// leaves has_stats false and every field here at its default. Timing is in
// milliseconds; a value < 0 means "not measured" (e.g. no active ghost pairs).
// The per-pair vectors all share length n_pairs and are index-aligned.
struct GhostRenderStats {
    // Stage timing (ms). CPU stages measured with std::chrono; the kernel stage
    // is the real (un-instrumented) launch. Stages need not sum to ms_total.
    float ms_grid_build = -1.0f;  // CPU: entrance-pupil grid + spectral table + host packing
    float ms_upload     = -1.0f;  // device alloc + H2D copies + output memset
    float ms_kernel     = -1.0f;  // ghost_kernel execution (real render, not the stats pass)
    float ms_download   = -1.0f;  // D2H copies
    float ms_total      = -1.0f;  // whole launch_ghost_render wall time

    // Launch dimensions.
    int n_pairs   = 0;   // active ghost pairs rendered
    int n_sources = 0;
    int n_grid    = 0;   // entrance-pupil samples after the aperture mask
    int n_spec    = 0;   // spectral samples

    // Survivor totals over all pairs x sources x grid x spectral. A "trace" is
    // one (pupil sample, spectral sample) ray through one ghost pair.
    unsigned long long traces_total     = 0;  // traces attempted (the denominator)
    unsigned long long traces_survived  = 0;  // trace returned OK (left the lens)
    unsigned long long traces_on_sensor = 0;  // OK AND splat centre inside the frame

    // Per-pair breakdown (each length n_pairs), summed over sources/grid/spectral.
    std::vector<int>                pair_surf_a, pair_surf_b;
    std::vector<unsigned long long> pair_traces, pair_survived, pair_on_sensor;

    // Concentration telemetry. Length
    // n_ps = n_pairs * n_sources, indexed ps = pair_idx * n_sources + src_idx
    // (so with a single source ps == the pair index and aligns with the
    // per-pair vectors above). Populated only when the concentration probe ran
    // (concentrate_samples on, n_pairs > 1, spectral != 3); empty otherwise.
    //   ps_hits: distinct on-sensor probe survivor positions for that
    //            (pair, source) — the value compared against CONC_MIN_HITS.
    //            hits < CONC_MIN_HITS => that (pair, source) falls back to
    //            full-pupil sampling (the speckle); >= => concentrated (smooth).
    //   ps_rect: 4 floats per ps {u0, v0, u1, v1}. u0 > u1 marks the fallback /
    //            invalid rect; otherwise it is the concentrated survivor box.
    std::vector<int>   ps_hits;
    std::vector<float> ps_rect;

    // Per-(pair, source) adaptive sample budget (the
    // n_r_ps sample count). Populated only when adaptive_sample_budgets is on;
    // lets a diagnostic report the budget distribution and the trace-time-weighted
    // budget fraction (Sum budget / Sum n_grid over live ps = the speedup proxy).
    std::vector<int>   ps_budget;
};

// Output buffers produced by the flare renderers.
// starburst_r/g/b is populated only when DiffractionConfig::starburst is set;
// left empty otherwise (the dict binding then omits the keys). It is a separate
// additive layer, composited on top of the ghost layer by the caller.
// veil_r/g/b is the additive veiling-glare layer, populated only when
// DiffractionConfig::veil is set and included in metering.
// gate_r/g/b is the film-gate scatter layer, populated only when
// GateConfig::enabled is set and is included in metering.
// aov_layers is empty unless aov_mode == GhostAovMode::PER_PAIR.
struct FlareBuffers {
    std::vector<float> ghost_r,  ghost_g,  ghost_b;
    std::vector<float> starburst_r, starburst_g, starburst_b;
    std::vector<float> veil_r, veil_g, veil_b;
    std::vector<float> gate_r, gate_g, gate_b;
    std::vector<FlareAovLayer> aov_layers;
    int width = 0, height = 0;

    bool             has_stats = false;  // true when stats below are populated
    GhostRenderStats stats;
};
