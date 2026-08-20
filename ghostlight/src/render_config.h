// ============================================================================
// render_config.h — Config hierarchy shared by the flare + PSF renderers
// ============================================================================
#pragma once

#include "spectral.h"

#include <utility>
#include <vector>

// Base quality settings shared by all three renderers.
// lens_file is NOT a config field — the caller loads the OpticalSystem and calls
// calibrate_lens() once, then passes both into each render call explicitly.
struct RenderConfig {
    int   ray_grid         = 64;
    // Three samples select the fixed 650/550/450 nm RGB preview path, which
    // bypasses XYZ conversion and output_cs.
    int   spectral_samples = 16;
    int   pupil_jitter     = 0;
    int   jitter_seed      = 0;

    SensorModel      sensor_model = SENSOR_CIE_1931;
    InputColorSpace  input_cs     = CS_INPUT_ACESCG;
    OutputColorSpace output_cs    = CS_ACESCG;
    float custom_input_to_xyz[3][3]  = {};
    float custom_xyz_to_output[3][3] = {};

    float sensor_half_w = 0.0f;  // mm half-width of capture plane; 0 = use calibrated value
    float sensor_half_h = 0.0f;  // mm half-height of capture plane; 0 = use calibrated value

    bool verbose = false;  // enable stdout progress/timing prints from render paths

    // Adds an instrumented kernel launch and returns stage timing and per-pair
    // survivor counters in the output's "stats" entry.
    bool collect_stats = false;

    // Corrects CMF integration bias across spectral sample counts in output
    // space without altering lens-dependent dispersion. The fixed three-sample
    // RGB path is not corrected.
    bool spectral_correction = true;
};

enum class GhostAovMode : int {
    NONE     = 0,   // default — no per-pair buffers
    PER_PAIR = 1,   // one buffer triplet per active ghost pair
};

// Applied after filter_ghost_pairs() and before GPU scattering. Pairs use
// canonical surface-index ordering (surf_a < surf_b). AOV buffers correspond
// to the filtered pair list.
struct GhostFilter {
    enum class Mode : int {
        ALL     = 0, // ignore `pairs` — render every active ghost (default)
        INCLUDE = 1, // render only ghosts listed in `pairs`
        EXCLUDE = 2, // render every active ghost EXCEPT those listed in `pairs`
    };
    Mode mode = Mode::ALL;
    // (surf_a, surf_b) tuples; comparison is order-sensitive — callers
    // must match the canonical GhostPair ordering (surf_a < surf_b).
    std::vector<std::pair<int, int>> pairs;
};

// Rectangular matte-box / barn-door flags that clip the entrance beam in a
// plane ahead of the front element. Designed to be consumed by BOTH the ray
// tracer (kill flagged rays) and the diffraction pupil builder (clip the pupil
// amplitude) from one definition, so the geometric flare and its diffraction
// always agree on the aperture.
//
// The flag plane sits `z_front_mm` ahead of the front vertex (toward the
// scene). Each ray is projected along its own direction to that plane; it is
// blocked when its (x, y) there exceeds any active half-extent. Offsets are
// half-widths from the optical axis in millimetres; a value >= MATTE_BOX_OPEN
// leaves that side unflagged. (top/bottom are +y/-y; right/left are +x/-x.)
constexpr float MATTE_BOX_OPEN = 1.0e9f;

struct MatteBox {
    bool  enabled    = false;
    float z_front_mm = 60.0f;             // flag plane distance ahead of the front vertex
    float top        = MATTE_BOX_OPEN;    // +y half-extent (mm)
    float bottom     = MATTE_BOX_OPEN;    // -y half-extent (mm)
    float left       = MATTE_BOX_OPEN;    // -x half-extent (mm)
    float right      = MATTE_BOX_OPEN;    // +x half-extent (mm)
};

// Front-of-lens clipping planes. build_gpu_baffles() merges the matte box as a
// rectangular baffle. Ray tracing, HURB, and starburst generation share these
// edges.
enum class BaffleShape : int {
    RECT   = 0,   // rectangle: top/bottom/left/right half-extents (matte-box form)
    CIRCLE = 1,   // circle / ellipse: centre (cx,cy), radius, aspect (hood / mount rim)
};

struct Baffle {
    BaffleShape shape = BaffleShape::RECT;
    float z_front_mm  = 60.0f;             // plane distance ahead of the front vertex (mm)
    // RECT half-extents from the axis (mm); a side at MATTE_BOX_OPEN never clips.
    float top    = MATTE_BOX_OPEN;
    float bottom = MATTE_BOX_OPEN;
    float left   = MATTE_BOX_OPEN;
    float right  = MATTE_BOX_OPEN;
    // CIRCLE: centre offset (mm), radius (mm), and aspect (ellipse y-scale). The
    // ray at (mx,my) is blocked when (mx-cx)^2/r^2 + (my-cy)^2/(r*aspect)^2 > 1.
    float cx     = 0.0f;
    float cy     = 0.0f;
    float radius = MATTE_BOX_OPEN;
    float aspect = 1.0f;
};

// Which engine renders the starburst layer.
//   SPRITE — one FFT into a fixed sprite, downsampled and area-resampled onto
//            the sensor. Fast; the sprite is a fixed grid resampled per source,
//            so a moving source can show mild reconstruction softening.
//   MDFT   — matrix-DFT evaluated at the
//            sensor pixel centres in a window around each source, per
//            wavelength (autocorrelation formulation, alias-free). Flux-matched
//            to SPRITE; removes the sub-pixel-motion artefacts. See mdft_render.h.
enum class StarburstEngine { SPRITE = 0, MDFT = 1 };

// HURB (Heisenberg Uncertainty Ray Bending) kick distribution. Each ghost
// ray that survives passing an edge gets a random angular kick of scale
// sigma ~ lambda*K/d (d = perpendicular edge distance); the distribution sets the
// envelope shape. GAUSSIAN is conservative (matches the sinc^2 core); LORENTZIAN
// has heavy 1/theta^2 tails.
enum class HurbKickDistribution : int { GAUSSIAN = 0, LORENTZIAN = 1 };

// Far-field aperture diffraction around each source.
//
// The starburst is the Fraunhofer pattern (squared-magnitude FFT) of the
// effective pupil, sized on the sensor by the calibrated first-order optics —
// pattern pitch = lambda * f_number * pupil_fill — not by an arbitrary knob.
struct DiffractionConfig {
    bool  starburst      = false;   // master switch for the starburst pass
    // SPRITE splats a baked spectral sprite; MDFT evaluates the integral
    // directly at the sensor pixels. See the enum above.
    StarburstEngine starburst_engine = StarburstEngine::SPRITE;
    int   starburst_grid = 1024;    // FFT grid side; power of two (>= 256)
    // Auto-extent ceiling. One FFT period spans grid*pitch; the grid grows when
    // that period does not cover the sensor. Rounded down to a power of two and
    // clamped to [starburst_grid, 16384].
    int   starburst_grid_cap = 8192;
    // Entrance-pupil radius as a fraction of the FFT grid's half-extent. Sets
    // the padding factor Q = 1 / pupil_fill (Q >= 2 required to avoid aliasing
    // the incoherent PSF). Smaller fill = more padding = finer PSF detail but a
    // smaller sensor window; 0.30 (Q ~ 3.3) balances spike reach against detail.
    float pupil_fill     = 0.30f;
    float starburst_gain = 1.0f;    // artistic multiplier on the physical brightness
    // Artistic multiplier on the physical sensor pitch. 1.0 = physically scaled
    // (the true, often sub-pixel-at-preview-resolution size); > 1 enlarges the
    // pattern for stylised renders without changing its structure.
    float scale_trim     = 1.0f;
    int   spectral_samples = 0;     // 0 = inherit RenderConfig::spectral_samples
    bool  use_survivor_mask    = true;  // clip the pupil by the ray-traced survivor set (off-axis cat's-eye)
    bool  use_surface_textures = true;  // fold front-element APERTURE_IMAGE bitmaps into the pupil
    // Side of the CPU-traced cat's-eye envelope grid before FFT upsampling.
    int   survivor_grid  = 192;
    MatteBox matte_box;             // shared with the tracer
    // Front-of-lens baffles; build_gpu_baffles also inserts the matte box.
    std::vector<Baffle> baffles;

    // Physical veiling glare.
    // Additive generalized-Lorentzian glare-spread function
    // g(r) = (a^2 / (r^2 + a^2))^veil_falloff, a = veil_spread * sensor-half —
    // integrated through the starburst's output-space CMF pipeline.
    bool  veil         = false;   // master switch for the veiling-glare pass
    // Veil energy as a fraction of source flux; independent of flare_gain.
    float veil_gain    = 0.03f;
    // GSF core radius as a fraction of sensor half-height.
    float veil_spread  = 0.12f;
    // Radial falloff power p of the generalized Lorentzian g = (a^2/(r^2+a^2))^p.
    // p = 1 integrates only over the bounded sprite; p > 1 is integrable over
    // the plane. Clamped to [1, 3].
    float veil_falloff = 1.5f;

    // HURB edge diffraction.
    // Heisenberg Uncertainty Ray Bending: every ghost ray that survives passing
    // an edge (the stop, a surface rim, a matte-box/baffle plane) gets a random
    // angular kick perpendicular to the nearest edge, of scale sigma ~ lambda*K/d
    // (d = perpendicular edge distance, K in hurb.h). Envelope only, no fringes;
    // energy-conserving (direction only) and chromatic. The disabled path is
    // removed by template specialization.
    bool  hurb = false;
    HurbKickDistribution hurb_kick = HurbKickDistribution::LORENTZIAN;
    // A far edge yields a tiny sigma; below this the kick is skipped (one
    // multiply-compare). Keeps distant edges from costing anything.
    float hurb_min_sigma_rad = 1.0e-5f;
    // Hard clamp on the kick magnitude (radians). Bounds the heavy Lorentzian
    // tail so a near-grazing edge ray can't be flung to a nonsense angle.
    float hurb_max_kick_rad  = 0.35f;
};

// Gate-wall scatter distribution.
//   CAUCHY_ISO — one isotropic Cauchy lobe (groove_aniso / groove_angle_deg are
//                ignored).
//   GROOVED    — (default) Cauchy across the marks, Gaussian along them,
//                rotated by groove_angle_deg.
enum class GateLobe : int { CAUCHY_ISO = 0, GROOVED = 1 };

// Film-gate flare is grazing reflection from the aperture plate, not diffraction.
//
// Model the film-gate opening as walls parallel to the optical axis. For a
// primary ray reaching sensor point p with final slope s = d.x/d.z, the ray
// crosses the +x wall plane (x = Gx) at
// z = -(p.x - Gx)/s. The wall spans z in [-(standoff + thickness), -standoff],
// so the ray scrapes it iff
//
//     standoff * s  <  (p.x - Gx)  <  (standoff + thickness) * s
//
// Reflection flips the wall-normal slope and lands at 2*Gx - p.x. Thickness
// controls the captured pupil band, standoff the propagation distance, and
// roughness the energy-conserving angular spread. The pass is additive and only
// rays landing outside the opening can contribute.
struct GateConfig {
    bool  enabled = false;      // master switch for the gate pass

    // --- Plate geometry -----------------------------------------------------
    // Depth of the cut (mm) — how deep the wall is. Sets the capture band width
    // t/(2N) and with it the layer's energy.
    float thickness_mm = 0.8f;
    // Gap (mm) from the wall's rear edge to the sensor plane. The lever arm:
    // it sets how far into frame the scattered light reaches, inversely to the
    // energy. 0 puts the wall flush against the sensor, which collapses the
    // capture test to its one-sided form and makes the effect a hairline.
    float standoff_mm  = 5.0f;
    // Per-side offsets (mm) on the opening, which otherwise derives from the
    // render's sensor extent (RenderConfig::sensor_half_w/h when set, else the
    // calibrated half-extent — the SAME value the ghost splat maps to, so the
    // wall always sits on the rendered frame edge). Positive opens a side
    // outward, negative crops it inward. All zero (default) = flush.
    float offset_left_mm   = 0.0f;   // -x
    float offset_right_mm  = 0.0f;   // +x
    float offset_bottom_mm = 0.0f;   // -y
    float offset_top_mm    = 0.0f;   // +y

    // --- Wall finish --------------------------------------------------------
    GateLobe lobe = GateLobe::GROOVED;
    // Cauchy HWHM (radians) of the scatter lobe ACROSS the machining marks —
    // the wide axis. Direction only, so it is energy-conserving: it changes
    // where the folded light lands, never how much there is. The on-sensor
    // displacement is (standoff .. standoff+thickness) * tan(kick), so a small
    // standoff makes even a large sigma invisible.
    float roughness_rad = 0.08f;
    // Ratio of along-mark width to roughness_rad; 1 is isotropic.
    float groove_aniso = 0.12f;
    // Orientation of the marks (degrees), measured in each wall's transverse
    // frame from that wall's frame-edge tangent, so all four walls rotate
    // coherently the way a cutter following the opening would leave them.
    // 0 = marks run along the frame edge, so the fan spreads into frame.
    float groove_angle_deg = 0.0f;
    // Hard clamp on one sampled kick (radians), bounding the heavy Cauchy tail
    // so a single draw cannot fling a ray across the frame. Same role as
    // DiffractionConfig::hurb_max_kick_rad.
    float max_kick_rad = 0.35f;
    // Schlick normal-incidence reflectance; the model approaches one at grazing.
    float reflectance_r0 = 0.04f;

    // --- Sampling / brightness ----------------------------------------------
    // Multiplier on traced source flux; independent of flare_gain.
    float gain = 1.0f;
    // Stochastic lobe draws per scraping (ray, wavelength, wall). The scraping
    // subset of the pupil is a thin annulus, so one draw is noisy. Each draw
    // carries 1/n of the weight, so total energy is independent of this;
    // only its smoothness changes.
    int   scatter_samples = 4;
    // Wavelengths integrated; 0 = inherit RenderConfig::spectral_samples. The
    // wall is achromatic, but the primary trace that decides the scrape is
    // dispersive, so the spectral loop is physical rather than decorative: each
    // wavelength crosses the gate edge at its own source position.
    int   spectral_samples = 0;
};

// Shared settings for any renderer that produces ghost flares.
// filter_ghost_pairs() and launch_ghost_render() accept this type directly.
struct FlareConfig : RenderConfig {
    // Override the entrance-pupil sampling mask derived from the is_stop
    // surface.  0 (default) = use the lens-derived shape (polygon if the stop
    // is polygon, circle otherwise).  >= 3 = force a polygon of this blade
    // count, with `aperture_rotation` (degrees) controlling its rotation.
    int   aperture_blades   = 0;
    float aperture_rotation = 0.0f;  // degrees; only meaningful when aperture_blades >= 3

    float flare_gain          = 1000.0f;
    float min_ghost_intensity = 1e-7f;
    bool  ghost_normalize     = true;
    float max_area_boost      = 100.0f;

    // Skip ghost pairs whose rays never reach the (effective) sensor —
    // the dominant off-axis waste (up to ~95% of pairs for a far off-axis source).
    // A coarse GPU pupil probe runs before the main scatter; a pair off-sensor for
    // EVERY source and wavelength is culled. This is disabled in AOV mode and
    // for the three-sample spectral path.
    bool  cull_dead_pairs     = true;

    // Concentrate each (pair, source)'s pupil samples into that pair's
    // probe-measured survivor bounding box instead of the full pupil. Same ray
    // budget laid where the ghost actually reaches the sensor: per-ray weight
    // scales by |R|/A_ref and mask-rejected samples count as zeros, so the
    // estimator remains unbiased while its variance drops roughly with the
    // pupil/bounding-box area ratio (10-100x for
    // off-axis ghosts). A (pair, source) with too little probe evidence
    // (< 3 surviving pupil positions) falls back to full-pupil sampling —
    // concentration never hard-culls. It shares culling's AOV and three-sample
    // guards.
    bool  concentrate_samples = true;

    // Adaptive per-(pair, source) sample budgets. When true, each
    // concentrated (pair, source) draws a sample COUNT scaled to its survivor-rect
    // area — harvesting concentration's over-sampling of tiny survivor regions as
    // render speed — with the per-ray weight rescaled by the same factor so the
    // estimator remains unbiased. Requires concentrate_samples.
    // adaptive_density_boost is the in-rect pupil sample density kept relative
    // to concentration: at boost=b a ghost gets b × (rect's fraction of the
    // pupil) × concentration's density, i.e. its expected on-sensor ray count is
    // b × the full-density path's. adaptive_min_samples is the absolute
    // per-(pair, source) floor so a dim ghost is never starved below visibility.
    // An axis-aligned box is inefficient for a thin, diagonal far-off-axis
    // anamorphic ghost: its survivor set is a thin diagonal sliver inside a
    // near-full-pupil axis-aligned rect, so most budgeted samples may miss it.
    bool  adaptive_sample_budgets = true;
    float adaptive_density_boost  = 8.0f;
    int   adaptive_min_samples    = 1024;

    GhostAovMode aov_mode      = GhostAovMode::NONE;
    int          aov_max_pairs = -1;  // -1 = all active pairs; N >= 0 = first N

    // Optional ghost-pair selection.
    GhostFilter ghost_filter;

    // Aperture-diffraction settings. Default (starburst = false) draws no
    // diffraction pass.
    DiffractionConfig diffraction;

    // Film-gate flare settings. A SIBLING of `diffraction`, not a member of it:
    // the gate is mechanical scatter off the aperture plate, not diffraction.
    // Default (enabled = false) runs no gate pass and emits no gate buffer.
    GateConfig gate;
};

// Single screen-space point source -> ghost buffers.
struct PointFlareConfig : FlareConfig {
    float source_x = 0.5f, source_y = 0.5f;  // [0,1]; out-of-range valid
    float source_r = 1.0f, source_g = 1.0f, source_b = 1.0f;
};

// How the PSF renderer decides each tile's anchor point on the sensor.
enum PSFCenterMode : int {
    PSF_CENTER_CHIEF_CENTROID = 0,  // unweighted 7×7 pupil-mean landing per source
    PSF_CENTER_FIXED_TARGET   = 1,  // aim the chief ray at target_x/y_mm; anchor the tile there
};

// PSF grid renderer — traces many aperture samples from one or more field
// points and accumulates each point's spread into its own tile.  Caller
// supplies the (angle_x, angle_y) field-point list and assigns each one a
// tile_idx in [0, grid_nx * grid_ny).
//
// Two centering modes (center_mode):
//   * CHIEF_CENTROID — each tile is centred on the unweighted mean of
//     its field point's traced pupil cloud, visualising aberration shape.
//   * FIXED_TARGET — each source carries a sensor-plane target (its cell centre);
//     the renderer aims a chief ray at that target and anchors the tile there, so
//     the composite is an upright map of the sensor with the chief dot per cell.
//
// Same kernel covers the diagnostic grid (e.g. 8×8 grid of 64×64 tiles) and
// the single-point high-resolution mode (1×1 grid, large tile, more samples).
struct PSFConfig : RenderConfig {
    int   grid_nx        = 8;       // tiles in X
    int   grid_ny        = 8;       // tiles in Y (1 in single-PSF mode)
    int   tile_w         = 64;      // per-tile pixel width
    int   tile_h         = 64;      // per-tile pixel height
    float tile_extent_mm = 0.05f;   // physical size of one tile on sensor (square)
    bool  monochromatic  = false;   // sum spectral samples into out_r only (g/b zero)
    int   center_mode    = PSF_CENTER_CHIEF_CENTROID;  // tile-anchor strategy (see above)

    // Gaussian per-ray footprint in sensor micrometres. Conversion through the
    // tile's mm-per-pixel keeps the physical footprint resolution-independent;
    // values <= 0 use a single-pixel bilinear splat.
    float splat_sigma_um = 4.0f;

    // Optional polygon-stop override (same semantics as FlareConfig).
    // 0 = use the lens's is_stop surface as-is.  >= 3 forces a polygon of N
    // blades with `aperture_rotation` (degrees) controlling orientation.
    int   aperture_blades   = 0;
    float aperture_rotation = 0.0f;
};
