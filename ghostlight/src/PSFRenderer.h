// ============================================================================
// PSFRenderer.h — Geometric point-spread function grid renderer.
//
// Traces many entrance-pupil samples from each of N field points, accumulates
// each point's spread (relative to its chief-ray hit) into one tile of a
// composite buffer.  Same kernel covers the diagnostic PSF grid and the
// single-point high-resolution mode (grid_nx=grid_ny=1, larger tile).
//
// Each source maps to one tile, indexed row-major: source i goes to tile at
// (gx = i % grid_nx, gy = i / grid_nx).  Source count must satisfy
// n_sources <= grid_nx * grid_ny.
// ============================================================================
#pragma once

#include "optical_system.h"
#include "render_config.h"

#include <cstdint>
#include <vector>

// Per-cell aiming outcome; diagnostic only and not fed back into rendering.
enum PSFCellStatus : uint8_t {
    PSF_CELL_OK              = 0,  // chief ray reached the cell centre
    PSF_CELL_CHIEF_VIGNETTED = 1,  // chief blocked but part of the pupil transmits
    PSF_CELL_DARK            = 2,  // no transmission near the cell
};

// One source for the PSF renderer — a field-point direction plus weight.
//
// In FIXED_TARGET mode angle_x/angle_y are the AIM SEED direction and
// target_x_mm/target_y_mm is the sensor-plane cell centre the chief ray is aimed
// at.  In CHIEF_CENTROID mode the target fields are ignored.
struct PSFSource
{
    float angle_x;                // radians off-axis (horizontal); aim seed in target mode
    float angle_y;                // radians off-axis (vertical)
    float r, g, b;                // spectral weight (typically 1,1,1 for white)
    float target_x_mm = 0.0f;     // FIXED_TARGET: cell-centre sensor position to aim at
    float target_y_mm = 0.0f;
};

struct PSFOutput
{
    std::vector<float> out_r, out_g, out_b;  // planar floats, composite_w * composite_h
    int composite_w = 0;                     // = grid_nx * tile_w
    int composite_h = 0;                     // = grid_ny * tile_h
    int tile_w      = 0;
    int tile_h      = 0;
    int grid_nx     = 0;
    int grid_ny     = 0;
    // Per-tile sensor extent (mm), applied to BOTH axes. Pixels stay square, so a
    // non-square tile (tile_w != tile_h) has visible y-extent this * tile_h/tile_w.
    float tile_extent_mm = 0.0f;             // mm extent of one tile on sensor

    // Per-source chief-ray sensor landing (mm), aligned with the source list.
    // FIXED_TARGET mode: the aimed landing (≈ the cell target).  CHIEF_CENTROID
    // mode: the pupil-mean landing, or (0, 0) when every pupil sample vignettes.
    std::vector<float> chief_x_mm, chief_y_mm;

    // Per-source diagnostic report, not used by the render. In CHIEF_CENTROID mode status
    // is OK for any source with at least one surviving pupil sample (else DARK),
    // pupil_fraction is the probe transmission, and aim_residual_mm is 0.
    std::vector<uint8_t> status;          // PSFCellStatus
    std::vector<float>   pupil_fraction;  // transmitting fraction of a unit-disk probe [0, 1]
    std::vector<float>   aim_residual_mm; // |aimed landing − target| (FIXED_TARGET); 0 otherwise
};

// Render a PSF grid.
//
// `sources` may contain fewer entries than grid_nx * grid_ny — unused tiles
// of the composite stay zero.  Passing more than grid_nx * grid_ny throws.
//
// The renderer takes no LensCalibration: it traces rays straight through the
// lens via d_trace_primary_ray, computing everything it needs per source (no
// precomputed field or distortion tables).
//
// Returns false on invalid input or render failure; `out` is left unchanged.
bool render_psf(const OpticalSystem&         lens,
                const std::vector<PSFSource>& sources,
                const PSFConfig&             cfg,
                PSFOutput&                   out);
