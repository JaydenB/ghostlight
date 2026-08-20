// ============================================================================
// source_map.h — Screen position <-> field angle, solved by tracing.
//
// The inverse map traces the area-weighted survivor mean of the renderer's
// spawn disc, so distortion, anamorphic squeeze, and pupil walk are included.
// Beyond the image circle it uses a bounded C1 arctangent continuation from
// the last convergent angle. A constant-focal-length map is the final fallback.
// ============================================================================
#pragma once

#include "lens_calibration.h"
#include "optical_system.h"

enum class SourceMapStatus {
    SOLVED    = 0,   // traced: the landing meets the target
    CONTINUED = 1,   // past the last convergent angle: C1 arctangent extension
    FALLBACK  = 2,   // no traced landing; constant-focal-length approximation
};

// A solved source position and its local forward map for nearby angles.
struct SourceMapSolve {
    float           angle_x  = 0.0f;   // radians — the answer
    float           angle_y  = 0.0f;
    SourceMapStatus status   = SourceMapStatus::FALLBACK;
    float           residual_mm = 0.0f;   // |landing - target| at the solve

    // Anchor of the forward map: an angle whose landing was actually measured.
    // For SOLVED this is the solved angle itself, so the forward map reproduces
    // the target exactly at the source. For CONTINUED it is the last angle that
    // converged, which is where the arctangent extension is C1-joined.
    float anchor_ax = 0.0f, anchor_ay = 0.0f;
    float anchor_x  = 0.0f, anchor_y  = 0.0f;   // its landing, mm

    // d(landing_mm)/d(angle_rad) at the anchor, row-major:
    //   [ dx/dax  dx/day ]
    //   [ dy/dax  dy/day ]
    // Measured once per solve and reused for extended-source offsets.
    float J[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // The frame the screen fraction is expressed in (mm half-extents).
    float eff_half_w = 0.0f, eff_half_h = 0.0f;

    // FALLBACK only: scale for the constant-focal-length approximation.
    float inv_tan_w = 0.0f, inv_tan_h = 0.0f;

    int probe_evals = 0;   // diagnostics; also what the cost gate measures
};

// Solve the source position (screen NDC, -1..1 with 0 at the optical axis) to a
// field angle, by tracing. eff_half_w/h are the rendered frame's half-extents
// in mm (the sensor override when there is one, else calib.sensor_half_*).
//
// Uses a one-entry memo keyed by surface bytes, calibration, and request.
SourceMapSolve solve_source_angle(const OpticalSystem&   lens,
                                  const LensCalibration& calib,
                                  float ndc_x, float ndc_y,
                                  float eff_half_w, float eff_half_h);

// Where a source at (angle_x, angle_y) appears, as a [0,1] fraction of the
// rendered frame per axis (0.5 = optical axis). This is the map's forward
// direction and the thing every splatted layer needs.
//
// Exact at the anchor; nearby offsets use the stored Jacobian and continuation
// envelope without tracing.
void source_map_screen(const SourceMapSolve& base,
                       float angle_x, float angle_y,
                       float* out_sx, float* out_sy);

// Area-weighted survivor-mean landing of the spawn disc at an arbitrary angle.
// Returns false when no ray survives.
bool source_map_landing(const OpticalSystem&   lens,
                        const LensCalibration& calib,
                        float angle_x, float angle_y,
                        float* out_x_mm, float* out_y_mm);
