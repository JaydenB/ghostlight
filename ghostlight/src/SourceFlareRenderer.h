// ============================================================================
// SourceFlareRenderer.h — Extended (area) source ghost renderer
// ============================================================================
#pragma once

#include "flare_buffers.h"
#include "optical_system.h"
#include "lens_calibration.h"
#include "render_config.h"

// Render ghost flares from an extended source sampled as N
// angular offsets around a screen-space center.
//
// The center direction comes from cfg.source_x/source_y exactly as in
// render_point_flare.  Each offset row [d_angle_x, d_angle_y, weight]
// contributes one collimated source at (center_angle + d_angle), colored
// cfg.source_r/g/b * weight.  Quadrature weights are the caller's contract:
// pass weights summing to 1 to average an area source, or a subset of those
// weights to accumulate a partial (progressive) render.
//
// offsets: (n_offsets, 3) row-major floats. n_offsets must be >= 1.
// Returns false on invalid input or render failure; out is unchanged.
bool render_source_flare(int width, int height,
                         const float*            offsets,
                         int                     n_offsets,
                         const OpticalSystem&    lens,
                         const LensCalibration&  calib,
                         const PointFlareConfig& cfg,
                         FlareBuffers&           out);
