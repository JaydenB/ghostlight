// ============================================================================
// PointFlareRenderer.h — Single screen-space point source flare renderer
// ============================================================================
#pragma once

#include "flare_buffers.h"
#include "optical_system.h"
#include "lens_calibration.h"
#include "render_config.h"

// Render ghost flares from a single screen-space point source.
//
// Returns false on CUDA error; out is unchanged on failure.
bool render_point_flare(int width, int height,
                        const OpticalSystem&       lens,
                        const LensCalibration&  calib,
                        const PointFlareConfig& cfg,
                        FlareBuffers&           out);
