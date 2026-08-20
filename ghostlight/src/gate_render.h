// ============================================================================
// gate_render.h — Film-gate scatter pass.
//
// Renders gate_r/g/b as a standalone additive layer: the aperture plate's cut
// edge reflecting light that would land just outside the opening back into
// frame. The geometry and every knob are documented on GateConfig
// (render_config.h); the shared band test / fold / lobe live in gate.h.
//
// The gate computes traced landing points rather than splatting at source
// positions, so it requires the complete sensor-mm to buffer-pixel mapping.
// ============================================================================
#pragma once

#include "flare_buffers.h"
#include "gate.h"
#include "ghost.h"            // FlareSource
#include "lens_calibration.h"
#include "optical_system.h"
#include "render_config.h"

#include <string>
#include <vector>

// Optional validation capture. Off the production path entirely: the scrape
// counters live behind a device-side allocation that is only made when a
// GateDebug is supplied, and the direct-image buffers are only filled then too.
struct GateDebug {
    GpuGate gate;                   // the resolved runtime params
    long long traces  = 0;          // primary traces attempted (src x pupil x lambda)
    long long scrapes = 0;          // (ray, wavelength, wall) band hits
    double    energy  = 0.0;        // sum over the deposited layer, all channels
    float     reach_mm = 0.0f;      // largest realised |fold - edge| (specular)
    float     band_x_mm = 0.0f;     // predicted t / (2 * f_number_x)
    float     band_y_mm = 0.0f;     // predicted t / (2 * f_number_y)

    // Direct-source reference from the same traces, used for radiometric
    // comparison. RGB planar width*height buffers; empty unless captured.
    std::vector<float> direct_r, direct_g, direct_b;
};

// Render the film-gate scatter layer for a pre-built source list.
//
// A no-op returning true when cfg.gate.enabled is false or the source list is
// empty (out_* left as the caller zeroed them).
//
// sensor_half_w/h: the render's EFFECTIVE half-extent — the same value the ghost
//   splat maps to (cfg override when set, else calibrated). The gate opening
//   derives from it, so passing the raw calibrated value under a sensor override
//   would stand the wall off the rendered frame edge.
// out_r/g/b: caller-zeroed width*height buffers; the gate is ADDED.
//
// Returns false and sets *out_error on CUDA failure.
bool render_gate(const OpticalSystem&            lens,
                 const LensCalibration&          calib,
                 const std::vector<FlareSource>& sources,
                 int                             width,
                 int                             height,
                 int                             fmt_w,
                 int                             fmt_h,
                 int                             fmt_x0_in_buf,
                 int                             fmt_y0_in_buf,
                 float                           sensor_half_w,
                 float                           sensor_half_h,
                 const FlareConfig&              cfg,
                 float*                          out_r,
                 float*                          out_g,
                 float*                          out_b,
                 std::string*                    out_error = nullptr,
                 GateDebug*                      dbg       = nullptr);
