// ============================================================================
// veil_render.h — Physical veiling-glare pass.
//
// Renders the broad, soft, spectral halo a real lens throws around every bright
// source (surface-reflection / stray-light "veiling glare") as a separate
// additive layer splatted onto each source.
//
// The halo is an energy-conserving glare-spread function (GSF): a radial
// generalized-Lorentzian kernel g(r) = (a^2 / (r^2 + a^2))^p (a = core radius on
// the sensor, p = veil_falloff). The halo is ACHROMATIC — surface-reflection
// veiling glare is broadband, unlike aperture diffraction's rainbow — and is
// tinted by the SOURCE's own colour at splat time, so a white source yields a
// neutral veil and a red source a red veil. No FFT — the kernel is analytic; it
// reuses the starburst's downsample / area-resample-splat plumbing (cloned here).
//
// A standalone additive layer: the GSF splat atomicAdds into the veil_r/g/b
// buffer, independent of the comp and the ghost path — exactly as ghost scatter
// and starburst splat write their own buffers.
// ============================================================================
#pragma once

#include "ghost.h"            // FlareSource
#include "lens_calibration.h"
#include "optical_system.h"
#include "render_config.h"

#include <string>
#include <vector>

// Optional capture of the normalized GSF, RGB sprite, and physical scale.
struct VeilDebug {
    int   grid = 0;                 // sprite grid side N
    std::vector<float> reference;   // N*N, DC-centred, energy-normalised (sum = 1) mono GSF
    std::vector<float> sprite_rgb;  // N*N*3 interleaved RGB veil sprite
    float dx_mm_x = 0.0f, dx_mm_y = 0.0f;  // sensor pitch of one sprite texel (mm)
    float core_mm  = 0.0f;          // GSF core radius a on the sensor (mm)
    float falloff  = 0.0f;          // radial power p actually used
    float core_texels = 0.0f;       // core radius in reference texels
};

// Render the veiling-glare layer for a pre-built source list.
//
// A no-op returning true when cfg.diffraction.veil is false or the source list
// is empty (out_* left as the caller zeroed them).
//
// src_px / src_py: each source's sensor position in BUFFER pixel coordinates
//   (the same mapping as source placement), length
//   sources.size(). The GSF is centred there.
// out_r/g/b: caller-zeroed width*height buffers; the veil is ADDED.
// sensor_half_w/h and fmt_* follow the ghost pipeline's conventions.
// dbg: optional; see VeilDebug.
//
// Returns false and sets *out_error on CUDA failure.
bool render_veil(const OpticalSystem&            lens,
                 const LensCalibration&          calib,
                 const std::vector<FlareSource>& sources,
                 const std::vector<float>&       src_px,
                 const std::vector<float>&       src_py,
                 int                             width,
                 int                             height,
                 int                             fmt_w,
                 int                             fmt_h,
                 float                           sensor_half_w,
                 float                           sensor_half_h,
                 const FlareConfig&              cfg,
                 float*                          out_r,
                 float*                          out_g,
                 float*                          out_b,
                 std::string*                    out_error = nullptr,
                 VeilDebug*                      dbg       = nullptr);
