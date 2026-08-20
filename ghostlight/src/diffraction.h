// ============================================================================
// diffraction.h — Aperture-diffraction starburst pass.
//
// Renders the far-field (Fraunhofer) diffraction pattern of the effective
// pupil as an additive layer splatted onto each bright source.  The pattern is
// the squared-magnitude FFT of the pupil amplitude, sized on the sensor by the
// calibrated first-order optics (pitch = lambda * f_number * pupil_fill), and
// integrated over wavelength through the same CMF pipeline the other renderers
// use.  Scatter-accumulate only; the FFT lives behind fft_backend.h.
// ============================================================================
#pragma once

#include "ghost.h"            // FlareSource
#include "lens_calibration.h"
#include "optical_system.h"
#include "render_config.h"

#include <string>
#include <vector>

// Optional capture of the intermediate pupil, PSF, RGB sprite, and scale.
struct StarburstDebug {
    int   grid = 0;                 // FFT grid side N
    std::vector<float> psf;         // N*N, DC-centred, energy-normalised (sum = 1)
    std::vector<float> sprite_rgb;  // N*N*3 interleaved RGB starburst sprite
    std::vector<float> pupil;       // N*N effective pupil amplitude A(u,v) in [0,1]
                                    // (stop x cat's-eye x matte box x front texture)
    float dx_mm_x = 0.0f, dx_mm_y = 0.0f;  // sensor pitch of one sprite texel (mm)
    float f_number_x = 0.0f, f_number_y = 0.0f;
    float focal_length_x = 0.0f, focal_length_y = 0.0f;
    float entrance_pupil_semi_x = 0.0f, entrance_pupil_semi_y = 0.0f;
    float lambda_ref_nm = 0.0f;
    // Representative field angle (rad) the effective pupil was built at, and the
    // pupil-throughput T (surviving pupil energy / on-axis reference) that dims
    // the starburst off-axis / behind a matte box.  T = 1 with no modifiers.
    float field_angle_x = 0.0f, field_angle_y = 0.0f;
    float throughput = 1.0f;
    float chief_offset_x = 0.0f, chief_offset_y = 0.0f;  // pupil-centring launch offset (mm)
};

// Render the starburst layer for a pre-built source list.
//
// A no-op returning true when cfg.diffraction.starburst is false or the source
// list is empty (out_* left as the caller zeroed them).
//
// src_px / src_py: each source's sensor position in BUFFER pixel coordinates
//   (the same mapping as source placement), length
//   sources.size().  The sprite is centred there.
// out_r/g/b: caller-zeroed width*height buffers; the starburst is ADDED.
// sensor_half_w/h and fmt_* follow the ghost pipeline's conventions.
// dbg: optional; see StarburstDebug.
//
// Returns false and sets *out_error on CUDA / FFT failure.
bool render_starburst(const OpticalSystem&            lens,
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
                      StarburstDebug*                 dbg       = nullptr);
