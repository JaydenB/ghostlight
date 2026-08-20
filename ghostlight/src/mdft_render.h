// ============================================================================
// mdft_render.h — Matrix-DFT starburst engine (exact, resample-free).
//
// An alternate to the sprite path in starburst_render.cu.  Instead of one FFT
// into a fixed sprite that is then downsampled and splatted (which resamples a
// fixed grid onto the sensor and so aliases as the source moves sub-pixel), the
// MDFT engine evaluates the diffraction integral DIRECTLY at the sensor pixel
// centres in a window around each source, per wavelength.
//
// It works in the INTENSITY / autocorrelation domain, which makes the per-pixel
// box integral analytic (a sinc factor) and therefore alias-free by
// construction:
//
//     I(x)      = |E(x)|^2 = sum_m C_m e^(-i2pi m.x / S),   C = autocorr(A)
//     box int   = px * sum_m C_m sinc(m.px/S) e^(-i2pi m.x_k / S)      (separable)
//
// C (the pupil autocorrelation, compact support ~2*fill*N) is built once per
// render from the SAME effective pupil A the sprite engine builds, then each
// (source x wavelength) is a small separable matrix triple product evaluated at
// the K x K pixel window.  No sprite, no downsample, no scatter — nothing to
// alias. Its normalization matches the sprite engine.
// ============================================================================
#pragma once

#include <string>
#include <vector>

struct GPUSpectralSample;   // spectral.h

// One source in BUFFER-pixel coordinates + its (already output-space) colour.
struct MdftSource {
    float px, py;   // sensor position, buffer pixels (sub-pixel, matches the sprite)
    float r, g, b;  // source colour (all sources of one render share hue; weight in it)
};

// Render the MDFT starburst layer for a pre-built source list into caller-owned
// host buffers (OVERWRITTEN, matching the sprite path's device->host copy).
//
//   d_pupil_amp : N*N DEVICE complex (float2) effective pupil AMPLITUDE A (.x =
//                 amplitude, .y == 0).  Consumed into the engine's own scratch, so
//                 the caller's buffer is preserved.  The engine builds one
//                 autocorrelation C from it, reused across colours.
//   dx_x/dx_y   : sensor pitch of one pattern texel (mm) = scale_trim * lambda_ref
//                 * f_number * pupil_fill, per axis (the sprite's dbg dx_mm_*).
//   spec        : the SAME spectral table the sprite bakes with (d_lambda-
//                 premultiplied + per-channel white-balanced), on host.
//   gain        : starburst_gain * throughput * aperture_scale (the sprite gain).
//
// Returns false and sets *err on CUDA / FFT failure.
bool mdft_starburst(const void*                          d_pupil_amp,
                    int                                  N,
                    float                                pupil_fill,
                    const std::vector<GPUSpectralSample>& spec,
                    float                                lambda_ref_nm,
                    float                                dx_x,
                    float                                dx_y,
                    float                                px_mm_x,
                    float                                px_mm_y,
                    const std::vector<MdftSource>&       sources,
                    float                                gain,
                    float*                               out_r,
                    float*                               out_g,
                    float*                               out_b,
                    int                                  width,
                    int                                  height,
                    std::string*                         err);
