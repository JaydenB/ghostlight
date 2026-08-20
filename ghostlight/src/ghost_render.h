// ============================================================================
// ghost_render.h — GPU ghost scatter kernel launcher
//
// Declares launch_ghost_render() + render_ghost_pipeline().  Scatter-accumulate
// side only; ray-trace math lives in trace_cuda.h.  The persistent device-memory
// pool (GpuBufferCache) is internal to ghost_render.cu.
// ============================================================================
#pragma once

#include "aperture_textures.h"
#include "coating_upload.h"
#include "flare_buffers.h"
#include "ghost.h"
#include "optical_system.h"
#include "lens_calibration.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// GpuBufferCache — the process-wide persistent device-memory pool — is an
// internal implementation detail of ghost_render.cu (a mutex-guarded leaked
// singleton, grown but never shrunk). Callers never construct or see it; it is
// defined in ghost_render.cu, not in this header.

// Run the full shared flare pipeline for a pre-built source list.
//
// Both flare renderers call this after building their source list. Stages
// performed, in order:
//   1. filter_ghost_pairs — discard pairs below intensity threshold.
//   2. If no active pairs: return true (ghost buffers remain zero).
//   3. launch_ghost_render — GPU scatter into ghost_r/g/b.
//
// ghost_r/g/b: caller-zeroed; receives rendered scatter.
// GPU scratch is the internal persistent singleton (see GpuBufferCache).
// Returns false and populates out_error on CUDA error; output buffers
// are then in an indeterminate state (caller should not move them to output).
bool render_ghost_pipeline(
    const OpticalSystem&               lens,
    const LensCalibration&          calib,
    const std::vector<FlareSource>& sources,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    float*                          ghost_r,
    float*                          ghost_g,
    float*                          ghost_b,
    const FlareConfig&              cfg,
    std::string*                    out_error = nullptr,
    std::vector<FlareAovLayer>*     aov_out   = nullptr,
    GhostRenderStats*               out_stats = nullptr);

// Launch the CUDA ghost rendering kernel.
//
// active_pairs / pair_area_boosts must already be pre-filtered (below-threshold
// pairs removed, area-normalisation boosts computed) — see filter_ghost_pairs()
// in ghost.cpp.
//
// sensor_half_w/h: half-dimensions of the sensor in mm.
//
// out_r/g/b: CPU-side output buffers, width×height, must be zeroed by caller.
//
// fmt_w/h, fmt_x0_in_buf/fmt_y0_in_buf: the format window inside the
// output BUFFER. The optics are calibrated to the format (fmt_w × fmt_h); the
// render can target a larger buffer (width × height) by placing the format origin
// at (fmt_x0_in_buf, fmt_y0_in_buf). A sensor position maps to a buffer pixel as
//   px = (pos.x / (2·sensor_half_w) + 0.5)·fmt_w + fmt_x0_in_buf   (y likewise).
// fmt_w/h may differ from the output buffer and the origin may be non-zero.
//
// out_error: if non-null and a CUDA error occurs, receives an error message.
void launch_ghost_render(
    const OpticalSystem&               lens,
    const std::vector<GhostPair>&   active_pairs,
    const std::vector<float>&       pair_area_boosts,
    const std::vector<FlareSource>& sources,
    float                           sensor_half_w,
    float                           sensor_half_h,
    float*                          out_r,
    float*                          out_g,
    float*                          out_b,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    const FlareConfig&              config,
    std::string*                    out_error = nullptr,
    GhostRenderStats*               out_stats = nullptr
);
