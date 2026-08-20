// ============================================================================
// psf_render.h — GPU PSF kernel launcher.
//
// CPU-side helper called by PSFRenderer.cpp after the chief-ray pre-pass.
// Internal GPU POD types are hidden inside psf_render.cu — this header keeps
// the host/device boundary clean (no <cuda_runtime.h> required).
// ============================================================================
#pragma once

#include "optical_system.h"
#include "render_config.h"

#include <string>
#include <vector>

// A field-point source already paired with its chief-ray landing position
// (computed by the CPU pre-pass) so the kernel can centre the tile on it.
struct PSFGpuSource
{
    float angle_x, angle_y;       // ray-spawn direction (radians)
    float r, g, b;                // source spectral weight
    float chief_x_mm, chief_y_mm; // chief-ray hit on sensor
    int   tile_x0, tile_y0;       // pixel origin of this source's tile in composite
};

// Launch the GPU PSF kernel.
//
// out_r/g/b: caller-zeroed composite buffers of size composite_w * composite_h.
// Returns on error via out_error (left empty on success).
void launch_psf_render(const OpticalSystem&             lens,
                       const std::vector<PSFGpuSource>& gpu_sources,
                       int   tile_w,
                       int   tile_h,
                       int   composite_w,
                       int   composite_h,
                       float mm_per_pixel,
                       bool  monochromatic,
                       float* out_r,
                       float* out_g,
                       float* out_b,
                       const PSFConfig& config,
                       std::string*     out_error);
