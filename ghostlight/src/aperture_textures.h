// ============================================================================
// aperture_textures.h — Lens APERTURE_IMAGE bitmap → cudaTextureObject upload.
//
// Shared by ghost_render.cu and psf_render.cu so both renderers' GPU traces
// can apply image-aperture vignetting via d_image_aperture_passes().
//
// CUDA-runtime types are hidden behind void*/uint64_t so this header is safe
// to include from .cpp files (no <cuda_runtime.h> dependency).
// ============================================================================
#pragma once

#include "optical_system.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// One per-render bundle of uploaded aperture textures.
//
// Lifetime: construct, call upload_aperture_textures(), pass `d_textures` to
// a kernel, then let the destructor free everything (destroys texture objects
// before the backing cudaArrays — order is enforced internally).
struct ApertureTexturePack
{
    // Per-surface texture handle (cudaTextureObject_t, 0 if none). Empty when
    // the lens has no image apertures.
    std::vector<std::uint64_t> textures;
    // Per-surface backing storage (cudaArray_t).  Parallel to `textures`.
    std::vector<std::uint64_t> arrays;
    // Device-side copy of `textures` indexed by surface.  nullptr when the
    // upload had nothing to do — kernel must treat as "no image apertures".
    void*       d_textures       = nullptr;

    // Destroys texture objects, frees backing arrays, frees device table.
    // Safe to call multiple times and on a partially-constructed pack.
    void release();

    ~ApertureTexturePack() { release(); }

    ApertureTexturePack() = default;
    ApertureTexturePack(const ApertureTexturePack&)            = delete;
    ApertureTexturePack& operator=(const ApertureTexturePack&) = delete;
};

// Upload all APERTURE_IMAGE surfaces' bitmaps as cudaTextureObjects.
//
// Lens surfaces with aperture_shape != APERTURE_IMAGE or whose ApertureImage
// has no pixels are skipped (their entry in `textures` stays 0; the device
// kernel's d_image_aperture_passes() treats 0-handles as pass-through).
//
// Returns false and populates `out_error` on CUDA failure; `pack` is fully
// released in that case.
bool upload_aperture_textures(const OpticalSystem& lens,
                              ApertureTexturePack& pack,
                              std::string*         out_error);

// Bake + upload the signed-distance field for the image-aperture STOP surface
// (find_sdf_target_surface), as one float4 texture [d_mm, nx, ny, 0] used by the
// HURB edge kick in the ghost trace. Only that one surface's slot is populated;
// all others stay handle 0 (no kick). No eligible surface -> released pack with
// nullptr d_textures (kernel treats every surface as pass-through). Reuses
// ApertureTexturePack (handles are format-agnostic); the pack is float4, so it
// must NEVER be sampled with tex2D<float> — keep it a distinct pack from the
// binary matte textures.
//
// Returns false and populates `out_error` on CUDA failure; `pack` is released.
bool upload_aperture_sdf_textures(const OpticalSystem& lens,
                                  ApertureTexturePack& pack,
                                  std::string*         out_error);
