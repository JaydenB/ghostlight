// ============================================================================
// coating_upload.h — device-side coating table arena + Surface pointer patch
//
// Table-backed coating models (SPECTRAL / ANGULAR / SPECTRAL_ANGULAR) store
// raw pointers on the Surface POD that point into host memory owned by
// OpticalSystem::coating_tables.  Surfaces are uploaded to the GPU with a
// verbatim cudaMemcpy, so those pointers must be rewritten to device
// addresses first.  upload_coating_tables() packs every surface's tables
// into a single device blob and produces a patched host copy of the surface
// array whose coating pointers reference the blob.
//
// Mirrors the aperture_textures.h lifecycle: callers keep a
// CoatingDeviceArena alive across frames (the ghost GpuBufferCache) or use a
// local one released per call (the PSF renderer).
// ============================================================================
#pragma once

#include "optical_system.h"

#include <cstddef>
#include <string>
#include <vector>

struct CoatingDeviceArena
{
    void*       d_blob     = nullptr; // one packed device allocation
    std::size_t blob_bytes = 0;       // current capacity (grows monotonically)

    // Free the device allocation and reset capacity.  Safe to call twice.
    void release();

    CoatingDeviceArena() = default;
    ~CoatingDeviceArena() { release(); }
    CoatingDeviceArena(const CoatingDeviceArena&)            = delete;
    CoatingDeviceArena& operator=(const CoatingDeviceArena&) = delete;
};

// Pack all table-backed coating data for `lens` into `arena` (growing its
// blob only when needed), upload it, and fill `patched` with a copy of
// lens.surfaces whose coating pointer fields reference device addresses.
//
// When no surface carries table data, `patched` is left EMPTY and nothing is
// uploaded — callers then memcpy lens.surfaces.data() directly, so the
// common table-free case costs nothing.
//
// Returns false on CUDA error (out_error receives a message when non-null).
bool upload_coating_tables(const OpticalSystem& lens,
                           CoatingDeviceArena&  arena,
                           std::vector<Surface>& patched,
                           std::string*          out_error);
