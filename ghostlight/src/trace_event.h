// ============================================================================
// trace_event.h — TraceStatus, TraceResult, and CPU-side diagnostic types
//
// TraceResult is GPU-safe (no STL) and shared by both CPU and GPU stages.
// TraceEvent and RayPath are CPU-only; they are guarded by #ifndef __CUDACC__
// so this header is safe to include from .cu files.
// ============================================================================
#pragma once

#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#include <cstdint>
#include "vec3.h"

// ---------------------------------------------------------------------------
// TraceStatus — outcome of tracing a ray through the lens.
// GPU-safe: uint8_t, no STL.
// ---------------------------------------------------------------------------
enum class TraceStatus : uint8_t {
    OK             = 0,
    VIGNETTED      = 1,  // clipped by aperture
    TIR            = 2,  // total internal reflection
    MISSED_SURFACE = 3,  // ray missed surface (malformed lens)
    INVALID_INPUT  = 4,
};

// ---------------------------------------------------------------------------
// TraceResult — minimal result shared by CPU and GPU trace paths.
// GPU-safe: no STL, no dynamic allocation.
// ---------------------------------------------------------------------------
struct TraceResult {
    Vec3f       position;  // sensor landing position (mm)
    float       weight;    // cumulative Fresnel x transmittance
    TraceStatus status;
};

// ---------------------------------------------------------------------------
// CPU-only diagnostic types — compiled only when not building device code.
// ---------------------------------------------------------------------------
#ifndef __CUDACC__

#include <vector>

// Per-surface hit record on the CPU diagnostic trace path.
struct TraceEvent {
    int         surface_index;
    Vec3f       hit_point;
    Vec3f       surface_normal;
    float       ior_before;
    float       ior_after;
    float       fresnel_weight;  // transmittance, or reflectance at a bounce surface
    TraceStatus status;
    bool        reflected;       // true = this surface was a ghost bounce
};

// Full CPU-side ray path — one event per surface the ray reached.
struct RayPath {
    std::vector<TraceEvent> events;
    TraceResult             result;
};

#endif // !__CUDACC__
