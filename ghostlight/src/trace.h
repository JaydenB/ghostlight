// ============================================================================
// trace.h — CPU ray tracing through a sequential lens system
// ============================================================================
#pragma once

#include "trace_core.h"   // Ray, Surface, OpticalSystem + shared __host__ __device__ math
#include "trace_event.h"  // TraceStatus, TraceResult, TraceEvent, RayPath

// Trace a ghost ray through the complete lens system.
//
// The ray enters the front of the lens and transmits through all surfaces
// except at bounce_a and bounce_b (where it reflects).  The path is:
//
//   Outbound leg: forward through surfaces 0..bounce_b (reflect at bounce_b)
//   Return leg: backward through surfaces bounce_b-1..bounce_a (reflect at bounce_a)
//   Sensor leg: forward through surfaces bounce_a+1..N-1 to the sensor
//
// ray.lambda is used for dispersion (wavelength-dependent IOR) and coating
// reflectance calculations.
//
// Invalid bounce indices return TraceStatus::INVALID_INPUT.
TraceResult trace_ghost_ray(const Ray& ray, const OpticalSystem& lens,
                            int bounce_a, int bounce_b);

// Diagnostic overload — records per-surface events in path_out.
// path_out.events is cleared and repopulated; path_out.result mirrors the return value.
TraceResult trace_ghost_ray(const Ray& ray, const OpticalSystem& lens,
                            int bounce_a, int bounce_b,
                            RayPath& path_out);

// Trace a primary ray through the complete lens system — pure forward refraction,
// no ghost bounces. Structurally identical to the sensor leg of trace_ghost_ray()
// but starting from surface 0.  ray.lambda is used for dispersion.
TraceResult trace_primary_ray(const Ray& ray, const OpticalSystem& lens);

// Diagnostic overload — records per-surface events in path_out.
// path_out.events is cleared and repopulated; path_out.result mirrors the return value.
TraceResult trace_primary_ray(const Ray& ray, const OpticalSystem& lens,
                              RayPath& path_out);
