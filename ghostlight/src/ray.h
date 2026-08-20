// ============================================================================
// ray.h — unified Ray struct for CPU and GPU
// ============================================================================
#pragma once

// Strip CUDA qualifiers when building with a regular C++ compiler so this
// header is safe to include from both .cpp and .cu translation units.
#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#include "vec3.h"

struct Ray
{
    Vec3f origin;
    Vec3f dir;    // must be normalized
    float lambda = 587.56f; // wavelength in nm; default = d-line (glass n_d reference)
};
