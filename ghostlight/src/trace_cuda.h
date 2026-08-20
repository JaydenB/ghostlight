// ============================================================================
// trace_cuda.h — GPU device-only ghost and primary ray tracing
//
// d_trace_ghost_ray is the GPU mirror of trace_ghost_ray in trace.cpp.
// It is defined __device__ __forceinline__ here so ghost_render.cu can
// include this header and call it without requiring separable compilation.
//
// All geometry and optics calls delegate to trace_core.h (__host__ __device__
// shared math) — no duplicated device-side math in this file.
//
// Include only from .cu files.  The CUDA qualifiers strip out silently when
// compiled by a host compiler but the body would reference device-only
// intrinsics (fabsf used via trace_core.h works fine; __device__ keyword
// is stripped when !__CUDACC__).
// ============================================================================
#pragma once

#include "trace_core.h"   // intersect_surface, refract_ray, reflect_ray, etc.
#include "trace_event.h"  // TraceStatus, TraceResult (GPU-safe)
#include "hurb.h"         // GpuHurb, aperture edge kick

#include <cuda_runtime.h> // cudaTextureObject_t, tex2D

// ---------------------------------------------------------------------------
// d_image_aperture_passes
//
// Bitmap-driven aperture test on the GPU side.  Mirrors sample_aperture_image()
// in trace.cpp.  `aperture_textures` may be nullptr (no images on this lens)
// in which case the function returns true for all surfaces; the per-surface
// handle is 0 for surfaces with no uploaded texture (image not loaded or not
// an image-aperture surface).
//
// Uses cudaFilterModeLinear bilinear filtering on the device — for tests
// against pixel boundaries (the CPU path is nearest-neighbour) results may
// differ; well away from boundaries the two paths agree.
// ---------------------------------------------------------------------------
__device__ __forceinline__
bool d_image_aperture_passes(const Surface& s, const Vec3f& hit,
                             const cudaTextureObject_t* aperture_textures,
                             int surface_index)
{
    if (s.aperture_shape != APERTURE_IMAGE)            return true;
    if (aperture_textures == nullptr)                  return true;
    cudaTextureObject_t tex = aperture_textures[surface_index];
    if (tex == 0)                                      return true;
    if (!(s.aperture_aspect > 0.0f)
        || !(s.aperture_semi_diameter > 0.0f))         return false;

    const Vec3f local = world_to_surface_point(hit, s);
    float hx = local.x / s.aperture_aspect;
    float u  = 0.5f + hx    / (2.0f * s.aperture_semi_diameter);
    float v  = 0.5f + local.y / (2.0f * s.aperture_semi_diameter);
    return tex2D<float>(tex, u, v) > 0.5f;
}

// ---------------------------------------------------------------------------
// d_aperture_edge_distance
//
// Device wrapper over aperture_edge_distance() (trace_core.h) that adds SDF
// support for image apertures — the analytic function has no texture access and
// returns a 1e30 sentinel for APERTURE_IMAGE. For an image surface with a baked
// SDF (only the stop, see upload_aperture_sdf_textures), one tex2D<float4> tap
// yields [signed_d_mm, nx, ny, 0]: the perpendicular edge distance (mm, clamped
// >=0 — bilinear can dip slightly negative at the boundary) and the transverse
// edge normal (world frame; sign is irrelevant to hurb_apply_kick). Image
// surfaces with no SDF (nullptr table / 0 handle — e.g. a non-stop matte) keep
// the sentinel and never kick. All other shapes defer to the analytic path.
//
// UV mapping matches d_image_aperture_passes exactly so SDF distance 0 coincides
// with the matte clip boundary.
// ---------------------------------------------------------------------------
__device__ __forceinline__
float d_aperture_edge_distance(const Vec3f& hit, const Surface& s,
                               int surface_index,
                               const cudaTextureObject_t* sdf_textures,
                               Vec3f& edge_normal_out)
{
    if (s.aperture_shape == APERTURE_IMAGE)
    {
        if (sdf_textures == nullptr || sdf_textures[surface_index] == 0)
        {
            edge_normal_out = Vec3f(0.0f, 0.0f, 0.0f);
            return 1e30f;
        }
        const Vec3f local = world_to_surface_point(hit, s);
        const float hx = local.x / s.aperture_aspect;
        const float u  = 0.5f + hx    / (2.0f * s.aperture_semi_diameter);
        const float v  = 0.5f + local.y / (2.0f * s.aperture_semi_diameter);
        const float4 t = tex2D<float4>(sdf_textures[surface_index], u, v);
        const Vec3f local_normal(t.y, t.z, 0.0f);
        edge_normal_out = (s.decenter_x != 0.0f || s.decenter_y != 0.0f
                           || !rot_is_identity(s.rot))
                        ? mat3_mul(s.rot, local_normal) : local_normal;
        return t.x > 0.0f ? t.x : 0.0f;
    }
    return aperture_edge_distance(hit, s, edge_normal_out);
}

// ---------------------------------------------------------------------------
// d_trace_primary_ray
//
// Pure forward refraction through all surfaces on the GPU — no ghost bounces.
// Mirror of trace_primary_ray() in trace.cpp.  The sensor is anchored at
// world z=0 by convention.
//
// out_dir, when non-null, receives the ray's final direction — the leg from the
// last surface to the sensor. Only the OK path writes it, so callers must check
// status before reading. Keeping direction optional avoids enlarging TraceResult
// in the ghost kernel's inner loop.
// ---------------------------------------------------------------------------
__device__ __forceinline__
TraceResult d_trace_primary_ray(const Ray& ray_in,
                                 const Surface* surfs, int n_surfs,
                                 const cudaTextureObject_t* aperture_textures = nullptr,
                                 Vec3f* out_dir = nullptr)
{
    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;

    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    float current_ior = 1.0f;

    for (int s = 0; s < n_surfs; ++s)
    {
        // Muted surfaces are transparent — skip cleanly.
        if (!surfs[s].is_active) continue;
        float t; Vec3f norm;
        if (!intersect_surface(ray, surfs[s], t, norm))
            return result;

        Vec3f hit = ray.origin + ray.dir * t;
        ray.origin = hit;
        if (!d_image_aperture_passes(surfs[s], hit, aperture_textures, s))
            return result; // image-aperture vignetted

        const float n1 = current_ior;
        const float n2 = surfs[s].ior_at(lambda_nm);

        Vec3f new_dir;
        if (!refract_ray(ray.dir, norm, n1, n2, new_dir))
        {
            result.status = TraceStatus::TIR;
            return result;
        }
        result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                        surfs[s].coating, lambda_nm, false);
        result.weight *= surface_attenuator(hit, surfs[s]);
        ray.dir = new_dir;
        current_ior = n2;
    }

    if (fabsf(ray.dir.z) < 1e-12f)
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return result;
    }

    float t_sensor = -ray.origin.z / ray.dir.z;
    if (!(t_sensor > 0.0f))
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return result;
    }

    result.position = ray.origin + ray.dir * t_sensor;
    if (out_dir) *out_dir = ray.dir;
    result.status   = TraceStatus::OK;
    return result;
}

// ---------------------------------------------------------------------------
// d_trace_ghost_ray
//
// Three-leg sequential ghost trace on the GPU. Uses the flat Surface array
// (not OpticalSystem, which owns a std::vector and cannot be uploaded directly).
//
// Identical logic to trace_ghost_ray() in trace.cpp; the d_ prefix signals
// this function is __device__-only (called from a kernel, not the CPU tracer).
// The sensor is anchored at world z=0 by convention.
// ---------------------------------------------------------------------------
// ior_at / ior_before are the per-surface IOR slices for THIS ray's wavelength
// (n_surfs floats each) — see GpuBufferCache d_ior_at/d_ior_before. They replace
// the per-hit surfs[s].ior_at(λ) eval and the O(N²) backward walk.
// lambda_nm is still needed for the coating (surface_weight).
//
// The template parameter compiles the HURB edge-diffraction kick in or
// out. HURB == false compiles the kick out entirely (every kick site is an
// `if constexpr` that vanishes), so it costs nothing and changes no output; the hurb / hurb_seed args
// are then unused. HURB == true kicks the outgoing direction after every surface
// interaction, drawing from a per-ray RNG seeded by the caller (hurb_seed) — see
// aperture_edge_distance() (trace_core.h) and hurb_apply_kick() (hurb.h).
template<bool HURB>
__device__ __forceinline__
TraceResult d_trace_ghost_ray(const Ray& ray_in,
                               const Surface* surfs, int n_surfs,
                               int bounce_a, int bounce_b,
                               const float* ior_at, const float* ior_before,
                               const cudaTextureObject_t* aperture_textures = nullptr,
                               const cudaTextureObject_t* sdf_textures = nullptr,
                               GpuHurb hurb = GpuHurb{},
                               unsigned int hurb_seed = 0u)
{
    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;

    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    float current_ior = 1.0f;

    // Per-ray HURB RNG state, threaded across this trace's edge kicks. Only read
    // when HURB; otherwise written once and dead-code-eliminated.
    [[maybe_unused]] unsigned int hurb_state = hurb_seed;

    // ----------------------------------------------------------------
    // Outbound leg: forward through surfaces 0..bounce_b
    //          transmit at all except bounce_b (reflect)
    // ----------------------------------------------------------------
    // Muted surfaces are transparent across all three legs. Ghost
    // enumeration ensures bounce_a / bounce_b themselves are active, so
    // the reflect branches always have a real surface to bounce on.
    for (int s = 0; s <= bounce_b; ++s)
    {
        if (!surfs[s].is_active) continue;
        float t; Vec3f norm;
        if (!intersect_surface(ray, surfs[s], t, norm))
            return result; // vignetted

        Vec3f hit = ray.origin + ray.dir * t;
        ray.origin = hit;
        if (!d_image_aperture_passes(surfs[s], hit, aperture_textures, s))
            return result; // image-aperture vignetted

        const float n1 = current_ior;
        const float n2 = ior_at[s];

        if (s == bounce_b)
        {
            result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                            surfs[s].coating, lambda_nm, true);
            result.weight *= surface_attenuator(hit, surfs[s]);
            ray.dir = reflect_ray(ray.dir, norm);
            // current_ior unchanged — still in the medium before bounce_b
            if constexpr (HURB) {
                Vec3f e_n; float e_d = d_aperture_edge_distance(hit, surfs[s], s, sdf_textures, e_n);
                hurb_apply_kick(hurb, ray.dir, e_n, e_d, lambda_nm, hurb_state);
            }
        }
        else
        {
            Vec3f new_dir;
            if (!refract_ray(ray.dir, norm, n1, n2, new_dir))
            {
                result.status = TraceStatus::TIR;
                return result;
            }
            result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                            surfs[s].coating, lambda_nm, false);
            result.weight *= surface_attenuator(hit, surfs[s]);
            ray.dir = new_dir;
            current_ior = n2;
            if constexpr (HURB) {
                Vec3f e_n; float e_d = d_aperture_edge_distance(hit, surfs[s], s, sdf_textures, e_n);
                hurb_apply_kick(hurb, ray.dir, e_n, e_d, lambda_nm, hurb_state);
            }
        }
    }

    // ----------------------------------------------------------------
    // Return leg: backward through surfaces bounce_b-1..bounce_a
    //          transmit at all except bounce_a (reflect)
    // ----------------------------------------------------------------
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        if (!surfs[s].is_active) continue;
        float t; Vec3f norm;
        if (!intersect_surface(ray, surfs[s], t, norm))
            return result; // vignetted

        Vec3f hit = ray.origin + ray.dir * t;
        ray.origin = hit;
        if (!d_image_aperture_passes(surfs[s], hit, aperture_textures, s))
            return result; // image-aperture vignetted

        // Backward: n1 = current medium, n2 = medium on the other side of surface s.
        // ior_before[s] already skips muted surfaces to the last active surface's
        // post-medium, matching OpticalSystem::ior_before on the CPU path.
        const float n1 = current_ior;
        const float n2 = ior_before[s];

        if (s == bounce_a)
        {
            result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                            surfs[s].coating, lambda_nm, true);
            result.weight *= surface_attenuator(hit, surfs[s]);
            ray.dir = reflect_ray(ray.dir, norm);
            // After reflecting at bounce_a the ray is in the medium to the right.
            current_ior = ior_at[bounce_a];
            if constexpr (HURB) {
                Vec3f e_n; float e_d = d_aperture_edge_distance(hit, surfs[s], s, sdf_textures, e_n);
                hurb_apply_kick(hurb, ray.dir, e_n, e_d, lambda_nm, hurb_state);
            }
        }
        else
        {
            Vec3f new_dir;
            if (!refract_ray(ray.dir, norm, n1, n2, new_dir))
            {
                result.status = TraceStatus::TIR;
                return result;
            }
            result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                            surfs[s].coating, lambda_nm, false);
            result.weight *= surface_attenuator(hit, surfs[s]);
            ray.dir = new_dir;
            current_ior = n2;
            if constexpr (HURB) {
                Vec3f e_n; float e_d = d_aperture_edge_distance(hit, surfs[s], s, sdf_textures, e_n);
                hurb_apply_kick(hurb, ray.dir, e_n, e_d, lambda_nm, hurb_state);
            }
        }
    }

    // ----------------------------------------------------------------
    // Sensor leg: forward through surfaces bounce_a+1..n_surfs-1
    // ----------------------------------------------------------------
    for (int s = bounce_a + 1; s < n_surfs; ++s)
    {
        if (!surfs[s].is_active) continue;
        float t; Vec3f norm;
        if (!intersect_surface(ray, surfs[s], t, norm))
            return result; // vignetted

        Vec3f hit = ray.origin + ray.dir * t;
        ray.origin = hit;
        if (!d_image_aperture_passes(surfs[s], hit, aperture_textures, s))
            return result; // image-aperture vignetted

        const float n1 = current_ior;
        const float n2 = ior_at[s];

        Vec3f new_dir;
        if (!refract_ray(ray.dir, norm, n1, n2, new_dir))
        {
            result.status = TraceStatus::TIR;
            return result;
        }
        result.weight *= surface_weight(ray.dir, norm, n1, n2,
                                        surfs[s].coating, lambda_nm, false);
        result.weight *= surface_attenuator(hit, surfs[s]);
        ray.dir = new_dir;
        current_ior = n2;
        if constexpr (HURB) {
            Vec3f e_n; float e_d = d_aperture_edge_distance(hit, surfs[s], s, sdf_textures, e_n);
            hurb_apply_kick(hurb, ray.dir, e_n, e_d, lambda_nm, hurb_state);
        }
    }

    // ----------------------------------------------------------------
    // Propagate to sensor plane
    // ----------------------------------------------------------------
    if (fabsf(ray.dir.z) < 1e-12f)
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return result;
    }

    float t_sensor = -ray.origin.z / ray.dir.z;
    if (!(t_sensor > 0.0f))  // catches negative t and NaN
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return result;
    }

    result.position = ray.origin + ray.dir * t_sensor;
    result.status   = TraceStatus::OK;
    return result;
}
