// ============================================================================
// trace_core.h — Shared __host__ __device__ ray-tracing math
//
// Single source of truth for the geometry and optics primitives used by both
// the CPU stage (trace.cpp) and the GPU stage (trace_cuda.h).  Every function
// here compiles for both host and device; NVCC emits them twice but the PTX
// for the device path is identical to a __device__-only version with the same
// body.
//
// Rules: no STL, no std::, no dynamic allocation — GPU-safe throughout.
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

#include "ray.h"
#include "optical_system.h"    // Surface + Coating definitions (POD, GPU-safe)
#include "fresnel.h" // __host__ __device__ dispersion + Fresnel

// ---------------------------------------------------------------------------
// mat3_mul / mat3_T_mul / rot_is_identity
//
// GPU-safe 3×3 row-major matrix helpers used by intersect_surface to
// transform rays into and out of a surface's canonical frame.
// ---------------------------------------------------------------------------
__host__ __device__ inline
Vec3f mat3_mul(const float m[9], const Vec3f& v)
{
    return Vec3f(m[0]*v.x + m[1]*v.y + m[2]*v.z,
                 m[3]*v.x + m[4]*v.y + m[5]*v.z,
                 m[6]*v.x + m[7]*v.y + m[8]*v.z);
}

__host__ __device__ inline
Vec3f mat3_T_mul(const float m[9], const Vec3f& v)
{
    return Vec3f(m[0]*v.x + m[3]*v.y + m[6]*v.z,
                 m[1]*v.x + m[4]*v.y + m[7]*v.z,
                 m[2]*v.x + m[5]*v.y + m[8]*v.z);
}

__host__ __device__ inline
bool rot_is_identity(const float m[9])
{
    const float eps = 1e-6f;
    return (fabsf(m[0]-1.f)<eps && fabsf(m[1])<eps && fabsf(m[2])<eps &&
            fabsf(m[3])<eps && fabsf(m[4]-1.f)<eps && fabsf(m[5])<eps &&
            fabsf(m[6])<eps && fabsf(m[7])<eps && fabsf(m[8]-1.f)<eps);
}

__host__ __device__ inline
Vec3f world_to_surface_point(const Vec3f& point, const Surface& s)
{
    if (s.decenter_x == 0.0f && s.decenter_y == 0.0f && rot_is_identity(s.rot))
        return point;
    const Vec3f vertex(s.decenter_x, s.decenter_y, s.z);
    return mat3_T_mul(s.rot, point - vertex) + Vec3f(0.0f, 0.0f, s.z);
}

// ---------------------------------------------------------------------------
// asphere_sag
//
// Sag z(r) of an aspheric surface per ISO 10110 / Zemax convention.
// inv_R = 1/R (curvature c), K = conic constant.
// terms[0..n_terms-1] = A4, A6, A8, ...
// Returns 1e30f when r is outside the valid conic domain (D ≤ 0).
// ---------------------------------------------------------------------------
__host__ __device__ inline
float asphere_sag(float r, float inv_R, float K,
                  const float* terms, int n_terms)
{
    if (n_terms < 0) n_terms = 0;
    if (n_terms > MAX_ASPHERE_TERMS) n_terms = MAX_ASPHERE_TERMS;
    float r2 = r * r;
    float D  = 1.0f - (1.0f + K) * inv_R * inv_R * r2;
    if (D <= 0.0f) return 1e30f;
    float z  = inv_R * r2 / (1.0f + sqrtf(D));
    float rp = r2 * r2; // r^4
    for (int i = 0; i < n_terms; ++i) { z += terms[i] * rp; rp *= r2; }
    return z;
}

// ---------------------------------------------------------------------------
// asphere_dsag_dr
//
// Derivative dz/dr of the asphere sag (needed for Newton-Raphson and normal).
// Returns 0 when r is outside the valid conic domain (D ≤ 0).
// ---------------------------------------------------------------------------
__host__ __device__ inline
float asphere_dsag_dr(float r, float inv_R, float K,
                      const float* terms, int n_terms)
{
    if (n_terms < 0) n_terms = 0;
    if (n_terms > MAX_ASPHERE_TERMS) n_terms = MAX_ASPHERE_TERMS;
    float r2 = r * r;
    float D  = 1.0f - (1.0f + K) * inv_R * inv_R * r2;
    if (D <= 0.0f) return 0.0f;
    float dz = inv_R * r / sqrtf(D);
    float rp = r2 * r; // r^3
    for (int i = 0; i < n_terms; ++i) {
        dz += (4.0f + 2.0f * i) * terms[i] * rp;
        rp *= r2;
    }
    return dz;
}

// ---------------------------------------------------------------------------
// check_aperture
//
// Shape-aware aperture vignetting test: rejects a hit point outside the surface's
// clear aperture, honouring non-circular aperture shapes (the simple circular
// case is `h.x*h.x + h.y*h.y <= semi_aperture²`).
//
// aperture_aspect stretches the X axis: the bounding region is an ellipse with
// X half-axis (semi_aperture * aspect) and Y half-axis semi_aperture.  The
// polygon test runs in the canonical (stretched) frame so the polygon stays
// regular relative to the unit Y bound and inherits the aspect stretch in X.
//
// A polygon stop with blade shape authored (curvature / twist / notch) tests
// against Surface::aperture_profile instead — the same normalised r(theta) the
// pupil mask and diffraction pupil use, so all three agree on one silhouette.
//
// The !(... <= ...) form (rather than > ) correctly rejects NaN hit points
// produced by degenerate ray directions.
// ---------------------------------------------------------------------------
__host__ __device__ inline
bool check_aperture(const Vec3f& hit, const Surface& s)
{
    // Apply aspect to X once.  All shape tests run in this stretched frame.
    float hx = hit.x / s.aperture_aspect;
    float hy = hit.y;
    float h2 = hx*hx + hy*hy;

    // Bounding-ellipse fast reject.  Also rejects NaN.
    if (!(h2 <= s.semi_aperture * s.semi_aperture))
        return false;

    if (s.aperture_shape == APERTURE_POLYGON)
    {
        float r = sqrtf(h2);
        if (r < 1e-9f) return true; // on-axis always passes

        // Blade shape authored (curvature / twist / notch): the boundary is the
        // conic-arc profile, evaluated in the normalised frame and scaled by
        // semi_aperture here.
        if (s.aperture_profile.deformed())
            return r <= s.semi_aperture * s.aperture_profile.radius_at(atan2f(hy, hx));

        // Keep the regular-polygon expression separate because evaluating the
        // equivalent conic changes its floating-point boundary.
        const float sector = 2.0f * (float)M_PI / (float)s.aperture_blades;   // exact mirror of aperture_edge_distance
        const float angle  = atan2f(hy, hx) - s.aperture_rotation_rad;
        float offset = fmodf(angle, sector);
        if (offset < 0.0f) offset += sector;
        offset -= 0.5f * sector; // normalise to [-sector/2, +sector/2]

        const float apothem = s.semi_aperture * cosf(0.5f * sector);
        return r <= apothem / cosf(offset);
    }

    // APERTURE_CIRCLE: bounding-ellipse test above is the only test.
    return true;
}

// ---------------------------------------------------------------------------
// aperture_edge_distance
//
// Perpendicular distance (mm) from a world-space hit point to the NEAREST
// clear-aperture edge of a surface, plus the (world-space) transverse edge
// normal at that point. Used by HURB (hurb.h): a ghost ray that clears an edge
// at small distance d gets an angular kick of scale sigma ~ lambda*K/d.
//
// The geometry mirrors check_aperture() exactly (same stretched frame, same
// polygon math), so "distance 0" coincides with check_aperture()'s pass/fail
// boundary. Derivations (stretched frame; aspect applied to X as in
// check_aperture, r_s = sqrt((x/aspect)^2 + y^2)):
//   * circle / ellipse: the boundary is the level set g(x,y)=semi_aperture of
//     g = r_s, whose gradient is (x/aspect^2, y)/r_s, so the first-order
//     distance is (semi_aperture - r_s) * r_s / sqrt(x^2/aspect^4 + y^2) and
//     the outward normal is proportional to (x/aspect^2, y). For aspect==1 this
//     collapses to the exact radial distance semi_aperture - r.
//   * polygon: distance to the nearest side line is apothem - r_s*cos(offset)
//     (offset = angle within the current sector, 0 at a side's midpoint); the
//     side's outward normal points at (point_angle - offset).
//   * bladed polygon (curvature / twist / notch): delegated to
//     ApertureProfile::edge_gap(), the first-order distance to the conic blade
//     edge (and the exact distance to a tip facet's chord, whichever is
//     nearer). It reduces to the straight-side expression above at zero
//     deformation; see its own comment for where the linearisation is loose.
// APERTURE_IMAGE has no analytic edge, so it returns a large sentinel (no kick).
// The transform into/out of the surface's canonical frame is copied from
// intersect_surface(); the normal is rotated back to world with mat3_mul(s.rot).
//
// The sign of edge_normal_out is irrelevant to the caller (the kick angle is
// symmetric), so no inward/outward convention is enforced.
// ---------------------------------------------------------------------------
__host__ __device__ inline
float aperture_edge_distance(const Vec3f& hit_world, const Surface& s,
                             Vec3f& edge_normal_out)
{
    // Image apertures: no analytic edge distance -> never kick.
    if (s.aperture_shape == APERTURE_IMAGE)
    {
        edge_normal_out = Vec3f(0.0f, 0.0f, 0.0f);
        return 1e30f;
    }

    // Canonical-frame transform (identical to intersect_surface's).
    bool transformed = (s.decenter_x != 0.0f || s.decenter_y != 0.0f
                        || !rot_is_identity(s.rot));
    Vec3f h = world_to_surface_point(hit_world, s);

    const float sa  = s.semi_aperture;
    const float hx  = h.x / s.aperture_aspect;   // stretched frame (as check_aperture)
    const float hy  = h.y;
    const float r_s = sqrtf(hx * hx + hy * hy);

    float d;
    Vec3f n_canon;

    if (s.aperture_shape == APERTURE_POLYGON && s.aperture_blades >= 3)
    {
        const ApertureProfile& pr = s.aperture_profile;
        const float sector  = 2.0f * (float)M_PI / (float)s.aperture_blades;
        const float apothem = sa * cosf(0.5f * sector);
        if (r_s < 1e-9f)
        {
            // Dead centre: no edge direction, and the nearest edge is the
            // blade's closest approach — the apothem, or the waist once the
            // edge is scooped.
            d       = pr.deformed() ? sa * pr.r_w : apothem;
            n_canon = Vec3f(1.0f, 0.0f, 0.0f);
        }
        else if (pr.deformed())
        {
            // The profile works in the normalised frame, so the radius goes in
            // and the distance comes back out scaled by semi_aperture; the
            // normal is already a unit vector.
            const float theta = atan2f(hy, hx);
            float nx, ny;
            d       = sa * pr.edge_gap(r_s / sa, theta, &nx, &ny);
            n_canon = Vec3f(nx, ny, 0.0f);
        }
        else
        {
            const float theta = atan2f(hy, hx);
            float offset = fmodf(theta - s.aperture_rotation_rad, sector);
            if (offset < 0.0f) offset += sector;
            offset -= 0.5f * sector;              // [-sector/2, +sector/2], 0 at side midpoint
            d = apothem - r_s * cosf(offset);     // distance to the nearest side line
            const float phi = theta - offset;     // that side's outward normal angle
            n_canon = Vec3f(cosf(phi), sinf(phi), 0.0f);
        }
    }
    else
    {
        // Circle / ellipse. gradient of the stretched radius = (x/aspect^2, y).
        const float asp2 = s.aperture_aspect * s.aperture_aspect;
        const float gx   = h.x / asp2;
        const float gy   = h.y;
        const float glen = sqrtf(gx * gx + gy * gy);
        if (glen < 1e-12f)
        {
            d       = sa;                         // dead centre: distance ~ semi-aperture
            n_canon = Vec3f(1.0f, 0.0f, 0.0f);
        }
        else
        {
            d       = (sa - r_s) * r_s / glen;
            n_canon = Vec3f(gx / glen, gy / glen, 0.0f);
        }
    }

    if (d < 0.0f) d = 0.0f;                        // inside-only caller; guard rounding

    edge_normal_out = transformed ? mat3_mul(s.rot, n_canon) : n_canon;
    return d;
}

// ---------------------------------------------------------------------------
// intersect_surface
//
// Intersect a ray with a lens surface (flat, spherical, aspheric, or
// cylindrical).  On success: t_out is the parametric distance along the ray
// to the hit; normal_out is the surface normal opposing the ray direction.
// Returns false on miss, aperture clip, or no positive t.
// Caller computes hit position as: r.origin + r.dir * t_out
//
// Aperture vignetting is delegated to check_aperture(); see its docstring for
// the bounding-ellipse + polygon math.
// ---------------------------------------------------------------------------
__host__ __device__ inline
bool intersect_surface(const Ray& r, const Surface& s,
                       float& t_out, Vec3f& normal_out)
{
    // ---- Canonical-frame transform ----
    // If this surface has a non-identity rigid-body transform (decenter and/or
    // tilt), rotate the ray into the surface's canonical frame (vertex at
    // (0,0,s.z), axis along +Z) before running the per-form math.  t is
    // invariant under a rigid transform so it is returned as-is; callers still
    // compute world hit = r.origin + r.dir * t.  The normal is rotated back to
    // world space before return.
    bool transformed = (s.decenter_x != 0.0f || s.decenter_y != 0.0f
                        || !rot_is_identity(s.rot));
    Ray rc = r; // canonical-frame copy; equal to r when !transformed
    if (transformed)
    {
        Vec3f V   = Vec3f(s.decenter_x, s.decenter_y, s.z);
        rc.origin = mat3_T_mul(s.rot, r.origin - V) + Vec3f(0.0f, 0.0f, s.z);
        rc.dir    = mat3_T_mul(s.rot, r.dir);
    }

    if (fabsf(s.radius) < 1e-6f)
    {
        // ---- Flat surface: ray–plane at z = s.z ----
        if (fabsf(rc.dir.z) < 1e-12f) return false; // parallel

        float t = (s.z - rc.origin.z) / rc.dir.z;
        if (!(t > 1e-6f)) return false; // catches t <= 0, NaN, Inf

        Vec3f hit = rc.origin + rc.dir * t;
        if (!check_aperture(hit, s))
            return false;

        t_out      = t;
        normal_out = Vec3f(0.0f, 0.0f, (rc.dir.z > 0.0f) ? -1.0f : 1.0f);
        if (transformed) normal_out = mat3_mul(s.rot, normal_out);
        return true;
    }
    else if (s.form == FORM_ASPHERE)
    {
        // ---- Aspheric surface: Newton-Raphson on sag residual ----
        // Solve F(t) = hit.z - s.z - asphere_sag(r_xy) = 0
        float R     = s.radius;
        float inv_R = 1.0f / R;
        float K     = s.conic_k;

        // Initial guess from spherical approximation (ignores K and poly terms).
        Vec3f ctr  = Vec3f(0.0f, 0.0f, s.z + R);
        Vec3f oc   = rc.origin - ctr;
        float qa   = dot(rc.dir, rc.dir);
        float qb   = 2.0f * dot(oc, rc.dir);
        float qc   = dot(oc, oc) - R * R;
        float disc = qb*qb - 4.0f*qa*qc;

        float t;
        if (disc >= 0.0f)
        {
            float sd     = sqrtf(disc);
            float inv_2a = 0.5f / qa;
            float t1     = (-qb - sd) * inv_2a;
            float t2     = (-qb + sd) * inv_2a;

            if (t1 > 1e-6f && t2 > 1e-6f)
            {
                float z1 = rc.origin.z + t1 * rc.dir.z;
                float z2 = rc.origin.z + t2 * rc.dir.z;
                t = (fabsf(z1 - s.z) < fabsf(z2 - s.z)) ? t1 : t2;
            }
            else if (t1 > 1e-6f) t = t1;
            else if (t2 > 1e-6f) t = t2;
            else return false;
        }
        else
        {
            // Sphere misses; fall back to plane tangent at vertex.
            if (fabsf(rc.dir.z) < 1e-12f) return false;
            t = (s.z - rc.origin.z) / rc.dir.z;
            if (!(t > 1e-6f)) return false;
        }

        for (int iter = 0; iter < 15; ++iter)
        {
            Vec3f hit_i = rc.origin + rc.dir * t;
            float rx    = hit_i.x;
            float ry    = hit_i.y;
            float rad   = sqrtf(rx*rx + ry*ry);

            float sag = asphere_sag(rad, inv_R, K, s.asphere_terms, s.n_asphere_terms);
            if (sag > 1e29f) return false; // outside valid conic domain

            float f = hit_i.z - s.z - sag;
            if (fabsf(f) < 1e-5f) break;

            float dsag   = asphere_dsag_dr(rad, inv_R, K, s.asphere_terms, s.n_asphere_terms);
            float r_safe = (rad > 1e-9f) ? rad : 1e-9f;
            float df_dt  = rc.dir.z - dsag * (rx * rc.dir.x + ry * rc.dir.y) / r_safe;
            if (fabsf(df_dt) < 1e-12f) break; // degenerate Jacobian
            t -= f / df_dt;
        }

        if (!(t > 1e-6f)) return false;

        Vec3f hit = rc.origin + rc.dir * t;
        float rad = sqrtf(hit.x * hit.x + hit.y * hit.y);
        if (!check_aperture(hit, s))
            return false;

        float r_safe = (rad > 1e-9f) ? rad : 1e-9f;
        float dsag   = asphere_dsag_dr(rad, inv_R, K, s.asphere_terms, s.n_asphere_terms);
        Vec3f normal = Vec3f(-dsag * hit.x / r_safe,
                             -dsag * hit.y / r_safe,
                             1.0f);
        float nlen = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        normal = normal * (1.0f / nlen);
        if (dot(normal, rc.dir) > 0.0f) normal = -normal;
        if (transformed) normal = mat3_mul(s.rot, normal);

        t_out      = t;
        normal_out = normal;
        return true;
    }
    else if (s.form == FORM_CYLINDRICAL)
    {
        // ---- Cylindrical surface: exact 2-D quadratic in canonical frame ----
        // The cylinder has curvature in one transverse plane only; the surface
        // normal has no component along the flat (cylinder) axis.
        //
        // Roll and tilt are handled by the canonical-frame transform above; the
        // cylinder math uses base axis directions only:
        //   CYL_AXIS_Y: curvature along X → v = (1, 0)
        //   CYL_AXIS_X: curvature along Y → v = (0, 1)
        float vx, vy;
        if (s.cyl_axis == CYL_AXIS_Y) { vx = 1.0f; vy = 0.0f; }
        else                           { vx = 0.0f; vy = 1.0f; }

        float R    = s.radius;
        float oc_v = rc.origin.x * vx + rc.origin.y * vy;
        float oc_z = rc.origin.z - s.z - R;
        float dv   = rc.dir.x * vx + rc.dir.y * vy;

        float qa = dv * dv + rc.dir.z * rc.dir.z;
        float qb = 2.0f * (oc_v * dv + oc_z * rc.dir.z);
        float qc = oc_v * oc_v + oc_z * oc_z - R * R;

        float disc = qb*qb - 4.0f*qa*qc;
        if (disc < 0.0f) return false;

        float sd     = sqrtf(disc);
        float inv_2a = 0.5f / qa;
        float t1     = (-qb - sd) * inv_2a;
        float t2     = (-qb + sd) * inv_2a;

        // Pick the intersection closest to the surface vertex z.
        float t;
        if (t1 > 1e-6f && t2 > 1e-6f)
        {
            float z1 = rc.origin.z + t1 * rc.dir.z;
            float z2 = rc.origin.z + t2 * rc.dir.z;
            t = (fabsf(z1 - s.z) < fabsf(z2 - s.z)) ? t1 : t2;
        }
        else if (t1 > 1e-6f) t = t1;
        else if (t2 > 1e-6f) t = t2;
        else return false;

        Vec3f hit = rc.origin + rc.dir * t;
        if (!check_aperture(hit, s))
            return false;

        // Normal: radial in the curving plane, zero component along the flat axis.
        float hv     = hit.x * vx + hit.y * vy;
        Vec3f normal = Vec3f(hv * vx, hv * vy, hit.z - s.z - R);

        float nlen = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        if (nlen < 1e-12f) return false;
        normal = normal * (1.0f / nlen);
        if (dot(normal, rc.dir) > 0.0f) normal = -normal;
        if (transformed) normal = mat3_mul(s.rot, normal);

        t_out      = t;
        normal_out = normal;
        return true;
    }
    else // FORM_SPHERE (or default)
    {
        // ---- Spherical surface ----
        // Centre of curvature: C = (0, 0, s.z + s.radius)
        float R    = s.radius;
        Vec3f ctr  = Vec3f(0.0f, 0.0f, s.z + R);
        Vec3f oc   = rc.origin - ctr;

        float a    = dot(rc.dir, rc.dir);
        float b    = 2.0f * dot(oc, rc.dir);
        float c    = dot(oc, oc) - R * R;
        float disc = b*b - 4.0f*a*c;
        if (disc < 0.0f) return false;

        float sd     = sqrtf(disc);
        float inv_2a = 0.5f / a;
        float t1     = (-b - sd) * inv_2a;
        float t2     = (-b + sd) * inv_2a;

        // Pick the intersection closest to the surface vertex z.
        float t;
        if (t1 > 1e-6f && t2 > 1e-6f)
        {
            float z1 = rc.origin.z + t1 * rc.dir.z;
            float z2 = rc.origin.z + t2 * rc.dir.z;
            t = (fabsf(z1 - s.z) < fabsf(z2 - s.z)) ? t1 : t2;
        }
        else if (t1 > 1e-6f) t = t1;
        else if (t2 > 1e-6f) t = t2;
        else return false;

        Vec3f hit = rc.origin + rc.dir * t;
        if (!check_aperture(hit, s))
            return false;

        // Normal from sphere centre toward hit point; ensure it opposes the ray.
        Vec3f normal = (hit - ctr) * (1.0f / fabsf(R));
        if (dot(normal, rc.dir) > 0.0f) normal = -normal;
        if (transformed) normal = mat3_mul(s.rot, normal);

        t_out      = t;
        normal_out = normal;
        return true;
    }
}

// ---------------------------------------------------------------------------
// refract_ray
//
// Snell's law refraction.  normal must oppose the incoming ray direction.
// Returns false on total internal reflection or a degenerate result.
// ---------------------------------------------------------------------------
__host__ __device__ inline
bool refract_ray(const Vec3f& dir, const Vec3f& normal,
                 float n1, float n2, Vec3f& out)
{
    float n_ratio = n1 / n2;
    float cos_i   = -dot(normal, dir);
    float sin2_t  = n_ratio * n_ratio * (1.0f - cos_i * cos_i);

    if (sin2_t >= 1.0f) return false; // total internal reflection

    float cos_t = sqrtf(1.0f - sin2_t);
    Vec3f tmp   = dir * n_ratio + normal * (n_ratio * cos_i - cos_t);

    float sq = tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z;
    if (sq < 1e-18f) return false; // degenerate direction

    float inv = 1.0f / sqrtf(sq);
    out = tmp * inv;
    return true;
}

// ---------------------------------------------------------------------------
// reflect_ray
//
// Specular reflection.  normal must oppose the incoming ray direction.
// ---------------------------------------------------------------------------
__host__ __device__ inline
Vec3f reflect_ray(const Vec3f& dir, const Vec3f& normal)
{
    Vec3f tmp = dir - normal * (2.0f * dot(dir, normal));
    return tmp.normalized();
}

// ---------------------------------------------------------------------------
// coating_lerp_1d
//
// Linear interpolation into a sorted-ascending CoatingTable1D array (key =
// lambda_nm or angle_deg).  Out-of-range keys clamp to the edge value, or —
// when discard_oob — set *discarded so the caller can zero the contribution.
// Linear scan: coating tables are small (≤ ~64 entries) and branch-friendly
// on the GPU.
// ---------------------------------------------------------------------------
__host__ __device__ inline
float coating_lerp_1d(const CoatingTable1D* t, int n, float key,
                      bool discard_oob, bool* discarded)
{
    if (!t || n <= 0) return 0.0f;

    if (key <= t[0].key)
    {
        if (discard_oob && key < t[0].key) *discarded = true;
        return t[0].r;
    }
    if (key >= t[n - 1].key)
    {
        if (discard_oob && key > t[n - 1].key) *discarded = true;
        return t[n - 1].r;
    }

    for (int i = 1; i < n; ++i)
    {
        if (key <= t[i].key)
        {
            float span = t[i].key - t[i - 1].key;
            float f    = (span > 0.0f) ? (key - t[i - 1].key) / span : 0.0f;
            return t[i - 1].r + f * (t[i].r - t[i - 1].r);
        }
    }
    return t[n - 1].r; // unreachable (key < last handled above)
}

// ---------------------------------------------------------------------------
// coating_ref_angle_deg
//
// Convert a local incidence angle (in the medium with IOR n1) to the coating
// table's reference-medium angle via the Snell invariant n·sinθ.  Coating
// tables are conventionally measured from the air side (angle_ref_ior = 1);
// a ghost bounce hits the same coating from the glass side, where the local
// angle is smaller for the same invariant.  Super-critical invariants (no
// real reference angle) clamp to grazing (90°), i.e. the table's last rows.
// ---------------------------------------------------------------------------
__host__ __device__ inline
float coating_ref_angle_deg(float cos_i, float n1, float angle_ref_ior)
{
    float sin_i = sqrtf(fmaxf(0.0f, 1.0f - cos_i * cos_i));
    float s_ref = (angle_ref_ior > 0.0f) ? (n1 * sin_i / angle_ref_ior) : sin_i;
    if (s_ref >= 1.0f) return 90.0f;
    return asinf(s_ref) * (180.0f / (float)M_PI);
}

// ---------------------------------------------------------------------------
// surface_weight
//
// Returns the energy weight for a ray interacting with a surface:
//   reflecting == true  → reflectance  R
//   reflecting == false → transmittance 1 - R
//
// Dispatches on coating.model.  Table-backed models with a null table pointer
// (defensive: sync/upload not run) fall back to bare Fresnel.  A table lookup
// out of range in discard mode kills the contribution entirely (weight 0 for
// both reflection and transmission).
// ---------------------------------------------------------------------------
__host__ __device__ inline
float surface_weight(const Vec3f& dir, const Vec3f& normal,
                     float n1, float n2, const Coating& coating,
                     float lambda_nm, bool reflecting)
{
    float cos_i = fabsf(dot(normal, dir));
    float R;

    switch (coating.model)
    {
        case CoatingModel::SIMPLE:
            R = surface_reflectance(cos_i, n1, n2, coating.ar_layers, lambda_nm);
            break;

        case CoatingModel::ARTIST:
            R = artist_reflectance(coating.tint_r, coating.tint_g,
                                   coating.tint_b, coating.tint_strength,
                                   lambda_nm);
            break;

        case CoatingModel::SPECTRAL:
        {
            if (!coating.table || coating.table_count <= 0)
            {
                R = fresnel_reflectance(cos_i, n1, n2);
                break;
            }
            bool discarded = false;
            R = coating_lerp_1d(coating.table, coating.table_count, lambda_nm,
                                coating.out_of_range_discard, &discarded);
            if (discarded) return 0.0f;
            break;
        }

        case CoatingModel::ANGULAR:
        {
            if (!coating.table || coating.table_count <= 0)
            {
                R = fresnel_reflectance(cos_i, n1, n2);
                break;
            }
            float ang = coating_ref_angle_deg(cos_i, n1, coating.angle_ref_ior);
            bool discarded = false;
            R = coating_lerp_1d(coating.table, coating.table_count, ang,
                                coating.out_of_range_discard, &discarded);
            if (discarded) return 0.0f;
            break;
        }

        case CoatingModel::SPECTRAL_ANGULAR:
        {
            const int nw = coating.sa_n_wavelengths;
            const int na = coating.sa_n_angles;
            if (!coating.sa_r || !coating.sa_wavelengths || !coating.sa_angles
                || nw <= 0 || na <= 0)
            {
                R = fresnel_reflectance(cos_i, n1, n2);
                break;
            }

            float ang = coating_ref_angle_deg(cos_i, n1, coating.angle_ref_ior);
            bool discarded = false;

            // Find bracketing indices + fractions on each axis (edge clamp).
            int   iw = 0; float fw = 0.0f;
            if (lambda_nm <= coating.sa_wavelengths[0])
            {
                if (coating.out_of_range_discard
                    && lambda_nm < coating.sa_wavelengths[0]) discarded = true;
            }
            else if (lambda_nm >= coating.sa_wavelengths[nw - 1])
            {
                iw = (nw > 1) ? nw - 2 : 0;
                fw = (nw > 1) ? 1.0f : 0.0f;
                if (coating.out_of_range_discard
                    && lambda_nm > coating.sa_wavelengths[nw - 1]) discarded = true;
            }
            else
            {
                for (int i = 1; i < nw; ++i)
                {
                    if (lambda_nm <= coating.sa_wavelengths[i])
                    {
                        iw = i - 1;
                        float span = coating.sa_wavelengths[i]
                                   - coating.sa_wavelengths[i - 1];
                        fw = (span > 0.0f)
                           ? (lambda_nm - coating.sa_wavelengths[i - 1]) / span
                           : 0.0f;
                        break;
                    }
                }
            }

            int   ia = 0; float fa = 0.0f;
            if (ang <= coating.sa_angles[0])
            {
                if (coating.out_of_range_discard
                    && ang < coating.sa_angles[0]) discarded = true;
            }
            else if (ang >= coating.sa_angles[na - 1])
            {
                ia = (na > 1) ? na - 2 : 0;
                fa = (na > 1) ? 1.0f : 0.0f;
                if (coating.out_of_range_discard
                    && ang > coating.sa_angles[na - 1]) discarded = true;
            }
            else
            {
                for (int i = 1; i < na; ++i)
                {
                    if (ang <= coating.sa_angles[i])
                    {
                        ia = i - 1;
                        float span = coating.sa_angles[i]
                                   - coating.sa_angles[i - 1];
                        fa = (span > 0.0f)
                           ? (ang - coating.sa_angles[i - 1]) / span
                           : 0.0f;
                        break;
                    }
                }
            }

            if (discarded) return 0.0f;

            const int iw1 = (iw + 1 < nw) ? iw + 1 : iw;
            const int ia1 = (ia + 1 < na) ? ia + 1 : ia;
            float r00 = coating.sa_r[iw  * na + ia ];
            float r01 = coating.sa_r[iw  * na + ia1];
            float r10 = coating.sa_r[iw1 * na + ia ];
            float r11 = coating.sa_r[iw1 * na + ia1];
            R = (1.0f - fw) * ((1.0f - fa) * r00 + fa * r01)
              +         fw  * ((1.0f - fa) * r10 + fa * r11);
            break;
        }

        // ATTENUATOR_GAUSS carries no reflectance model of its own — the
        // positional attenuation is applied by surface_attenuator(); the
        // interface itself reflects/transmits as bare Fresnel.
        case CoatingModel::ATTENUATOR_GAUSS:
        default:
            R = fresnel_reflectance(cos_i, n1, n2);
            break;
    }

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    return reflecting ? R : (1.0f - R);
}

// ---------------------------------------------------------------------------
// surface_attenuator
//
// Returns the amplitude multiplier for a positional attenuator coating.
// Returns 1.0 for any non-attenuator coating model.
// ---------------------------------------------------------------------------
__host__ __device__ inline
float surface_attenuator(const Vec3f& hit_point, const Surface& surface)
{
    const Coating& coating = surface.coating;
    if (coating.model == CoatingModel::ATTENUATOR_GAUSS)
    {
        if (!(coating.gauss_sigma > 0.0f))
            return coating.gauss_background;
        const Vec3f local = world_to_surface_point(hit_point, surface);
        float dx = local.x - coating.gauss_decenter_x;
        float dy = local.y - coating.gauss_decenter_y;
        float r2 = dx*dx + dy*dy;
        float s2 = coating.gauss_sigma * coating.gauss_sigma;
        return coating.gauss_background +
               coating.gauss_peak * expf(-0.5f * r2 / s2);
    }

    return 1.0f;
}
