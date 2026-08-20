// ============================================================================
// aperture_profile.h — Bladed aperture boundary in the normalised frame.
//
// r(theta) in (0, 1], where 1.0 is the blade-tip (circumscribed) radius.
// Rotation is carried here; callers apply semi_aperture and aperture_aspect.
// At rotation 0 a blade tip points along +X (see sector_angle).
//
// Each half of a blade edge is a conic arc with its focus at the aperture
// centre, which makes it linear in u = 1/r:
//
//     u(phi) = A*cos(phi - phi_w) + C
//
// Pinning u(phi_w) = 1/r_w (the waist) and u(tip) = 1 gives A and C in closed
// form. The construction reduces EXACTLY to a/cos(phi) — the regular polygon —
// at curvature 0 and EXACTLY to u = 1 — the circle — at curvature +1, is
// monotone waist->tip (so star-convex by construction, no root selection, no
// feasibility limit), is C1 at the waist, and costs one trig call and one
// divide.
// Tip facets are straight chords, also linear in u, so the whole profile is a
// max over same-shaped terms.
//
// Rules: no STL, no dynamic allocation — the struct is a pass-by-value POD that
// rides on Surface into the GPU kernels.
// ============================================================================
#pragma once

#include <cmath>

#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Deformation constants define the semantics of authored lens values.
//
// NEG_DEPTH: negative curvature scoops the waist by a fraction OF THE APOTHEM,
// not of the absolute tip-to-apothem gap. That gap shrinks as blades rise (0.5
// at 3 blades, 0.04 at 11), so scooping by it would pinch a 3-blade pupil shut
// while doing almost nothing at 11.
// MIN_SPAN: the narrowest half-edge twist may leave, so neither side ever
// collapses to a wall too steep for the pupil raster to resolve.
// NOTCH_*: notch is authored in degrees but applied as a fraction of the
// half-sector so its effect is stable across blade counts.
constexpr float AP_NEG_DEPTH    = 0.55f;
constexpr float AP_MIN_SPAN     = 4.0f * (float)(M_PI / 180.0);
constexpr float AP_NOTCH_FULL   = 45.0f * (float)(M_PI / 180.0);
constexpr float AP_NOTCH_MAX    = 0.45f;
constexpr float AP_NOTCH_SECTOR = 0.90f;
constexpr int   AP_AREA_SAMPLES = 512;   // Simpson panels over one sector (even)

// Authored ranges, hard-clamped wherever a profile is built.
constexpr float AP_NOTCH_LIMIT       = (float)(M_PI / 4.0);   // +/- 45 deg
constexpr float AP_NOTCH_ANGLE_LIMIT = (float)(M_PI / 4.0);   // 0 .. 45 deg

// Blade-shape controls as authored. Grouped so every consumer that builds or
// compares a profile names the same four things in the same order.
struct ApertureShapeParams
{
    float curvature   = 0.0f;   // [-1, +1]  waist depth: -1 scalloped, +1 circle
    float twist       = 0.0f;   // [-1, +1]  waist position along the edge
    float notch       = 0.0f;   // radians, [-45deg, +45deg], nominal
    float notch_angle = 0.0f;   // radians, [0, 45deg]        nominal skew
};

__host__ __device__ inline float ap_clamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Derived blade geometry. Every member is a 4-byte type with a default member
// initialiser: this struct is embedded in Surface, which source_map.cpp hashes
// as raw bytes for its cache key, so a padding hole or an indeterminate member
// would make that key nondeterministic.
struct ApertureProfile
{
    int   blades   = 0;      // < 3: not a bladed stop, the profile is unused
    // 1 selects the direct regular-polygon expression and avoids unnecessary
    // conic evaluation when no deformation is authored.
    int   plain    = 1;
    int   facets   = 0;      // 1 when a tip facet is active
    float rotation = 0.0f;   // radians, copied from the surface

    float sigma    = 0.0f;   // 2*pi / blades
    float half     = 0.0f;   // sigma / 2  (tips sit at +/- half)
    float r_w      = 1.0f;   // waist radius
    float phi_w    = 0.0f;   // waist position within the sector

    // Conic edge, stored in the cancellation-free form (see u_edge).
    float u_w      = 1.0f;   // 1 / r_w — u at the waist
    float u_rise   = 0.0f;   // u_w - 1 — how far u falls from waist to tip
    float qR       = 0.0f;   // 1 / sin^2(dR/2), waist -> +tip
    float qL       = 0.0f;   // 1 / sin^2(dL/2), waist -> -tip

    float p    = 0.0f;       // facet chord: distance from centre
    float beta = 0.0f;       // facet chord: normal direction
    float eA   = 0.0f;       // facet span, clockwise of the tip
    float eB   = 0.0f;       // facet span, anticlockwise of the tip

    // Enclosed area as a fraction of the unit disk (1.0 when unused). The stop
    // profile's value becomes LensCalibration::pupil_area_frac.
    float area_frac = 1.0f;

    // True when the silhouette needs the conic evaluation rather than the plain
    // polygon expression.
    __host__ __device__ bool deformed() const { return blades >= 3 && !plain; }

    // The conic edge and its slope at a sector-local angle. Every consumer goes
    // through here so the value and the derivative can't drift apart.
    //
    // Avoid cancellation in u(phi) = A*cos(phi - phi_w) + C by evaluating the
    // algebraically equivalent interpolation
    //
    //     u = u_w - (u_w - 1) * sin^2(psi/2) / sin^2(delta/2)
    //
    // which has no large opposing intermediate terms.
    __host__ __device__ void edge_at(float phi, float* u, float* du) const
    {
        const float psi = phi - phi_w;
        const float sh  = sinf(0.5f * psi);
        const float q   = (psi >= 0.0f) ? qR : qL;
        *u  = u_w - u_rise * (sh * sh) * q;
        // d/dpsi of sin^2(psi/2) is sin(psi)/2.
        *du = -0.5f * u_rise * q * sinf(psi);
    }

    __host__ __device__ float u_edge(float phi) const
    {
        float u, du;
        edge_at(phi, &u, &du);
        return u;
    }

    // Fold an angle into [-sigma/2, +sigma/2).
    __host__ __device__ float fold(float ang) const
    {
        float a = fmodf(ang + half, sigma);
        if (a < 0.0f) a += sigma;
        return a - half;
    }

    // u = 1/r at a sector-local angle. One facet per tip, so the facet family is
    // sigma-periodic and folding against the TIP direction picks the relevant
    // one. The facet is a SEGMENT, not a half-plane — applied only across its
    // own angular span, because a skewed chord's infinite line would cut regions
    // the real facet never reaches and collapse the silhouette. Both endpoints
    // lie ON the edge, so the two constraints agree exactly at the span
    // boundaries and the profile stays continuous.
    __host__ __device__ float u_raw(float phi) const
    {
        float u = u_edge(phi);
        if (facets)
        {
            const float w = fold(phi - half);
            if (w >= -eA && w <= eB)
            {
                const float uf = cosf(w + half - beta) / p;
                if (uf > u) u = uf;
            }
        }
        return u;
    }

    // Sector-local angle for an absolute angle theta (rotation removed), zero
    // at the middle of a blade edge and +/- half at the tips.
    //
    // The polygon convention places a vertex at angle zero, so the blade edge
    // midpoint is offset by half a sector.
    __host__ __device__ float sector_angle(float theta) const
    {
        float phi = fmodf(theta - rotation, sigma);
        if (phi < 0.0f) phi += sigma;
        return phi - half;
    }

    // Boundary radius at absolute angle theta, in the normalised frame.
    __host__ __device__ float radius_at(float theta) const
    {
        if (blades < 3) return 1.0f;
        return 1.0f / u_raw(sector_angle(theta));
    }

    // First-order distance from a point at normalised (r, theta) to the blade
    // edge bounding it, with the boundary normal there. Positive inside.
    //
    // The constraint curve g = r - R(theta) has polar gradient (1, -R'/r), so
    // the first-order distance to it is (R - r)/sqrt(1 + k^2) with k = R'/r,
    // and its outward normal is (cos + k*sin, sin - k*cos)/sqrt(1 + k^2). A tip
    // facet is a straight chord, so its distance is exact rather than
    // linearised, and the minimum of the two is taken — beyond the facet's own
    // angular span the chord line lies outside the blade edge, so the edge wins
    // and the result stays continuous across the join.
    //
    // Exact on the boundary and first-order with depth. At a tip it is one-sided
    // because it uses the blade belonging to the point's angular sector.
    // Near twisted blade tips this first-order distance can overestimate the
    // true nearest distance; an exact solution requires the conic quartic.
    __host__ __device__ float edge_gap(float r, float theta,
                                       float* nx, float* ny) const
    {
        const float phi = sector_angle(theta);
        const float ct  = cosf(theta), st = sinf(theta);
        float u, du;
        edge_at(phi, &u, &du);
        const float k   = (-du / (u * u)) / r;
        const float inv = 1.0f / sqrtf(1.0f + k * k);

        float best = (1.0f / u - r) * inv;
        *nx = (ct + k * st) * inv;
        *ny = (st - k * ct) * inv;

        if (facets)
        {
            const float ang = fold(phi - half) + half - beta;
            const float d   = p - r * cosf(ang);
            if (d < best)
            {
                best = d;
                *nx  = cosf(theta - ang);
                *ny  = sinf(theta - ang);
            }
        }
        return best;
    }

    // dr/dtheta of the boundary. One-sided at a tip, where the silhouette has a
    // corner.
    __host__ __device__ float dr_dtheta(float theta) const
    {
        if (blades < 3) return 0.0f;
        const float phi = sector_angle(theta);
        float u, du;
        edge_at(phi, &u, &du);
        if (facets)
        {
            const float w = fold(phi - half);
            if (w >= -eA && w <= eB)
            {
                const float uf = cosf(w + half - beta) / p;
                if (uf > u)
                {
                    u  = uf;
                    du = -sinf(w + half - beta) / p;
                }
            }
        }
        return -du / (u * u);
    }
};

// 1 / sin^2(d/2) — the edge's span normaliser (see ApertureProfile::u_edge).
// d is bounded below by MIN_SPAN, so the reciprocal cannot blow up.
inline float ap_inv_sin2_half(float d)
{
    const float s = sinf(0.5f * d);
    return 1.0f / (s * s);
}

// Enclosed-area fraction by Simpson integration over one sector. Accumulate in
// double because the result scales brightness.
inline float ap_area_fraction(const ApertureProfile& pr)
{
    if (pr.blades < 3) return 1.0f;
    const int    n = (AP_AREA_SAMPLES % 2 == 0) ? AP_AREA_SAMPLES : AP_AREA_SAMPLES + 1;
    const double h = (double)pr.sigma / n;
    double total = 0.0;
    for (int i = 0; i <= n; ++i)
    {
        const float  phi = (float)(-(double)pr.half + i * h);
        const double r   = 1.0 / (double)pr.u_raw(phi);
        const double w   = (i == 0 || i == n) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
        total += w * r * r;
    }
    return (float)(pr.blades * (total * h / 3.0) / (2.0 * M_PI));
}

// Build the deterministic derived block used in Surface byte-hashed cache keys.
inline ApertureProfile make_aperture_profile(int blades, float rotation_rad,
                                             const ApertureShapeParams& in)
{
    ApertureProfile pr;
    if (blades < 3) return pr;                 // circular / image stop: unused

    const float curvature = ap_clamp(in.curvature, -1.0f, 1.0f);
    const float twist     = ap_clamp(in.twist,     -1.0f, 1.0f);
    const float notch     = ap_clamp(in.notch, -AP_NOTCH_LIMIT, AP_NOTCH_LIMIT);
    const float rake      = ap_clamp(in.notch_angle, 0.0f, AP_NOTCH_ANGLE_LIMIT);

    pr.blades   = blades;
    pr.rotation = rotation_rad;
    pr.sigma    = 2.0f * (float)M_PI / (float)blades;
    pr.half     = 0.5f * pr.sigma;
    // notch_angle alone is a no-op — it only skews a bite that notch opens —
    // so it does not count as deformation.
    pr.plain    = (curvature == 0.0f && twist == 0.0f && notch == 0.0f) ? 1 : 0;

    const float a = cosf(pr.half);             // plain-polygon apothem
    pr.r_w = (curvature >= 0.0f) ? (a + curvature * (1.0f - a))
                                 : (a * (1.0f + curvature * AP_NEG_DEPTH));

    const float lim = fmaxf(0.0f, pr.half - AP_MIN_SPAN);
    pr.phi_w = ap_clamp(twist * pr.half, -lim, lim);

    const float dR = pr.half - pr.phi_w;       // angular span, waist -> +tip
    const float dL = pr.half + pr.phi_w;       // angular span, waist -> -tip
    pr.u_w    = 1.0f / pr.r_w;
    pr.u_rise = pr.u_w - 1.0f;
    pr.qR     = ap_inv_sin2_half(dR);
    pr.qL     = ap_inv_sin2_half(dL);

    // Tip facet: the CHORD between two points that stay ON the blade edge, one
    // either side of the tip. Skew moves those two points in opposite
    // directions rather than pivoting a free line — an unanchored line raked off
    // the tip direction sweeps inward and slices the whole aperture instead of
    // trimming a corner.
    const float e = (fabsf(notch) / AP_NOTCH_FULL) * AP_NOTCH_MAX * pr.half;
    const float s = ap_clamp(rake / (float)(M_PI / 4.0), 0.0f, 1.0f);
    float eA = fminf(e * (1.0f + s), AP_NOTCH_SECTOR * pr.half);
    float eB = e * (1.0f - s);
    if (notch < 0.0f) { const float t = eA; eA = eB; eB = t; }   // leaning side

    if (e > 1e-9f && pr.r_w < 1.0f && (eA + eB) > 1e-9f)
    {
        const float pa = pr.half - eA;                 // on this sector's edge
        const float ra = 1.0f / pr.u_edge(pa);
        const float pb = pr.half + eB;                 // on the neighbour's edge,
        const float rb = 1.0f / pr.u_edge(-pr.half + eB);   // in its own coords
        const float ax = ra * cosf(pa), ay = ra * sinf(pa);
        const float bx = rb * cosf(pb), by = rb * sinf(pb);
        const float dx = bx - ax,       dy = by - ay;
        const float dlen = sqrtf(dx * dx + dy * dy);
        if (dlen >= 1e-9f)
        {
            float nx = dy / dlen, ny = -dx / dlen;     // chord normal
            if (ax * nx + ay * ny < 0.0f) { nx = -nx; ny = -ny; }   // outward
            const float p = ax * nx + ay * ny;
            if (p > 1e-6f)
            {
                pr.facets = 1;
                pr.p      = p;
                pr.beta   = atan2f(ny, nx);
                pr.eA     = eA;
                pr.eB     = eB;
            }
        }
    }

    // No renormalisation. A facet REMOVES material from a blade tip, so the
    // aperture legitimately gets smaller; rescaling back out to radius 1 would
    // inflate the whole silhouette and flatten the scallops curvature just made.
    // The invariant is max r <= 1, with equality when notch == 0 — which is all
    // the bounding-ellipse fast reject needs, and the true per-axis support
    // extent is measured independently by calibration.
    pr.area_frac = ap_area_fraction(pr);
    return pr;
}
