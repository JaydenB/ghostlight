// ============================================================================
// fresnel.h — Fresnel equations, thin-film AR coatings, glass dispersion
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

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Dispersion (Cauchy via Abbe number) --------------------------------

// Compute wavelength-dependent IOR from the d-line IOR and Abbe number.
// Reference wavelengths: F = 486.13 nm, d = 587.56 nm, C = 656.27 nm
__host__ __device__ inline float dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f)
        return n_d; // air or non-dispersive

    constexpr float lF = 486.13f;
    constexpr float lC = 656.27f;
    constexpr float ld = 587.56f;

    // n_F - n_C = (n_d - 1) / V_d
    float dn = (n_d - 1.0f) / V_d;

    // Cauchy: n(λ) = A + B/λ²
    float inv_lF2 = 1.0f / (lF * lF);
    float inv_lC2 = 1.0f / (lC * lC);
    float inv_ld2 = 1.0f / (ld * ld);

    float B = dn / (inv_lF2 - inv_lC2);
    float A = n_d - B * inv_ld2;

    return A + B / (lambda_nm * lambda_nm);
}

// Sellmeier dispersion: n²(λ) = 1 + Σ B[i]λ²/(λ²−C[i])
// B and C are the three-term Sellmeier coefficients stored on Surface.
// lambda_nm: wavelength in nm (converted internally to μm for the formula).
__host__ __device__ inline float sellmeier_n(const float B[3], const float C[3],
                                              float lambda_nm)
{
    float l  = lambda_nm * 1e-3f;   // nm → μm
    float l2 = l * l;
    float n2 = 1.0f;
    n2 += B[0] * l2 / (l2 - C[0]);
    n2 += B[1] * l2 / (l2 - C[1]);
    n2 += B[2] * l2 / (l2 - C[2]);
    return (n2 > 0.0f) ? sqrtf(n2) : 1.0f;  // guard: n² can go negative outside valid range
}

// ---- Fresnel equations --------------------------------------------------

// Unpolarized Fresnel reflectance at a dielectric interface.
// cos_i: cosine of incidence angle (positive).
__host__ __device__ inline float fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i = fabsf(cos_i);
    float eta = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);

    if (sin2_t >= 1.0f)
        return 1.0f; // total internal reflection

    float cos_t = sqrtf(1.0f - sin2_t);

    float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    float rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);

    return 0.5f * (rs * rs + rp * rp);
}

// ---- Thin-film AR coating -----------------------------------------------

// Single-layer coating reflectance (thin-film interference).
// coating_n : coating refractive index (e.g. MgF2 = 1.38)
// d_nm      : physical coating thickness in nm
// lambda_nm : light wavelength in nm
__host__ __device__ inline float coating_reflectance(float cos_i, float n1, float n2,
                                                     float coating_n, float d_nm,
                                                     float lambda_nm)
{
    // Snell into coating
    float sin2_c = (n1 / coating_n) * (n1 / coating_n) * (1.0f - cos_i * cos_i);
    if (sin2_c >= 1.0f)
        return fresnel_reflectance(cos_i, n1, n2);
    float cos_c = sqrtf(1.0f - sin2_c);

    // Phase thickness (double pass through the film)
    float delta = 2.0f * (float)M_PI * coating_n * d_nm * cos_c / lambda_nm;

    // Fresnel amplitude coefficients at each interface
    float r01 = (n1 * cos_i - coating_n * cos_c) / (n1 * cos_i + coating_n * cos_c);

    float sin2_2 = (coating_n / n2) * (coating_n / n2) * (1.0f - cos_c * cos_c);
    if (sin2_2 >= 1.0f)
        return fresnel_reflectance(cos_i, n1, n2);
    float cos_2 = sqrtf(1.0f - sin2_2);
    float r12 = (coating_n * cos_c - n2 * cos_2) / (coating_n * cos_c + n2 * cos_2);

    // Airy formula for total reflectance
    float cos_2delta = cosf(2.0f * delta);
    float num = r01 * r01 + r12 * r12 + 2.0f * r01 * r12 * cos_2delta;
    float den = 1.0f + r01 * r01 * r12 * r12 + 2.0f * r01 * r12 * cos_2delta;

    float R = num / den;
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

// ---- Artist coating -------------------------------------------------------

// Artist-driven coating: smooth, plausible R(λ) synthesized from an RGB tint
// and a strength.  Three Gaussian basis functions centred on the red/green/
// blue thirds of the visible range are blended by the tint and normalized as
// a partition of unity:
//
//   R(λ) = strength · (r·B650(λ) + g·B550(λ) + b·B450(λ)) / (B650+B550+B450)
//
// so a white tint (1,1,1) gives R(λ) ≡ strength exactly and a pure hue gives
// a smooth bump peaking in that band.  Angle-independent by design — artists
// get the same ghost tint regardless of incidence.  Result clamped to [0,1].
__host__ __device__ inline float artist_reflectance(float tint_r, float tint_g,
                                                    float tint_b, float strength,
                                                    float lambda_nm)
{
    constexpr float inv_2s2 = 1.0f / (2.0f * 65.0f * 65.0f); // σ = 65 nm

    float dr = lambda_nm - 650.0f;
    float dg = lambda_nm - 550.0f;
    float db = lambda_nm - 450.0f;

    float br = expf(-dr * dr * inv_2s2);
    float bg = expf(-dg * dg * inv_2s2);
    float bb = expf(-db * db * inv_2s2);

    float sum = br + bg + bb; // > 0 for any finite λ
    float R   = strength * (tint_r * br + tint_g * bg + tint_b * bb) / sum;

    return fminf(fmaxf(R, 0.0f), 1.0f);
}

// ---- Combined surface reflectance ---------------------------------------

// Returns Fresnel reflectance at a lens surface, accounting for AR coating.
// coating_layers: 0 = uncoated, 1 = single-layer MgF2, 2+ ≈ multi-coat
__host__ __device__ inline float surface_reflectance(float cos_i, float n1, float n2,
                                                     int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return fresnel_reflectance(cos_i, n1, n2);

    // MgF2 single-layer: n=1.38, quarter-wave thickness at 550 nm
    constexpr float mgf2_n = 1.38f;
    constexpr float design_lambda = 550.0f;               // nm
    float qw_thickness = design_lambda / (4.0f * mgf2_n); // ≈ 99.6 nm

    float R = coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thickness, lambda_nm);

    // Multi-layer coatings give progressively lower reflectance
    // (very rough approximation — real multi-layer stacks are more complex)
    for (int i = 1; i < coating_layers; ++i)
        R *= 0.25f;

    return fminf(fmaxf(R, 0.0f), 1.0f);
}
