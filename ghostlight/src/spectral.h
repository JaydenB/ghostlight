// ============================================================================
// spectral.h — Spectral path tracing: sensor profiles, color space conversion,
//              and GPU spectral sample table construction.
//
// GPUSpectralSample is shared between CPU and GPU — it is the struct uploaded
// to the device.  All other declarations (enums, SensorProfile, build functions)
// are CPU-only: they use STL and are not safe to compile as device code.
// ============================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Color space enums
// ---------------------------------------------------------------------------

enum InputColorSpace {
    CS_INPUT_ACESCG,       // ACES AP1, D60 (default)
    CS_INPUT_SRGB_LINEAR,  // sRGB / Rec.709 linear, D65
    CS_INPUT_CUSTOM,       // user-supplied matrix in RenderConfig::custom_input_to_xyz
};

enum OutputColorSpace {
    CS_ACESCG,        // ACES AP1, D60 (default)
    CS_SRGB_LINEAR,   // sRGB / Rec.709 linear, D65
    CS_P3_D65,        // DCI-P3 D65
    CS_P3_D60,        // DCI-P3 D60 (theatrical DCI)
    CS_XYZ,           // CIE XYZ pass-through (debug / custom downstream)
    CS_CUSTOM,        // user-supplied matrix in RenderConfig::custom_xyz_to_output
};

// ---------------------------------------------------------------------------
// Sensor model enum
// ---------------------------------------------------------------------------

enum SensorModel {
    SENSOR_CIE_1931,  // human observer (x̄ȳz̄); matrix_to_xyz = identity
};

// ---------------------------------------------------------------------------
// GPU spectral sample — one entry in the per-wavelength table uploaded to the
// device.  cmf_r/g/b are already baked to output_cs (after sensor → XYZ →
// output_cs matrix chain and trapezoid normalization).
// ---------------------------------------------------------------------------
struct GPUSpectralSample
{
    float lambda;               // nm — passed to Sellmeier / dispersion IOR
    float cmf_r, cmf_g, cmf_b; // output-space sensor weights at λ
};

// ---------------------------------------------------------------------------
// SensorProfile — describes a spectral sensor (human observer, camera, film).
//
// For SENSOR_CIE_1931: sens_r/g/b = x̄/ȳ/z̄ and matrix_to_xyz = identity.
// For cameras/film: sens_r/g/b = measured channel sensitivities in
//   sensor-native space; matrix_to_xyz converts native → CIE XYZ.
//   Populate from published spectral-response measurements, not synthetic curves.
// ---------------------------------------------------------------------------
struct SensorProfile
{
    const char*  name;               // display name
    const float* wavelengths;        // nm, ascending, count entries
    const float* sens_r;             // channel R spectral sensitivity
    const float* sens_g;             // channel G spectral sensitivity
    const float* sens_b;             // channel B spectral sensitivity
    int          count;              // number of tabulated entries
    float        matrix_to_xyz[3][3]; // sensor-native → CIE XYZ (identity for CIE 1931)
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Return the built-in CIE 1931 2° observer profile.
// The returned reference is statically allocated; do not free.
const SensorProfile& get_sensor_profile(SensorModel model);

// Linear interpolation into a SensorProfile's sensitivity curves at λ nm.
// Clamps to zero outside the profile's wavelength range.
void sensor_interpolate(const SensorProfile& prof, float lambda_nm,
                        float* sr, float* sg, float* sb);

// Build the M_input_to_xyz 3×3 matrix for a given input color space.
// This is applied to source RGB before spectral upsampling so that the kernel
// always receives values in a colorimetrically meaningful coordinate system.
// Returns identity for CS_INPUT_CUSTOM when custom_input_to_xyz is null.
void resolve_input_matrix(InputColorSpace cs,
                          const float     custom_input_to_xyz[3][3],
                          float           out[3][3]);

// Build the per-wavelength M_xyz_to_output 3×3 matrix for a given output color
// space.  Composes Bradford E→D60 for D60-referenced spaces (CS_ACESCG,
// CS_P3_D60); uses the output matrix directly for D65 spaces; returns identity
// for CS_XYZ; copies custom_xyz_to_output for CS_CUSTOM.
void resolve_output_matrix(OutputColorSpace cs,
                           const float      custom_xyz_to_output[3][3],
                           float            out[3][3]);

// Build the spectral sample table from a sensor profile and output matrix.
// n_samples wavelengths are spaced uniformly in [lambda_min, lambda_max].
// The green channel is trapezoid-normalized to ∫ cmf_g dλ = 1.
std::vector<GPUSpectralSample> build_spectral_table(
    int                  n_samples,
    float                lambda_min,
    float                lambda_max,
    const SensorProfile& sensor,
    const float          M_xyz_to_output[3][3]);

// Compute the spectral colour/brightness correction 3×3.
//
// The ghost kernel accumulates, per unit source under λ-independent transmission,
// output = O_n · source where O_n = Σ_i cmf(λ_i)⊗cmf(λ_i) is the CMF Gram matrix
// of the n-sample table (table_n).  O_n's colour balance and overall scale drift
// with the sample count.  This returns C such that applying it to the accumulated
// ghost buffer maps every sample count to the SAME image: converged colour (from a
// dense ~1 nm reference table) at a fixed reference brightness (the Gram of an
// anchor-count table, SPECTRAL_ANCHOR_SAMPLES).  C -> identity as n -> anchor and
// its colour part -> 0 as n grows, so high-sample renders are ~untouched.
//
// CMF-only (no lens transmission), so it removes just the CMF-quadrature artifact
// and leaves physical dispersion tint intact.  table_n is the SAME table the kernel
// uploaded; lambda_min/max/sensor/M_out rebuild the dense + anchor references.
void compute_spectral_correction(
    const std::vector<GPUSpectralSample>& table_n,
    float                lambda_min,
    float                lambda_max,
    const SensorProfile& sensor,
    const float          M_xyz_to_output[3][3],
    float                C_out[3][3]);
