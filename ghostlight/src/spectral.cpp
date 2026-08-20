// ============================================================================
// spectral.cpp — CPU-only spectral integration: CIE 1931 tables, matrix math,
//                sensor interpolation, and spectral sample table build.
// ============================================================================

#include "spectral.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

// ===========================================================================
// CIE 1931 2° standard observer — 81 entries, 380–780 nm at 5 nm steps.
// Source: CIE publication 15:2004 (Colorimetry, 3rd ed.), Table 1.
// ===========================================================================

static const int CIE_COUNT = 81;

static const float CIE_WAVELENGTHS[CIE_COUNT] = {
    380, 385, 390, 395, 400, 405, 410, 415, 420, 425,
    430, 435, 440, 445, 450, 455, 460, 465, 470, 475,
    480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
    530, 535, 540, 545, 550, 555, 560, 565, 570, 575,
    580, 585, 590, 595, 600, 605, 610, 615, 620, 625,
    630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
    680, 685, 690, 695, 700, 705, 710, 715, 720, 725,
    730, 735, 740, 745, 750, 755, 760, 765, 770, 775,
    780
};

static const float CIE_X[CIE_COUNT] = {
    0.001368f, 0.002236f, 0.004243f, 0.007650f, 0.014310f,
    0.023190f, 0.043510f, 0.077630f, 0.134380f, 0.214770f,
    0.283900f, 0.328500f, 0.348280f, 0.348060f, 0.336200f,
    0.318700f, 0.290800f, 0.251100f, 0.195360f, 0.142100f,
    0.095640f, 0.057950f, 0.032010f, 0.014700f, 0.004900f,
    0.002400f, 0.009300f, 0.029100f, 0.063270f, 0.109600f,
    0.165500f, 0.225750f, 0.290400f, 0.359700f, 0.433450f,
    0.512050f, 0.594500f, 0.678400f, 0.762100f, 0.842500f,
    0.916300f, 0.978600f, 1.026300f, 1.056700f, 1.062200f,
    1.045600f, 1.002600f, 0.938400f, 0.854450f, 0.751400f,
    0.642400f, 0.541900f, 0.447900f, 0.360800f, 0.283500f,
    0.218700f, 0.164900f, 0.121200f, 0.087400f, 0.063600f,
    0.046770f, 0.032900f, 0.022700f, 0.015840f, 0.011359f,
    0.008111f, 0.005790f, 0.004109f, 0.002899f, 0.002049f,
    0.001440f, 0.001000f, 0.000690f, 0.000476f, 0.000332f,
    0.000235f, 0.000166f, 0.000117f, 0.000083f, 0.000059f,
    0.000042f
};

static const float CIE_Y[CIE_COUNT] = {
    0.000039f, 0.000064f, 0.000120f, 0.000217f, 0.000396f,
    0.000640f, 0.001210f, 0.002180f, 0.004000f, 0.007300f,
    0.011600f, 0.016840f, 0.023000f, 0.029800f, 0.038000f,
    0.048000f, 0.060000f, 0.073900f, 0.090980f, 0.112600f,
    0.139020f, 0.169300f, 0.208020f, 0.258600f, 0.323000f,
    0.407300f, 0.503000f, 0.608200f, 0.710000f, 0.793200f,
    0.862000f, 0.914850f, 0.954000f, 0.980300f, 0.994950f,
    1.000000f, 0.995000f, 0.978600f, 0.952000f, 0.915400f,
    0.870000f, 0.816300f, 0.757000f, 0.694900f, 0.631000f,
    0.566800f, 0.503000f, 0.441200f, 0.381000f, 0.321000f,
    0.265000f, 0.217000f, 0.175000f, 0.138200f, 0.107000f,
    0.081600f, 0.061000f, 0.044580f, 0.032000f, 0.023200f,
    0.017000f, 0.011920f, 0.008210f, 0.005723f, 0.004102f,
    0.002929f, 0.002091f, 0.001484f, 0.001047f, 0.000740f,
    0.000520f, 0.000361f, 0.000249f, 0.000172f, 0.000120f,
    0.000085f, 0.000060f, 0.000042f, 0.000030f, 0.000021f,
    0.000015f
};

static const float CIE_Z[CIE_COUNT] = {
    0.006450f, 0.010550f, 0.020050f, 0.036210f, 0.067850f,
    0.110200f, 0.207400f, 0.371300f, 0.645600f, 1.039050f,
    1.385600f, 1.622960f, 1.747060f, 1.782600f, 1.772110f,
    1.744100f, 1.669200f, 1.528100f, 1.287640f, 1.041900f,
    0.812950f, 0.616200f, 0.465180f, 0.353300f, 0.272000f,
    0.212300f, 0.158200f, 0.111700f, 0.078250f, 0.057250f,
    0.042160f, 0.029840f, 0.020300f, 0.013400f, 0.008750f,
    0.005750f, 0.003900f, 0.002750f, 0.002100f, 0.001800f,
    0.001650f, 0.001400f, 0.001100f, 0.001000f, 0.000800f,
    0.000600f, 0.000340f, 0.000240f, 0.000190f, 0.000100f,
    0.000050f, 0.000030f, 0.000020f, 0.000010f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f
};

// ===========================================================================
// Static CIE 1931 profile (returned by get_sensor_profile)
// ===========================================================================

static const SensorProfile s_cie1931 = {
    "CIE 1931 2\xc2\xb0 Observer",  // UTF-8 degree sign
    CIE_WAVELENGTHS,
    CIE_X, CIE_Y, CIE_Z,
    CIE_COUNT,
    // matrix_to_xyz = identity (CIE CMFs are already XYZ)
    { {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f} }
};

const SensorProfile& get_sensor_profile(SensorModel model)
{
    (void)model;  // CIE 1931 is the only built-in profile.
    return s_cie1931;
}

// ===========================================================================
// sensor_interpolate — linear interpolation into any SensorProfile
// ===========================================================================

void sensor_interpolate(const SensorProfile& prof, float lambda_nm,
                        float* sr, float* sg, float* sb)
{
    if (prof.count <= 0) { *sr = *sg = *sb = 0.0f; return; }

    const float lam_min = prof.wavelengths[0];
    const float lam_max = prof.wavelengths[prof.count - 1];

    if (lambda_nm < lam_min || lambda_nm > lam_max) {
        *sr = *sg = *sb = 0.0f;
        return;
    }

    // Binary search for the interval
    int lo = 0, hi = prof.count - 2;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (prof.wavelengths[mid + 1] < lambda_nm)
            lo = mid + 1;
        else
            hi = mid;
    }

    const float w0 = prof.wavelengths[lo];
    const float w1 = prof.wavelengths[lo + 1];
    const float t  = (lambda_nm - w0) / (w1 - w0);

    *sr = prof.sens_r[lo] + t * (prof.sens_r[lo + 1] - prof.sens_r[lo]);
    *sg = prof.sens_g[lo] + t * (prof.sens_g[lo + 1] - prof.sens_g[lo]);
    *sb = prof.sens_b[lo] + t * (prof.sens_b[lo + 1] - prof.sens_b[lo]);
}

// ===========================================================================
// 3×3 matrix helpers
// ===========================================================================

static void mat3_identity(float m[3][3])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m[r][c] = (r == c) ? 1.0f : 0.0f;
}

static void mat3_copy(const float src[3][3], float dst[3][3])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            dst[r][c] = src[r][c];
}

// dst = A * B
static void mat3_mul(const float A[3][3], const float B[3][3], float dst[3][3])
{
    float tmp[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                tmp[r][c] += A[r][k] * B[k][c];
    mat3_copy(tmp, dst);
}

// ===========================================================================
// Precomputed matrices used by resolve_output_matrix()
//
// CIE 1931 CMFs are tabulated under equal-energy (E) normalization.
// For D60-referenced output spaces we compose the Bradford E→D60 chromatic
// adaptation before the XYZ→output matrix.  D65-referenced spaces use the
// standard XYZ→output matrix directly (CIE E normalization aligns with D65
// conventions for sRGB / P3-D65).
// ===========================================================================

// ACEScg (AP1, D60) → CIE XYZ — inverse of M_XYZ_TO_AP1.
// From ACES CTL reference (AMF-0-2008-001).
static const float M_AP1_TO_XYZ[3][3] = {
    { 0.66246f,  0.13400f,  0.15618f},
    { 0.27223f,  0.67408f,  0.05369f},
    {-0.00557f,  0.00406f,  1.01034f}
};

// sRGB linear (D65) → CIE XYZ — IEC 61966-2-1.
static const float M_SRGB_TO_XYZ[3][3] = {
    {0.4124f, 0.3576f, 0.1805f},
    {0.2126f, 0.7152f, 0.0722f},
    {0.0193f, 0.1192f, 0.9505f}
};

// ===========================================================================
// resolve_input_matrix
// ===========================================================================

void resolve_input_matrix(InputColorSpace cs,
                          const float     custom_input_to_xyz[3][3],
                          float           out[3][3])
{
    switch (cs)
    {
    case CS_INPUT_ACESCG:
        mat3_copy(M_AP1_TO_XYZ, out);
        break;
    case CS_INPUT_SRGB_LINEAR:
        mat3_copy(M_SRGB_TO_XYZ, out);
        break;
    case CS_INPUT_CUSTOM:
        if (custom_input_to_xyz)
            mat3_copy(custom_input_to_xyz, out);
        else
            mat3_identity(out);
        break;
    default:
        mat3_identity(out);
        break;
    }
}

// Bradford chromatic adaptation: CIE equal-energy E → D60
// Derived from M_Bradford and D60 white XYZ = [0.95045, 1.0, 1.08906].
static const float M_BRADFORD_E_TO_D60[3][3] = {
    { 0.95307f, -0.02663f,  0.02375f},
    {-0.03845f,  1.02927f,  0.00937f},
    { 0.00258f, -0.00298f,  1.08912f}
};

// XYZ → ACEScg (AP1 primaries, D60 white) — ACES CTL reference
static const float M_XYZ_TO_AP1[3][3] = {
    { 1.6410f, -0.3248f, -0.2358f},
    {-0.6636f,  1.6153f,  0.0167f},
    { 0.0117f, -0.0082f,  0.9883f}
};

// XYZ → sRGB linear (IEC 61966-2-1, D65)
static const float M_XYZ_TO_SRGB[3][3] = {
    { 3.2406f, -1.5372f, -0.4986f},
    {-0.9689f,  1.8758f,  0.0415f},
    { 0.0557f, -0.2040f,  1.0570f}
};

// XYZ → DCI-P3 D65
static const float M_XYZ_TO_P3D65[3][3] = {
    { 2.4935f, -0.9314f, -0.4027f},
    {-0.8295f,  1.7627f,  0.0236f},
    { 0.0358f, -0.0761f,  0.9569f}
};

// XYZ → DCI-P3 D60 (P3 primaries, D60 white — theatrical DCI/ACES)
// Derived from P3 primaries under D60 white point.
static const float M_XYZ_TO_P3D60[3][3] = {
    { 2.4932f, -0.9316f, -0.4027f},
    {-0.8320f,  1.7648f,  0.0232f},
    { 0.0363f, -0.0765f,  0.9568f}
};

// ===========================================================================
// resolve_output_matrix
// ===========================================================================

void resolve_output_matrix(OutputColorSpace cs,
                           const float      custom_xyz_to_output[3][3],
                           float            out[3][3])
{
    switch (cs)
    {
    case CS_ACESCG:
        // XYZ-E → D60 → AP1
        mat3_mul(M_XYZ_TO_AP1, M_BRADFORD_E_TO_D60, out);
        break;

    case CS_SRGB_LINEAR:
        mat3_copy(M_XYZ_TO_SRGB, out);
        break;

    case CS_P3_D65:
        mat3_copy(M_XYZ_TO_P3D65, out);
        break;

    case CS_P3_D60:
        // XYZ-E → D60 → P3-D60
        mat3_mul(M_XYZ_TO_P3D60, M_BRADFORD_E_TO_D60, out);
        break;

    case CS_XYZ:
        mat3_identity(out);
        break;

    case CS_CUSTOM:
        if (custom_xyz_to_output)
            mat3_copy(custom_xyz_to_output, out);
        else
            mat3_identity(out);
        break;

    default:
        mat3_identity(out);
        break;
    }
}

// ===========================================================================
// build_spectral_table
// ===========================================================================

std::vector<GPUSpectralSample> build_spectral_table(
    int                  n_samples,
    float                lambda_min,
    float                lambda_max,
    const SensorProfile& sensor,
    const float          M_xyz_to_output[3][3])
{
    std::vector<GPUSpectralSample> table(n_samples);
    const float delta = (lambda_max - lambda_min) / n_samples;

    const auto& Msen = sensor.matrix_to_xyz;

    for (int i = 0; i < n_samples; ++i)
    {
        float lam = lambda_min + (i + 0.5f) * delta;
        table[i].lambda = lam;

        // Interpolate sensor sensitivity curves at this wavelength.
        float sr, sg, sb;
        sensor_interpolate(sensor, lam, &sr, &sg, &sb);

        // sensor-native → XYZ
        float xr = Msen[0][0]*sr + Msen[0][1]*sg + Msen[0][2]*sb;
        float xg = Msen[1][0]*sr + Msen[1][1]*sg + Msen[1][2]*sb;
        float xb = Msen[2][0]*sr + Msen[2][1]*sg + Msen[2][2]*sb;

        // XYZ → output_cs
        table[i].cmf_r = M_xyz_to_output[0][0]*xr + M_xyz_to_output[0][1]*xg + M_xyz_to_output[0][2]*xb;
        table[i].cmf_g = M_xyz_to_output[1][0]*xr + M_xyz_to_output[1][1]*xg + M_xyz_to_output[1][2]*xb;
        table[i].cmf_b = M_xyz_to_output[2][0]*xr + M_xyz_to_output[2][1]*xg + M_xyz_to_output[2][2]*xb;
    }

    // Normalize: trapezoid integration of cmf_g → 1.
    // Apply the same scalar to cmf_r and cmf_b.
    float sum_g = 0.0f;
    for (int i = 0; i < n_samples; ++i)
        sum_g += table[i].cmf_g * delta;

    if (sum_g > 1e-9f)
    {
        float inv = 1.0f / sum_g;
        for (auto& s : table) {
            s.cmf_r *= inv;
            s.cmf_g *= inv;
            s.cmf_b *= inv;
        }
    }

    return table;
}

// ===========================================================================
// compute_spectral_correction — output-space colour/brightness correction
// ===========================================================================

// Reference sampling for the "converged" colour operator (~1 nm over 400-700).
static constexpr int SPECTRAL_DENSE_SAMPLES  = 300;
// Brightness anchor for corrected renders.
static constexpr int SPECTRAL_ANCHOR_SAMPLES = 16;

// CMF Gram operator O[k][j] = Σ_i cmf_k(λ_i) · cmf_j(λ_i) for a spectral table.
// This is what the ghost kernel effectively applies to source RGB (per unit
// λ-independent transmission): output = O · source.
static void spectral_gram(const std::vector<GPUSpectralSample>& t, float O[3][3])
{
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            O[a][b] = 0.0f;
    for (const auto& s : t) {
        const float c[3] = { s.cmf_r, s.cmf_g, s.cmf_b };
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                O[a][b] += c[a] * c[b];
    }
}

// Green output for a white (1,1,1) source = row-1 sum. Brightness proxy.
static float spectral_white_green(const float O[3][3])
{
    return O[1][0] + O[1][1] + O[1][2];
}

static bool mat3_inverse(const float m[3][3], float inv[3][3])
{
    const float det =
        m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
      - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
      + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (std::fabs(det) < 1e-20f) { mat3_identity(inv); return false; }
    const float id = 1.0f / det;
    inv[0][0] =  (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * id;
    inv[0][1] = -(m[0][1]*m[2][2] - m[0][2]*m[2][1]) * id;
    inv[0][2] =  (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * id;
    inv[1][0] = -(m[1][0]*m[2][2] - m[1][2]*m[2][0]) * id;
    inv[1][1] =  (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * id;
    inv[1][2] = -(m[0][0]*m[1][2] - m[0][2]*m[1][0]) * id;
    inv[2][0] =  (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * id;
    inv[2][1] = -(m[0][0]*m[2][1] - m[0][1]*m[2][0]) * id;
    inv[2][2] =  (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * id;
    return true;
}

void compute_spectral_correction(
    const std::vector<GPUSpectralSample>& table_n,
    float                lambda_min,
    float                lambda_max,
    const SensorProfile& sensor,
    const float          M_xyz_to_output[3][3],
    float                C_out[3][3])
{
    // Operators: kernel's n-sample, dense reference (converged colour), anchor
    // (brightness reference). Dense + anchor share the n-table's λ range/output.
    float O_n[3][3], O_d[3][3], O_a[3][3];
    spectral_gram(table_n, O_n);
    spectral_gram(build_spectral_table(SPECTRAL_DENSE_SAMPLES,  lambda_min, lambda_max, sensor, M_xyz_to_output), O_d);
    spectral_gram(build_spectral_table(SPECTRAL_ANCHOR_SAMPLES, lambda_min, lambda_max, sensor, M_xyz_to_output), O_a);

    const float bn = spectral_white_green(O_n);
    const float bd = spectral_white_green(O_d);
    const float ba = spectral_white_green(O_a);
    if (!(bn > 1e-20f) || !(bd > 1e-20f)) { mat3_identity(C_out); return; }

    // Luminance-normalized operators isolate the colour rotation from brightness.
    float O_n_hat[3][3], O_d_hat[3][3];
    const float inv_bn = 1.0f / bn, inv_bd = 1.0f / bd;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            O_n_hat[a][b] = O_n[a][b] * inv_bn;
            O_d_hat[a][b] = O_d[a][b] * inv_bd;
        }

    // C = (ba/bn) · (Ô_dense · Ô_n⁻¹).  Then C·O_n = ba·Ô_dense: every sample
    // count -> converged colour at the anchor brightness.  s -> 1 at n == anchor;
    // the colour rotation -> identity as n grows.
    float O_n_hat_inv[3][3], C_col[3][3];
    mat3_inverse(O_n_hat, O_n_hat_inv);
    mat3_mul(O_d_hat, O_n_hat_inv, C_col);

    const float s = ba / bn;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            C_out[a][b] = C_col[a][b] * s;
}
