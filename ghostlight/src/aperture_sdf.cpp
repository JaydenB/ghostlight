// ============================================================================
// aperture_sdf.cpp — see aperture_sdf.h.
//
// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher 2004, the
// O(n) lower-envelope-of-parabolas method), run once toward the blocked set and
// once toward the pass set to form a signed field. 1D passes are parameterised
// by physical texel spacing (sx along rows, sy down columns) so the result is
// world-mm distance directly, matching the trace UV convention.
// ============================================================================

#include "aperture_sdf.h"

#include <cmath>
#include <vector>

namespace {

// "Infinity" for the DT seed. Any value far above the largest real squared
// distance (image diagonal^2, ~1e3 mm^2 for a big fast lens) reads as
// unreachable, while staying small enough that the a*q^2 parabola term survives
// float addition (an actual FLT_MAX would swamp it via catastrophic cancellation
// in the intersection math). 1e10 mm^2 -> ~1e5 mm distance -> sigma underflow.
constexpr float SDF_INF = 1.0e10f;

// 1D squared-distance transform of `f` (length n) under parabola coefficient
// a = spacing^2, writing squared distances into `d`. `v`/`z` are scratch of
// length n and n+1. Intersection math in double to avoid the a-scaled
// cancellation near SDF_INF.
void dt_1d(const float* f, float* d, int n, float a,
           int* v, double* z)
{
    int k = 0;
    v[0] = 0;
    z[0] = -1e300;
    z[1] = +1e300;
    for (int q = 1; q < n; ++q)
    {
        double s;
        while (true)
        {
            const int p = v[k];
            s = (( (double)f[q] + (double)a * (double)q * q)
               - ( (double)f[p] + (double)a * (double)p * p))
                / (2.0 * (double)a * (double)(q - p));
            if (s <= z[k]) --k;
            else break;
        }
        ++k;
        v[k]     = q;
        z[k]     = s;
        z[k + 1] = +1e300;
    }
    k = 0;
    for (int q = 0; q < n; ++q)
    {
        while (z[k + 1] < (double)q) ++k;
        const int   p  = v[k];
        const float dx = (float)(q - p);
        d[q] = a * dx * dx + f[p];
    }
}

// Full 2D EDT (squared, physical units) of the seed field `seed` (0 on target
// texels, SDF_INF elsewhere). Column pass (y, spacing sy) then row pass (x,
// spacing sx); the two compose because seeds are squared physical distances.
void edt_2d(std::vector<float>& seed, int W, int H, float sx, float sy)
{
    const float ax = sx * sx;
    const float ay = sy * sy;

    std::vector<int>    v(W > H ? W : H);
    std::vector<double> z((W > H ? W : H) + 1);
    std::vector<float>  colf(H), cold(H);

    // Pass 1: down each column (Y).
    for (int i = 0; i < W; ++i)
    {
        for (int j = 0; j < H; ++j) colf[j] = seed[(size_t)j * W + i];
        dt_1d(colf.data(), cold.data(), H, ay, v.data(), z.data());
        for (int j = 0; j < H; ++j) seed[(size_t)j * W + i] = cold[j];
    }
    // Pass 2: along each row (X), using pass-1 output as f.
    std::vector<float> rowd(W);
    for (int j = 0; j < H; ++j)
    {
        float* row = &seed[(size_t)j * W];
        dt_1d(row, rowd.data(), W, ax, v.data(), z.data());
        for (int i = 0; i < W; ++i) row[i] = rowd[i];
    }
}

} // namespace

int find_sdf_target_surface(const OpticalSystem& lens)
{
    const int n = (int)lens.surfaces.size();
    for (int i = 0; i < n; ++i)
    {
        const Surface& s = lens.surfaces[i];
        if (!s.is_stop)                          continue;
        if (s.aperture_shape != APERTURE_IMAGE)  return -1;   // stop isn't image
        if (i >= (int)lens.aperture_images.size()) return -1;
        const ApertureImage& img = lens.aperture_images[i];
        if (img.pixels.empty() || img.width <= 0 || img.height <= 0) return -1;
        return i;
    }
    return -1;
}

bool bake_aperture_sdf(const ApertureImage& img, float semi_diameter_mm,
                       float aspect, ApertureSdfBake& out)
{
    out = ApertureSdfBake{};

    const int W = img.width;
    const int H = img.height;
    if (W <= 0 || H <= 0 || (int)img.pixels.size() < W * H) return false;
    if (semi_diameter_mm <= 0.0f) return false;
    if (aspect <= 0.0f) aspect = 1.0f;

    // World mm per texel — matches d_image_aperture_passes UV:
    //   u = 0.5 + (x/aspect)/(2*sd)  ->  x spans [-aspect*sd, +aspect*sd] over W
    //   v = 0.5 + y/(2*sd)           ->  y spans [-sd, +sd] over H
    const float sx = 2.0f * aspect * semi_diameter_mm / (float)W;
    const float sy = 2.0f * semi_diameter_mm / (float)H;

    // Binary inside mask (pass region), matching the trace threshold.
    std::vector<unsigned char> inside((size_t)W * H);
    for (size_t p = 0; p < (size_t)W * H; ++p)
        inside[p] = (img.pixels[p] > 0.5f) ? 1u : 0u;

    // Seed toward the blocked set, then toward the pass set.
    std::vector<float> d_to_blocked((size_t)W * H);
    std::vector<float> d_to_pass((size_t)W * H);
    for (size_t p = 0; p < (size_t)W * H; ++p)
    {
        d_to_blocked[p] = inside[p] ? SDF_INF : 0.0f;   // 0 on blocked
        d_to_pass[p]    = inside[p] ? 0.0f : SDF_INF;    // 0 on pass
    }
    edt_2d(d_to_blocked, W, H, sx, sy);
    edt_2d(d_to_pass,    W, H, sx, sy);

    // Signed field (positive inside), raw center-to-center magnitude.
    std::vector<float> signed_d((size_t)W * H);
    for (size_t p = 0; p < (size_t)W * H; ++p)
        signed_d[p] = inside[p] ? sqrtf(d_to_blocked[p]) : -sqrtf(d_to_pass[p]);

    out.width  = W;
    out.height = H;
    out.sx     = sx;
    out.sy     = sy;
    out.texels.assign((size_t)W * H * 4, 0.0f);

    auto at = [&](int i, int j) -> float {
        i = i < 0 ? 0 : (i >= W ? W - 1 : i);
        j = j < 0 ? 0 : (j >= H ? H - 1 : j);
        return signed_d[(size_t)j * W + i];
    };

    for (int j = 0; j < H; ++j)
    for (int i = 0; i < W; ++i)
    {
        // Central-difference gradient of the signed field (world units).
        float gx = (at(i + 1, j) - at(i - 1, j)) / (2.0f * sx);
        float gy = (at(i, j + 1) - at(i, j - 1)) / (2.0f * sy);
        float gl = sqrtf(gx * gx + gy * gy);
        float nx, ny;
        if (gl < 1.0e-12f) { nx = 1.0f; ny = 0.0f; }     // ridge fallback
        else               { nx = gx / gl; ny = gy / gl; }

        // Half-texel correction: raw EDT is center-to-center, but the true 0.5
        // boundary lies ~half a texel inward. Shift the magnitude toward 0 by
        // the texel half-extent projected on the normal (exact for axis-aligned
        // edges, first-order otherwise). Makes the zero-level coincide with the
        // matte clip boundary.
        const float c   = 0.5f * sqrtf((nx * sx) * (nx * sx) + (ny * sy) * (ny * sy));
        float       mag = fabsf(signed_d[(size_t)j * W + i]) - c;
        if (mag < 0.0f) mag = 0.0f;
        const float sd  = (signed_d[(size_t)j * W + i] >= 0.0f) ? mag : -mag;

        float* t = &out.texels[((size_t)j * W + i) * 4];
        t[0] = sd;
        t[1] = nx;
        t[2] = ny;
        t[3] = 0.0f;
    }

    return true;
}
