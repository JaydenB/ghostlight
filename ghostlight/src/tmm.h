// ============================================================================
// tmm.h — Transfer Matrix Method for thin-film coating stacks (host-only)
//
// Evaluates the unpolarized reflectance of an ambient → layer stack →
// substrate system at a given wavelength and angle of incidence using the
// standard 2×2 characteristic-matrix formulation (e.g. Macleod, "Thin-Film
// Optical Filters").  Used ONLY at lens-load / edit time to bake physical
// layer stacks into SPECTRAL_ANGULAR lookup tables — never inside the
// tracer.  Host-only on purpose: uses std::complex, which we keep out of
// CUDA translation units.  Include from .cpp files only.
// ============================================================================
#pragma once

#include <complex>
#include <vector>
#include <cmath>

namespace tmm {

struct Layer
{
    std::complex<float> n;            // complex refractive index n + ik
    float               thickness_nm = 0.0f;
};

namespace detail {

using cplx = std::complex<float>;

// cos(θ) inside a medium of index n for a given transverse invariant
// n0·sin(θ0).  Complex square root picks the decaying branch automatically
// for absorbing media / evanescent waves.
inline cplx cos_theta_in(const cplx& n, float invariant)
{
    cplx s = cplx(invariant, 0.0f) / n;
    cplx c = std::sqrt(cplx(1.0f, 0.0f) - s * s);
    // Enforce the physical branch: Im(n·cosθ) >= 0 (decaying into the medium).
    if ((n * c).imag() < 0.0f) c = -c;
    return c;
}

// Amplitude reflectance of the full stack for one polarization.
// eta values are the "tilted admittances": n·cosθ for s-pol, n/cosθ for p-pol.
inline cplx stack_amplitude(const std::vector<Layer>& layers,
                            const cplx& n_ambient, const cplx& n_substrate,
                            float lambda_nm, float invariant, bool s_pol)
{
    const cplx one(1.0f, 0.0f);

    auto eta_of = [&](const cplx& n) -> cplx {
        cplx c = cos_theta_in(n, invariant);
        return s_pol ? n * c : n / c;
    };

    cplx eta0 = eta_of(n_ambient);
    cplx etas = eta_of(n_substrate);

    // Characteristic matrix product M = M1 · M2 · ... · Mk
    cplx m00 = one, m01(0.0f, 0.0f), m10(0.0f, 0.0f), m11 = one;
    for (const Layer& L : layers)
    {
        cplx c     = cos_theta_in(L.n, invariant);
        cplx eta   = s_pol ? L.n * c : L.n / c;
        cplx delta = cplx(2.0f * 3.14159265358979323846f * L.thickness_nm
                          / lambda_nm, 0.0f) * L.n * c;

        cplx cd = std::cos(delta);
        cplx sd = std::sin(delta);
        cplx i_sd(-sd.imag(), sd.real()); // i·sin(delta)

        cplx l00 = cd,          l01 = i_sd / eta;
        cplx l10 = i_sd * eta,  l11 = cd;

        cplx t00 = m00 * l00 + m01 * l10;
        cplx t01 = m00 * l01 + m01 * l11;
        cplx t10 = m10 * l00 + m11 * l10;
        cplx t11 = m10 * l01 + m11 * l11;
        m00 = t00; m01 = t01; m10 = t10; m11 = t11;
    }

    cplx B = m00 + m01 * etas;
    cplx C = m10 + m11 * etas;

    return (eta0 * B - C) / (eta0 * B + C);
}

} // namespace detail

// Unpolarized reflectance of ambient → layers → substrate at `aoi_deg`
// (measured in the ambient medium) and `lambda_nm`.  Layers are ordered from
// the ambient side inward.  Result clamped to [0,1].
inline float reflectance(const std::vector<Layer>& layers,
                         std::complex<float> n_ambient,
                         std::complex<float> n_substrate,
                         float lambda_nm, float aoi_deg)
{
    using detail::cplx;

    float theta0    = aoi_deg * (3.14159265358979323846f / 180.0f);
    float invariant = n_ambient.real() * std::sin(theta0);

    cplx rs = detail::stack_amplitude(layers, n_ambient, n_substrate,
                                      lambda_nm, invariant, /*s_pol=*/true);
    cplx rp = detail::stack_amplitude(layers, n_ambient, n_substrate,
                                      lambda_nm, invariant, /*s_pol=*/false);

    float R = 0.5f * (std::norm(rs) + std::norm(rp));
    if (R < 0.0f) R = 0.0f;
    if (R > 1.0f) R = 1.0f;
    return R;
}

} // namespace tmm
