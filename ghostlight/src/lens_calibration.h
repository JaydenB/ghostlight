// ============================================================================
// lens_calibration.h — Ray-traced covered field of view + first-order optics
// ============================================================================
#pragma once

#include "optical_system.h"

#include <cmath>
#include <vector>

// Results of a lens calibration.
struct LensCalibration {
    // The covered field and the landing that goes with it: the angle at which
    // the surviving area of the renderers' spawn disc falls to 90% of its axial
    // value, and the survivor-mean sensor landing there. This pair is what the
    // source-position map runs on. The 90% point is off the flat top of the
    // throughput curve and remains well conditioned.
    //
    // This is the onset of vignetting, not the image circle; use
    // image_circle_semi_* for illuminated frame coverage.
    float sensor_half_w;       // mm — image half-width at the covered field
    float sensor_half_h;       // mm — image half-height at the covered field
    float max_half_angle_h;    // radians — horizontal covered-field half-angle
    float max_half_angle_v;    // radians — vertical covered-field half-angle

    // Image-circle edge at 5% of axial throughput.
    float image_circle_semi_w = 0.0f;   // mm
    float image_circle_semi_h = 0.0f;   // mm

    // First-order (paraxial) quantities, ray-traced at the d-line. Needed by
    // the diffraction starburst pass to map the aperture Fraunhofer pattern to
    // physical sensor micrometres without an arbitrary scale knob: the pattern
    // pitch on the sensor is lambda * f_number * pupil_fill. Stored per axis so
    // anamorphic (aperture_aspect != 1) systems scale correctly on each axis.
    //
    // focal_length: image height / tan(field angle) at a small field angle
    //   (y' = f * tan(theta) definition; distortion is negligible near axis).
    // entrance_pupil_semi: on-axis marginal-ray half-height at the front-element
    //   plane — the radius of the beam the system actually admits (mm).
    // f_number: focal_length / (2 * entrance_pupil_semi).
    // A value of 0 means the solve failed (degenerate lens) and callers should
    // treat the first-order data as unavailable.
    float focal_length_x = 0.0f, focal_length_y = 0.0f;   // mm
    float entrance_pupil_semi_x = 0.0f, entrance_pupil_semi_y = 0.0f;  // mm
    float f_number_x = 0.0f, f_number_y = 0.0f;

    // Open area of the stop as a fraction of the disk its support radii bound.
    // 1.0 for a circular or image stop; for a bladed one it is the aperture
    // profile's own area fraction (0.827 for a plain hexagon). f-number is a
    // bounding measure and can't see how much of that bound the silhouette
    // actually fills, so the diffraction starburst multiplies its aperture
    // light-collection term by this: a scalloped iris passes less light and
    // genuinely dims. Traced ghost throughput already includes this area.
    float pupil_area_frac = 1.0f;
};

// Build a wavelength list for n_samples uniformly spaced in [lambda_min, lambda_max].
// Cell centers: lambda_i = lambda_min + (lambda_max - lambda_min) * (i + 0.5) / n_samples.
inline std::vector<float> build_spectral_lambdas(int n_samples,
                                                  float lambda_min = 400.0f,
                                                  float lambda_max = 700.0f)
{
    std::vector<float> out(n_samples);
    for (int i = 0; i < n_samples; ++i)
        out[i] = lambda_min + (lambda_max - lambda_min) * (i + 0.5f) / n_samples;
    return out;
}

// Calibrate a lens by ray tracing. Call once per OpticalSystem — fully deterministic.
// d_line_nm: reference wavelength for covered-field calibration (default: Fraunhofer d-line 587.56 nm).
LensCalibration calibrate_lens(const OpticalSystem&         lens,
                                float                     d_line_nm = 587.56f);
