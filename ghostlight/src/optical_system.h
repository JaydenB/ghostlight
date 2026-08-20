// ============================================================================
// optical_system.h — OpticalSystem data structures and JSON lens file parser
//
// Loads `ghostlight-optical` JSON lens files (see lenses/README.md) and
// flattens the element/surface hierarchy into the flat std::vector<Surface>
// sequence the tracer + CUDA kernel consume.
//
// The file format groups surfaces into elements (N surfaces + N-1 materials)
// plus an optional top-level pivots[] rig that translates/rotates groups of
// elements around a pivot point. Pivots are baked into surface decenter/rot/z
// at load time; the runtime never sees them.
//
// Surface is trivially copyable (safe for cudaMemcpy).  It is NOT strictly POD
// because Coating has default member initializers, but all members are plain
// data with trivial copy/move/destroy — cudaMemcpy remains valid.
// Coating pointer fields (table, sa_*) are null for SIMPLE coatings and must
// be patched to device-side copies before GPU upload for table-backed coatings.
//
// Format features:
//   - asphere / cylindrical form parameters
//   - Sellmeier dispersion coefficients for the medium AFTER each surface;
//     n_d / V_d are backfilled for ghost pre-filter and print_summary()
//   - UUID strings, held parallel in OpticalSystem (kept off the POD so
//     cudaMemcpy can copy `surfaces.data()` verbatim).
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

#include <cstdint>
#include <string>
#include <vector>
#include "aperture_profile.h"
#include "fresnel.h"

// Lens-file version shared by the loader and Python writer.
constexpr int LENS_FORMAT_MAJOR = 1;
constexpr int LENS_FORMAT_MINOR = 0;

// Maximum number of asphere polynomial terms (A4, A6, A8, ...) carried per
// surface. Eight terms covers A4 through A18.
constexpr int MAX_ASPHERE_TERMS = 8;

// Surface form discriminator.  Matches the `form.type` field in the lens JSON.
enum SurfaceForm : int
{
    FORM_SPHERE      = 0, // radius (0 = flat / plane)
    FORM_ASPHERE     = 1, // radius, conic_k, asphere_terms[]
    FORM_CYLINDRICAL = 2, // radius, cyl_axis
};

enum CylinderAxis : int
{
    CYL_AXIS_X = 0,
    CYL_AXIS_Y = 1,
};

// Aperture stop shape.  Drives both vignetting (in check_aperture()) and the
// entrance-pupil sampling mask.  Defaults to CIRCLE, so a Surface produced by a
// code path that doesn't set the aperture-shape fields behaves as a plain circle.
enum ApertureShape : int
{
    APERTURE_CIRCLE  = 0, // bounding ellipse only — current circle when aspect = 1
    APERTURE_POLYGON = 1, // regular n-gon inscribed within bounding ellipse
    APERTURE_IMAGE   = 2, // bitmap-driven mask, pixel data on OpticalSystem::aperture_images
};

// Dispersion model for the medium AFTER a surface.  Two models are supported;
// air is special-cased so per-surface storage stays uniform.
enum DispersionModel : int
{
    DISP_AIR       = 0, // n = 1, no dispersion
    DISP_ABBE      = 1, // n_d + V_d via Cauchy approximation
    DISP_SELLMEIER = 2, // n²(λ) = 1 + Σ B[i]λ²/(λ²−C[i]), λ in μm
};

// ---- Coating types (GPU-safe POD, no STL) --------------------------------

// Discriminator for the coating model stored on a surface.
enum class CoatingModel : uint8_t
{
    SIMPLE           = 0, // ar_layers integer
    SPECTRAL         = 1, // wavelength-dependent 1D lookup
    ANGULAR          = 2, // incidence-angle-dependent 1D lookup
    SPECTRAL_ANGULAR = 3, // 2D lookup (wavelength × angle)
    ATTENUATOR_GAUSS = 4, // Gaussian amplitude modifier (position-based)
    ARTIST           = 5, // RGB tint + strength, analytic smooth R(λ)
};

// One entry in a 1-D spectral or angular reflectance table.
struct CoatingTable1D
{
    float key; // lambda_nm (SPECTRAL) or angle_deg (ANGULAR)
    float r;   // reflectance at this key
};

// Per-surface coating descriptor.  GPU-safe: no STL, only plain types and
// raw pointers.  Pointer members (table, sa_*) are patched to device copies
// when surfaces are uploaded to the GPU; they are null for SIMPLE coatings.
struct Coating
{
    CoatingModel model = CoatingModel::SIMPLE;

    // SIMPLE
    int ar_layers = 0;

    // SPECTRAL / ANGULAR — sorted ascending by key
    const CoatingTable1D* table              = nullptr;
    int                   table_count        = 0;
    bool                  out_of_range_discard = false; // false = clamp

    // SPECTRAL_ANGULAR
    const float* sa_wavelengths   = nullptr; // [sa_n_wavelengths], nm
    const float* sa_angles        = nullptr; // [sa_n_angles], deg
    const float* sa_r             = nullptr; // [sa_n_wavelengths * sa_n_angles], row-major
    int          sa_n_wavelengths = 0;
    int          sa_n_angles      = 0;

    // ATTENUATOR_GAUSS
    float gauss_sigma      = 0.0f;
    float gauss_background = 0.0f;
    float gauss_peak       = 0.0f;
    float gauss_decenter_x = 0.0f;
    float gauss_decenter_y = 0.0f;

    // ANGULAR / SPECTRAL_ANGULAR: IOR of the medium the table's angle axis is
    // measured in (1.0 = air, the manufacturer-data convention).  Lookups
    // convert the local incidence angle via the Snell invariant n·sinθ so a
    // glass-side ghost bounce reads the air-equivalent row.
    float angle_ref_ior = 1.0f;

    // ARTIST — RGB tint + strength, evaluated analytically (no tables).
    // White tint → flat R(λ) = tint_strength; a pure hue → smooth bump.
    float tint_r        = 1.0f;
    float tint_g        = 1.0f;
    float tint_b        = 1.0f;
    float tint_strength = 0.04f; // ≈ bare-glass single-surface reflectance
};

// One optical surface in the sequential trace list.
//
// Trivially copyable — safe for cudaMemcpy to the GPU.  Not strictly POD
// (Coating has default member initializers) but copy/move/destroy are all
// trivial.  SIMPLE coatings have null pointer fields; table-backed coatings require
// pointer patching to device-side arena copies before GPU upload.
// Adding fields is safe (sizeof grows; host and device agree because both
// include this header).
struct Surface
{
    // ---- Trace-facing geometry and materials ----
    float radius;        // signed radius of curvature in mm (0 = flat)
    float thickness;     // axial distance to next surface (mm)
    float ior;           // n_d of medium AFTER this surface (1 = air)
    float abbe_v;        // V_d of medium AFTER this surface (0 = air)
    float   semi_aperture; // clear semi-diameter (mm)
    Coating coating;       // coating descriptor (SIMPLE by default)
    bool    is_stop;       // is this the aperture stop?
    // Inactive surfaces are skipped by both the primary trace and ghost
    // enumeration: the ray passes through unbent, the surface contributes
    // no Fresnel weight, and no ghost pair includes it. Geometry and nominal
    // positions are preserved.
    bool    is_active = true;

    float z;             // world axial position of surface vertex (mm), set by load()

    // ---- Aperture shape ----
    // semi_aperture remains the Y-axis half-extent of the bounding ellipse.
    // With aspect ≠ 1 the bound is an ellipse with X half-axis
    // (semi_aperture * aperture_aspect) and Y half-axis semi_aperture.
    int   aperture_shape        = APERTURE_CIRCLE;
    int   aperture_blades       = 0;     // POLYGON: blade count (≥ 3)
    float aperture_rotation_rad = 0.0f;  // POLYGON: rotation in radians (parser converts from degrees)
    float aperture_aspect       = 1.0f;  // X-axis scale; bounding ellipse when ≠ 1
    float aperture_semi_diameter = 0.0f; // IMAGE: world-space radius at the image boundary (mm)

    // ---- Blade shape (POLYGON only) ----
    // Authored controls; all zero reproduces the plain regular n-gon exactly.
    // Notch and notch angle are stored in radians but are NOMINAL — they are
    // applied as a fraction of the blade's own sector, not as literal angles.
    float aperture_curvature       = 0.0f;  // [-1, +1]
    float aperture_twist           = 0.0f;  // [-1, +1]
    float aperture_notch_rad       = 0.0f;  // [-45 deg, +45 deg]
    float aperture_notch_angle_rad = 0.0f;  // [0, 45 deg]

    // Derived from the four controls above plus blades / rotation. Lives on the
    // POD because check_aperture() gets nothing but a `const Surface&` on both
    // CPU and GPU. Rebuild it with refresh_aperture_profile() after any edit —
    // the pybind setters and OpticalSystem::sync_aperture_profiles() do.
    ApertureProfile aperture_profile;

    // Rigid-body transform from the surface's canonical frame to world.
    // decenter_x/y: world lateral offset of the vertex (mm); 0 = on-axis.
    // rot: row-major 3×3 local→world rotation (identity = untransformed).
    // All three are set by load() from the element transform (position x/y,
    // rotation tilt_x/tilt_y/roll, and the pivot's centre-of-rotation offset).
    // A programmatically-built Surface keeps the identity defaults below.
    // NOTE: finalize() does NOT touch these — it only relays `z` from the
    // thickness chain. Editors call finalize() on every spacing edit, so
    // clearing decenter/rot there would silently un-tilt an aligned system.
    float decenter_x = 0.0f;
    float decenter_y = 0.0f;
    float rot[9] = {1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f};

    // ---- Surface form (asphere / cylindrical) ----
    int   form;                              // SurfaceForm
    float conic_k;                           // FORM_ASPHERE: conic constant K
    float asphere_terms[MAX_ASPHERE_TERMS];  // FORM_ASPHERE: A4, A6, A8, ...
    int   n_asphere_terms;                   // populated entries in asphere_terms
    int   cyl_axis;                          // FORM_CYLINDRICAL: CylinderAxis

    // ---- Dispersion for medium AFTER this surface (parsed) ----
    int   disp_model;     // DispersionModel
    float sellmeier_B[3]; // DISP_SELLMEIER
    float sellmeier_C[3]; // DISP_SELLMEIER

    // Wavelength-dependent IOR. Dispatches on disp_model:
    //   DISP_SELLMEIER — direct Sellmeier evaluation (accurate across full UV-IR range)
    //   DISP_ABBE / DISP_AIR — Cauchy approximation via backfilled n_d / V_d
    __host__ __device__ float ior_at(float lambda_nm) const
    {
        if (disp_model == DISP_SELLMEIER)
            return sellmeier_n(sellmeier_B, sellmeier_C, lambda_nm);
        return dispersion_ior(ior, abbe_v, lambda_nm);
    }

    // Rebuild aperture_profile from the authored aperture fields. Cheap (one
    // bake, ~500 evaluations for the area integral) and idempotent.
    void refresh_aperture_profile()
    {
        aperture_profile = (aperture_shape == APERTURE_POLYGON)
            ? make_aperture_profile(aperture_blades, aperture_rotation_rad,
                                    ApertureShapeParams{aperture_curvature,
                                                        aperture_twist,
                                                        aperture_notch_rad,
                                                        aperture_notch_angle_rad})
            : ApertureProfile{};
    }
};

// CPU-side bitmap data backing an APERTURE_IMAGE surface.  One entry per
// surface (held parallel in OpticalSystem::aperture_images); only entries
// whose matching Surface has aperture_shape == APERTURE_IMAGE carry
// meaningful content.  Pixels are row-major [0,1] luminance; an empty
// `pixels` vector means "image not yet loaded" — both CPU and GPU paths pass
// the ray through in that state.
//
// `source_path` is populated by the parser when it sees an image-aperture
// modifier; consuming code (typically a Python helper that uses an image
// library) decodes it and fills width / height / pixels.  Kept off the
// Surface POD so cudaMemcpy(surfaces.data()) stays valid.
struct ApertureImage
{
    int                width  = 0;
    int                height = 0;
    std::vector<float> pixels;        // row-major [0,1] luminance; empty = not loaded
    float              semi_diameter = 0.0f;
    std::string        source_path;   // populated by parser; consumed by image loader
};

// One thin-film layer of a physical coating stack (TMM source data).  Kept as
// the authoring source-of-truth so the writer can re-emit the "layers" JSON;
// the tracer only ever sees the SPECTRAL_ANGULAR table baked from it.
// nk_* are parallel arrays sorted ascending by wavelength (λ in μm, as
// authored in the lens file; converted to nm once, at bake time).
struct CoatingLayerSpec
{
    std::string        material;            // display name, e.g. "MgF2"
    float              thickness_nm = 0.0f;
    std::vector<float> nk_lambda_um;        // sample wavelengths (μm)
    std::vector<float> nk_n;                // refractive index at each sample
    std::vector<float> nk_k;                // extinction coefficient at each sample
};

// CPU-side owning storage for one surface's table-backed coating data.  Held
// parallel to `surfaces` on OpticalSystem (like aperture_images) so the
// Surface POD stays cudaMemcpy-safe; Surface.coating's pointer fields are
// patched into these vectors by OpticalSystem::sync_coating_pointers().
struct CoatingTables
{
    std::vector<CoatingTable1D> table;          // SPECTRAL / ANGULAR
    std::vector<float>          sa_wavelengths; // SPECTRAL_ANGULAR, nm
    std::vector<float>          sa_angles;      // SPECTRAL_ANGULAR, deg
    std::vector<float>          sa_r;           // [n_wavelengths * n_angles], row-major
    std::vector<CoatingLayerSpec> layers;       // TMM stack source; empty = none

    bool has_data() const
    {
        return !table.empty() || !sa_r.empty();
    }
};

// Complete optical system: flat ordered sequence of surfaces + sensor plane,
// produced by flattening the element hierarchy and baking the pivot rig.
struct OpticalSystem
{
    OpticalSystem() = default;
    OpticalSystem(const OpticalSystem& other);
    OpticalSystem& operator=(const OpticalSystem& other);
    OpticalSystem(OpticalSystem&& other) noexcept;
    OpticalSystem& operator=(OpticalSystem&& other) noexcept;

    // ---- Identity / metadata ----
    std::string name;
    // Metadata only — NOT used by the ray tracer directly.
    // Ray tracing is fully determined by surface geometry (radius, thickness, z, ior, ...).
    // Used by lens_calibration.cpp as a fallback when ray tracing fails to determine
    // sensor half-dimensions, and may be read by the Python bindings for display purposes.
    float focal_length = 0;

    // ---- Sequential trace data ----
    // Convention: the sensor lives at world z=0. Surfaces are laid out at
    // z ≤ 0, with light travelling in +z. The last surface's `thickness`
    // is the back focal distance (gap from back vertex to sensor).
    std::vector<Surface> surfaces;

    // Parallel to `surfaces`: surface UUIDs.  Kept off the POD so cudaMemcpy on
    // `surfaces.data()` stays valid.  Entries are empty strings for surfaces
    // that didn't carry an `id` in the source file.
    std::vector<std::string> surface_ids;

    // Parallel to `surfaces`: image-aperture pixel data.  Default-constructed
    // entries (empty `pixels`, source_path "") for surfaces that don't carry
    // an image-aperture modifier.  Sized by load() / finalize() to match
    // surfaces.size().  Held off the Surface POD so cudaMemcpy stays valid.
    std::vector<ApertureImage> aperture_images;

    // Parallel to `surfaces`: owning storage for table-backed coating data
    // (SPECTRAL / ANGULAR / SPECTRAL_ANGULAR tables + TMM layer specs).
    // Surface.coating's pointer fields point INTO these vectors; call
    // sync_coating_pointers() after any mutation of this member.  Sized by
    // load() / finalize() to match surfaces.size().
    std::vector<CoatingTables> coating_tables;

    // Re-patch every Surface.coating pointer field (table, sa_*) to this
    // system's own coating_tables entries (null when the entry is empty or
    // missing).  Must be called after any edit to coating_tables and after
    // structural surface edits; load() and finalize() call it.  Also makes
    // surfaces copied in from another system safe: their foreign pointers
    // are overwritten (table data itself must be moved via coating_tables).
    void sync_coating_pointers();

    // Rebuild every derived aperture_profile after direct edits to authored
    // blade fields. Pybind setters perform this automatically.
    void sync_aperture_profiles()
    {
        for (auto& s : surfaces) s.refresh_aperture_profile();
    }

    // Evaluate the TMM layer stack on surface i (coating_tables[i].layers)
    // and bake it into a SPECTRAL_ANGULAR table (λ 400–700 nm × AOI 0–85°),
    // setting the surface's coating model and angle_ref_ior.  Requires final
    // surface IORs, so callers run it after flattening/material assignment.
    // No-op (returns true) when the layer list is empty.
    bool bake_coating_layers(int i, std::string* err = nullptr);

    // FNV-1a hash over all per-surface coating state: the Coating POD scalar
    // fields plus table contents and layer specs.  Cheap cache key for the
    // Python binding's calibration invalidation.
    uint64_t coating_state_hash() const;
    uint64_t aperture_image_state_hash() const;

    void insert_surface(int index, const Surface& surface,
                        const std::string& surface_id = std::string());
    void erase_surface(int index);

    // Load a `ghostlight-optical` JSON lens file.
    bool load(const char *filename);

    // IOR of the medium BEFORE surface idx (air for the first surface).
    // Walks back across inactive (muted) surfaces — those don't change the
    // medium since the ray passes through unbent — so the medium just to
    // the left of idx is whatever the last *active* surface before idx
    // transitioned to. If no active surface precedes idx, returns air.
    float ior_before(int idx) const
    {
        if (idx > (int)surfaces.size()) idx = (int)surfaces.size();
        for (int k = idx - 1; k >= 0; --k)
            if (surfaces[k].is_active)
                return surfaces[k].ior;
        return 1.0f;
    }

    // Wavelength-dependent version.
    float ior_before(int idx, float lambda_nm) const
    {
        if (idx > (int)surfaces.size()) idx = (int)surfaces.size();
        for (int k = idx - 1; k >= 0; --k)
            if (surfaces[k].is_active)
                return surfaces[k].ior_at(lambda_nm);
        return 1.0f;
    }

    int num_surfaces() const { return (int)surfaces.size(); }

    // Recompute Surface.z so the surface chain ends at z=0 (the sensor).
    // Walks backward from 0 by surface thicknesses, so surfaces[0].z is the
    // most negative. Call after programmatic construction; load() already
    // calls this internally. Returns false if surfaces is empty.
    bool finalize();

    void print_summary() const;

private:
    bool load_in_place(const char *filename);
};
