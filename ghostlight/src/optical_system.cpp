// ============================================================================
// optical_system.cpp — `ghostlight-optical` JSON loader + flatten-to-sequential.
//
// Loads the lens schema (optical_system → elements → surfaces + materials,
// plus an optional top-level `pivots[]` rig) and produces a flat
// std::vector<Surface> ready for the sequential tracer / CUDA kernel.
//
// Flatten algorithm:
//   1. Parse glass catalogue.
//   2. Walk optical_system once to resolve every element's transform into a
//      world-frame Transform. Elements with `position.mode = "absolute"` are
//      taken verbatim; `relative_to_preceding` has its z rewritten to
//      prev_resolved_z + z (relative is z-only; x/y stay absolute and the
//      previous element's rotation/decenter are ignored).
//   3. Parse `pivots` (optional). For each pivot, compute the pivot point
//      (centroid of resolved element origins, or manual) and compose the
//      pivot's rotation/translation onto every targeted element's transform.
//      Multiple pivots stack in array order.
//   4. For each element, emit one Surface per geometry entry.  Within-element
//      gaps come from surface.thickness; inter-element gaps are derived from
//      nominal (pre-tilt) z positions.  The medium after surface i is
//      materials[i] for i < N_el-1, air for the last surface of the element.
//   5. Resolve glass names through the catalogue and populate Surface IOR
//      data.  Sellmeier glasses are stored exactly; n_d / V_d are backfilled
//      for ghost pre-filter and print_summary() which read them directly.
//   6. UUIDs land in OpticalSystem::surface_ids (parallel to surfaces).
// ============================================================================

#include "optical_system.h"

#include "nlohmann/json.hpp"
#include "tmm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using nlohmann::json;

// ---------------------------------------------------------------------------
// Glass catalogue — internal representation parsed from `glass_catalogue`.
// ---------------------------------------------------------------------------

namespace
{

struct GlassEntry
{
    // The catalogue's `name` field is authoring provenance only: the tracer
    // keys glasses by their catalogue key, so nothing here needs it.  It is
    // still written and schema-valid; it is simply not parsed.
    int         model;       // DispersionModel
    float       n_d  = 1.0f; // d-line index (DISP_ABBE / backfilled from Sellmeier)
    float       V_d  = 0.0f; // Abbe number
    float       B[3] = {0,0,0}; // Sellmeier B
    float       C[3] = {0,0,0}; // Sellmeier C  (μm²)
};

// Backfill n_d / V_d from Sellmeier coefficients for callers that read them
// directly (ghost pre-filter, print_summary) rather than going through ior_at().
void sellmeier_to_abbe(const float B[3], const float C[3],
                       float &out_n_d, float &out_V_d)
{
    float n_d = sellmeier_n(B, C, 587.56f);   // d-line
    float n_F = sellmeier_n(B, C, 486.13f);   // F-line
    float n_C = sellmeier_n(B, C, 656.27f);   // C-line
    out_n_d = n_d;
    out_V_d = (n_F - n_C > 1e-9f) ? (n_d - 1.0f) / (n_F - n_C) : 0.0f;
}

// ---- Catalogue parser -----------------------------------------------------

bool parse_glass_catalogue(const json &j_cat,
                           std::map<std::string, GlassEntry> &out,
                           const char *path)
{
    if (!j_cat.is_object())
    {
        fprintf(stderr, "ERROR: %s: glass_catalogue must be an object\n", path);
        return false;
    }
    for (auto it = j_cat.begin(); it != j_cat.end(); ++it)
    {
        const std::string key = it.key();
        const json       &je  = it.value();

        GlassEntry g;

        if (!je.contains("dispersion") || !je["dispersion"].is_object())
        {
            fprintf(stderr, "ERROR: %s: glass '%s' missing dispersion block\n",
                    path, key.c_str());
            return false;
        }
        const json &jd    = je["dispersion"];
        const std::string model = jd.value("model", std::string());

        if (model == "sellmeier")
        {
            if (!jd.contains("B") || !jd.contains("C")
                || !jd["B"].is_array() || !jd["C"].is_array()
                || jd["B"].size() != 3 || jd["C"].size() != 3)
            {
                fprintf(stderr, "ERROR: %s: glass '%s' sellmeier needs B[3] and C[3]\n",
                        path, key.c_str());
                return false;
            }
            for (int i = 0; i < 3; ++i)
            {
                g.B[i] = jd["B"][i].get<float>();
                g.C[i] = jd["C"][i].get<float>();
            }
            g.model = DISP_SELLMEIER;
            sellmeier_to_abbe(g.B, g.C, g.n_d, g.V_d);
        }
        else if (model == "abbe")
        {
            if (!jd.contains("nd") || !jd.contains("Vd"))
            {
                fprintf(stderr, "ERROR: %s: glass '%s' abbe needs nd + Vd\n",
                        path, key.c_str());
                return false;
            }
            g.n_d   = jd["nd"].get<float>();
            g.V_d   = jd["Vd"].get<float>();
            g.model = DISP_ABBE;
        }
        else
        {
            fprintf(stderr, "ERROR: %s: glass '%s' unknown dispersion model '%s'\n",
                    path, key.c_str(), model.c_str());
            return false;
        }

        out.emplace(key, std::move(g));
    }
    return true;
}

// ---- Transform helpers ----------------------------------------------------

// Full rigid-body transform: world position of the element origin (mm) and a
// row-major 3×3 local→world rotation matrix.
//
// `piv_corr` carries the element's own centre-of-rotation offset. The element's
// map is
//
//     E(p) = pos + P + rot·(p − P)
//          = (pos + piv_corr) + rot·p,     piv_corr = P − rot·P
//
// so a non-zero pivot is exactly an extra world-space translation on the vertex
// bake. It is held apart from `pos` on purpose: `pos[2]` is read as the
// **nominal axial z** by the relative-position chain, the inter-element air-gap
// patch, and the sensor rebase. Folding a tilt-dependent axial correction into
// those would corrupt the sequential thickness chain — the same reason the gap
// patch already works off nominal rather than tilted z. Only flatten_element
// adds piv_corr, and only to the surface vertex.
//
struct Transform
{
    float pos[3];       // x, y, z (mm)
    float rot[9];       // row-major 3×3
    float piv_corr[3];  // world-space vertex-bake offset from the pivot

    Transform()
    {
        pos[0] = pos[1] = pos[2] = 0.0f;
        rot[0]=1; rot[1]=0; rot[2]=0;
        rot[3]=0; rot[4]=1; rot[5]=0;
        rot[6]=0; rot[7]=0; rot[8]=1;
        piv_corr[0] = piv_corr[1] = piv_corr[2] = 0.0f;
    }
};

// Build a row-major rotation matrix from Euler angles (degrees).
// Convention: R = Ry(tilt_y) * Rx(tilt_x) * Rz(roll)
static void make_rotation(float tilt_x_deg, float tilt_y_deg, float roll_deg,
                           float out[9])
{
    const float PI = 3.14159265358979323846f;
    float tx = tilt_x_deg * (PI / 180.0f);
    float ty = tilt_y_deg * (PI / 180.0f);
    float rz = roll_deg   * (PI / 180.0f);
    float cx = std::cos(tx), sx = std::sin(tx);
    float cy = std::cos(ty), sy = std::sin(ty);
    float cz = std::cos(rz), sz = std::sin(rz);
    // R = Ry * Rx * Rz  (row-major, right-handed)
    out[0] =  cy*cz + sy*sx*sz;   out[1] = -cy*sz + sy*sx*cz;   out[2] =  sy*cx;
    out[3] =  cx*sz;               out[4] =  cx*cz;               out[5] = -sx;
    out[6] = -sy*cz + cy*sx*sz;   out[7] =  sy*sz + cy*sx*cz;   out[8] =  cy*cx;
}

// 3×3 row-major matrix-vector multiply: c = A * b
static void mat3_mul_vec(const float A[9], const float b[3], float c[3])
{
    c[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
    c[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];
    c[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];
}

// 3×3 row-major matrix multiply: C = A * B
static void mat3_mul_mat(const float A[9], const float B[9], float C[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
                s += A[i*3+k] * B[k*3+j];
            C[i*3+j] = s;
        }
}

// Compose two transforms: M = outer ∘ inner
//   M.pos      = outer.pos + outer.rot * inner.pos
//   M.rot      = outer.rot * inner.rot
//   M.piv_corr = outer.piv_corr + outer.rot * inner.piv_corr
// The pivot correction is a world-space vector in the inner frame, so it
// rotates with the outer transform just like a position does.
static Transform compose_transforms(const Transform& outer, const Transform& inner)
{
    Transform M;
    float rp[3];
    mat3_mul_vec(outer.rot, inner.pos, rp);
    M.pos[0] = outer.pos[0] + rp[0];
    M.pos[1] = outer.pos[1] + rp[1];
    M.pos[2] = outer.pos[2] + rp[2];
    mat3_mul_mat(outer.rot, inner.rot, M.rot);
    float rc[3];
    mat3_mul_vec(outer.rot, inner.piv_corr, rc);
    M.piv_corr[0] = outer.piv_corr[0] + rc[0];
    M.piv_corr[1] = outer.piv_corr[1] + rc[1];
    M.piv_corr[2] = outer.piv_corr[2] + rc[2];
    return M;
}

// Parse a transform JSON block into a full rigid-body Transform.
// Reads position.{x,y,z}, rotation.{tilt_x, tilt_y, roll} and pivot.{x,y,z};
// all default to 0.  `position.mode` is ignored here — the caller applies it
// inline after this returns (see the position.mode pass in
// flatten_optical_system).
//
// `pivot` is the centre of rotation in the element's local frame, relative to
// its first surface vertex. An absent or zero pivot rotates about that vertex.
Transform parse_transform(const json &j_xform)
{
    Transform result;
    if (!j_xform.is_object()) return result;

    float tilt_x = 0.0f, tilt_y = 0.0f, roll = 0.0f;
    float pivot[3] = {0.0f, 0.0f, 0.0f};

    if (j_xform.contains("position") && j_xform["position"].is_object())
    {
        const json &p = j_xform["position"];
        if (p.contains("x")) result.pos[0] = p["x"].get<float>();
        if (p.contains("y")) result.pos[1] = p["y"].get<float>();
        if (p.contains("z")) result.pos[2] = p["z"].get<float>();
    }
    if (j_xform.contains("rotation") && j_xform["rotation"].is_object())
    {
        const json &r = j_xform["rotation"];
        if (r.contains("tilt_x")) tilt_x = r["tilt_x"].get<float>();
        if (r.contains("tilt_y")) tilt_y = r["tilt_y"].get<float>();
        if (r.contains("roll"))   roll    = r["roll"].get<float>();
    }
    if (j_xform.contains("pivot") && j_xform["pivot"].is_object())
    {
        const json &v = j_xform["pivot"];
        if (v.contains("x")) pivot[0] = v["x"].get<float>();
        if (v.contains("y")) pivot[1] = v["y"].get<float>();
        if (v.contains("z")) pivot[2] = v["z"].get<float>();
    }
    make_rotation(tilt_x, tilt_y, roll, result.rot);

    // piv_corr = P − rot·P; zero for an identity rotation or zero pivot.
    if (pivot[0] != 0.0f || pivot[1] != 0.0f || pivot[2] != 0.0f)
    {
        float rp[3];
        mat3_mul_vec(result.rot, pivot, rp);
        result.piv_corr[0] = pivot[0] - rp[0];
        result.piv_corr[1] = pivot[1] - rp[1];
        result.piv_corr[2] = pivot[2] - rp[2];
    }
    return result;
}

// ---- Coating extraction ---------------------------------------------------

namespace
{

inline float clamp01(float v)
{
    return (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
}

// Parse a `data: [{<key_field>: k, "r": v}, ...]` array into a sorted
// CoatingTable1D vector.  Returns false (and leaves `out` empty) when no
// valid entries are found.
bool parse_coating_table_1d(const json &m, const char *key_field,
                            std::vector<CoatingTable1D> &out,
                            const char *path)
{
    out.clear();
    if (!m.contains("data") || !m["data"].is_array()) return false;

    for (const auto &e : m["data"])
    {
        if (!e.is_object() || !e.contains(key_field) || !e.contains("r"))
            continue;
        CoatingTable1D row;
        row.key = e[key_field].get<float>();
        row.r   = clamp01(e["r"].get<float>());
        out.push_back(row);
    }
    if (out.empty())
    {
        fprintf(stderr,
                "WARNING: %s: coating modifier has an empty/invalid 'data' "
                "table; treating as uncoated.\n", path);
        return false;
    }
    std::sort(out.begin(), out.end(),
              [](const CoatingTable1D &a, const CoatingTable1D &b)
              { return a.key < b.key; });
    return true;
}

} // namespace

// Walk a surface's `modifiers` array and write a Coating for any coating
// modifier found (last one wins).  Default: SIMPLE, ar_layers=0 (uncoated).
// Returns false — failing the whole load — on a malformed coating; a coating
// that silently degrades to "uncoated" is a physics change hiding in a typo.
//
// Table-backed models fill `out_tables` (the caller pushes it onto
// OpticalSystem::coating_tables parallel to the surface); the Coating's
// pointer fields are left null here and patched by sync_coating_pointers()
// once the flatten completes.  `model: "layers"` is a physical TMM stack: the
// specs are stored on out_tables and baked to a SPECTRAL_ANGULAR table in a
// post-flatten pass (surface IORs are not final yet at this point).
bool extract_coating(const json &j_modifiers, Coating &out,
                     CoatingTables &out_tables, const char *path)
{
    Coating c; // default-constructed: SIMPLE, ar_layers=0, all pointers null
    if (!j_modifiers.is_array()) { out = c; return true; }

    auto reset_tables = [&]() { out_tables = CoatingTables{}; };

    for (const auto &m : j_modifiers)
    {
        if (!m.is_object()) continue;
        if (m.value("type", std::string()) != "coating") continue;

        const std::string model = m.value("model", std::string());

        // Every coating modifier must declare its model.
        if (model.empty())
        {
            fprintf(stderr,
                    "ERROR: %s: coating modifier has no 'model' field.%s\n",
                    path,
                    m.contains("layers")
                        ? "  A physical layer stack must declare"
                          " \"model\": \"layers\"."
                        : "");
            return false;
        }

        if (model == "layers")
        {
            if (!m.contains("layers") || !m["layers"].is_array())
            {
                fprintf(stderr,
                        "ERROR: %s: coating model 'layers' has no 'layers' "
                        "array.\n", path);
                return false;
            }
            // Physical layer stack (TMM).
            std::vector<CoatingLayerSpec> specs;
            bool ok = true;
            for (const auto &jl : m["layers"])
            {
                if (!jl.is_object()) { ok = false; break; }
                CoatingLayerSpec spec;
                spec.material     = jl.value("material", std::string());
                spec.thickness_nm = jl.value("thickness_nm", 0.0f);
                if (jl.contains("nk_table") && jl["nk_table"].is_array())
                {
                    for (const auto &nk : jl["nk_table"])
                    {
                        if (!nk.is_object() || !nk.contains("lambda_um"))
                            continue;
                        spec.nk_lambda_um.push_back(nk["lambda_um"].get<float>());
                        spec.nk_n.push_back(nk.value("n", 1.0f));
                        spec.nk_k.push_back(nk.value("k", 0.0f));
                    }
                }
                if (spec.nk_lambda_um.empty() || spec.thickness_nm <= 0.0f)
                {
                    ok = false;
                    break;
                }
                specs.push_back(std::move(spec));
            }
            if (!ok || specs.empty())
            {
                fprintf(stderr,
                        "ERROR: %s: coating 'layers' stack is empty or has a "
                        "layer with no nk_table / non-positive thickness.\n",
                        path);
                return false;
            }
            c = Coating{};
            reset_tables();
            out_tables.layers = std::move(specs);
            // Model + baked table are produced by bake_coating_layers() in
            // the post-flatten pass; until then the surface reads as SIMPLE
            // uncoated (bare Fresnel) which is the safe fallback.
            continue;
        }

        if (model == "simple")
        {
            c = Coating{};
            reset_tables();
            c.ar_layers = m.value("ar_layers", 0);
        }
        else if (model == "artist")
        {
            c = Coating{};
            reset_tables();
            c.model = CoatingModel::ARTIST;
            if (m.contains("tint") && m["tint"].is_array()
                && m["tint"].size() == 3)
            {
                c.tint_r = clamp01(m["tint"][0].get<float>());
                c.tint_g = clamp01(m["tint"][1].get<float>());
                c.tint_b = clamp01(m["tint"][2].get<float>());
            }
            c.tint_strength = clamp01(m.value("strength", 0.04f));
        }
        else if (model == "spectral" || model == "angular")
        {
            std::vector<CoatingTable1D> table;
            const char *key = (model == "spectral") ? "lambda_nm" : "angle_deg";
            if (!parse_coating_table_1d(m, key, table, path))
                continue;
            c = Coating{};
            reset_tables();
            c.model = (model == "spectral") ? CoatingModel::SPECTRAL
                                            : CoatingModel::ANGULAR;
            c.out_of_range_discard =
                (m.value("out_of_range", std::string("clamp")) == "discard");
            if (model == "angular")
                c.angle_ref_ior = m.value("angle_ref_ior", 1.0f);
            out_tables.table = std::move(table);
        }
        else if (model == "spectral_angular")
        {
            std::vector<float> wl, ang, r;
            if (m.contains("wavelengths_nm") && m["wavelengths_nm"].is_array())
                for (const auto &v : m["wavelengths_nm"])
                    wl.push_back(v.get<float>());
            if (m.contains("angles_deg") && m["angles_deg"].is_array())
                for (const auto &v : m["angles_deg"])
                    ang.push_back(v.get<float>());
            if (m.contains("r") && m["r"].is_array())
                for (const auto &row : m["r"])
                    if (row.is_array())
                        for (const auto &v : row)
                            r.push_back(clamp01(v.get<float>()));

            const bool sorted_wl  = std::is_sorted(wl.begin(), wl.end());
            const bool sorted_ang = std::is_sorted(ang.begin(), ang.end());
            if (wl.empty() || ang.empty()
                || r.size() != wl.size() * ang.size()
                || !sorted_wl || !sorted_ang)
            {
                fprintf(stderr,
                        "WARNING: %s: spectral_angular coating has "
                        "inconsistent or unsorted wavelengths/angles/r arrays; "
                        "treating as uncoated.\n", path);
                continue;
            }
            c = Coating{};
            reset_tables();
            c.model = CoatingModel::SPECTRAL_ANGULAR;
            c.out_of_range_discard =
                (m.value("out_of_range", std::string("clamp")) == "discard");
            c.angle_ref_ior = m.value("angle_ref_ior", 1.0f);
            out_tables.sa_wavelengths = std::move(wl);
            out_tables.sa_angles      = std::move(ang);
            out_tables.sa_r           = std::move(r);
        }
        else if (model == "attenuator_gaussian")
        {
            c = Coating{};
            reset_tables();
            c.model            = CoatingModel::ATTENUATOR_GAUSS;
            c.gauss_sigma      = m.value("sigma", 0.0f);
            c.gauss_background = m.value("attenuation_background", 0.0f);
            c.gauss_peak       = m.value("attenuation_peak", 0.0f);
            c.gauss_decenter_x = m.value("decenter_x", 0.0f);
            c.gauss_decenter_y = m.value("decenter_y", 0.0f);
        }
        else
        {
            fprintf(stderr,
                    "ERROR: %s: coating model '%s' is not recognised.\n",
                    path, model.c_str());
            return false;
        }
    }
    out = c;
    return true;
}

// ---- Aperture modifier extraction ----------------------------------------

// Reset a surface to a plain circular aperture with the given aspect.  Every
// non-polygon outcome (explicit "circular", and each malformed-modifier
// fallback) funnels through here so a shape switch can't leave polygon-only
// state — blade count, rotation, blade shape — behind on the surface.
static void set_circular_aperture(Surface &s, float aspect)
{
    s.aperture_shape              = APERTURE_CIRCLE;
    s.aperture_blades             = 0;
    s.aperture_rotation_rad       = 0.0f;
    s.aperture_aspect             = aspect;
    s.aperture_curvature          = 0.0f;
    s.aperture_twist              = 0.0f;
    s.aperture_notch_rad          = 0.0f;
    s.aperture_notch_angle_rad    = 0.0f;
}

// Read one optional numeric blade-shape control, clamping to its authored range
// with a warning.  Absent means 0 (the plain-polygon default), never "leave what
// was there" — a modifier fully describes the aperture it declares.
static float read_blade_control(const json &m, const char *key,
                                float lo, float hi,
                                const char *path, const char *unit)
{
    if (!m.contains(key)) return 0.0f;
    const float v = m[key].get<float>();
    if (v < lo || v > hi)
    {
        const float c = v < lo ? lo : hi;
        fprintf(stderr,
                "WARNING: %s: aperture %s must be within [%g, %g]%s; got %g — "
                "clamping to %g.\n",
                path, key, lo, hi, unit, v, c);
        return c;
    }
    return v;
}

// Walk a surface's `modifiers` array and apply any aperture modifier(s) to s
// (last one wins; matches extract_coating()'s convention). With no aperture
// modifier, the Surface defaults to a circle with aspect 1.0.
//
// `out_image`: when a "shape": "image" modifier successfully parses, its
// source_path and semi_diameter are copied here.  The caller is responsible
// for placing the result at the matching index in OpticalSystem::aperture_images.
// Pixel data is loaded by a separate helper — the parser never touches pixels.
void apply_aperture_modifiers(const json &j_modifiers, Surface &s,
                              ApertureImage &out_image,
                              const char *path)
{
    if (!j_modifiers.is_array()) return;

    for (const auto &m : j_modifiers)
    {
        if (!m.is_object()) continue;
        if (m.value("type", std::string()) != "aperture") continue;

        const std::string shape = m.value("shape", std::string());

        // aperture_aspect: optional on every shape.  ≤ 0 is invalid → warn and
        // ignore (leave whatever value was previously set — default 1.0).
        float aspect = 1.0f;
        bool  aspect_present = m.contains("aperture_aspect");
        if (aspect_present)
        {
            aspect = m["aperture_aspect"].get<float>();
            if (!(aspect > 0.0f))
            {
                fprintf(stderr,
                        "WARNING: %s: aperture_aspect must be > 0; got %g — "
                        "clamping to 1.0.\n",
                        path, aspect);
                aspect = 1.0f;
            }
        }

        if (shape == "circular")
        {
            set_circular_aperture(s, aspect);
        }
        else if (shape == "polygon")
        {
            int blades = m.value("blades", 0);
            if (blades < 3)
            {
                fprintf(stderr,
                        "WARNING: %s: polygon aperture needs blades >= 3; "
                        "got %d — falling back to circular.\n",
                        path, blades);
                set_circular_aperture(s, aspect);
                continue;
            }
            const float PI = 3.14159265358979323846f;
            const float DEG = PI / 180.0f;
            float rot_deg = m.value("rotation_deg", 0.0f);
            set_circular_aperture(s, aspect);          // clear, then declare
            s.aperture_shape        = APERTURE_POLYGON;
            s.aperture_blades       = blades;
            s.aperture_rotation_rad = rot_deg * DEG;
            // Blade controls default to a regular polygon. Notch angles are
            // normalized to the blade sector by make_aperture_profile().
            s.aperture_curvature =
                read_blade_control(m, "curvature", -1.0f, 1.0f, path, "");
            s.aperture_twist =
                read_blade_control(m, "twist", -1.0f, 1.0f, path, "");
            s.aperture_notch_rad =
                read_blade_control(m, "notch_deg", -45.0f, 45.0f, path, " degrees") * DEG;
            s.aperture_notch_angle_rad =
                read_blade_control(m, "notch_angle_deg", 0.0f, 45.0f, path, " degrees") * DEG;
        }
        else if (shape == "image")
        {
            // Pixel data is loaded by a caller-supplied helper (see
            // OpticalSystem::load_aperture_images / the Python wrapper) — the
            // parser only records the path and bounding semi-diameter so the
            // file can be decoded later without dragging an image library
            // into the C++ loader.
            if (!m.contains("image_path"))
            {
                fprintf(stderr,
                        "WARNING: %s: image aperture missing 'image_path'; "
                        "falling back to circular.\n",
                        path);
                set_circular_aperture(s, aspect);
                continue;
            }
            if (!m.contains("semi_diameter"))
            {
                fprintf(stderr,
                        "WARNING: %s: image aperture missing 'semi_diameter'; "
                        "falling back to circular.\n",
                        path);
                set_circular_aperture(s, aspect);
                continue;
            }
            float semi_diameter = m["semi_diameter"].get<float>();
            if (!(semi_diameter > 0.0f))
            {
                fprintf(stderr,
                        "WARNING: %s: image aperture semi_diameter must be > 0; "
                        "falling back to circular.\n",
                        path);
                set_circular_aperture(s, aspect);
                continue;
            }
            set_circular_aperture(s, aspect);          // clear, then declare
            s.aperture_shape         = APERTURE_IMAGE;
            s.aperture_semi_diameter = semi_diameter;
            out_image.source_path    = m["image_path"].get<std::string>();
            out_image.semi_diameter  = semi_diameter;
        }
        else
        {
            fprintf(stderr,
                    "WARNING: %s: unknown aperture shape '%s'; "
                    "falling back to circular.\n",
                    path, shape.c_str());
            set_circular_aperture(s, aspect);
        }
    }
}

// ---- Surface form ---------------------------------------------------------

bool parse_form(const json &j_form, Surface &s, const char *path)
{
    if (!j_form.is_object())
    {
        fprintf(stderr, "ERROR: %s: surface 'form' must be an object\n", path);
        return false;
    }

    const std::string type = j_form.value("type", std::string());

    if (type == "sphere")
    {
        s.form   = FORM_SPHERE;
        s.radius = j_form.value("radius", 0.0f);
        return true;
    }
    if (type == "asphere")
    {
        s.form    = FORM_ASPHERE;
        s.radius  = j_form.value("radius", 0.0f);
        s.conic_k = j_form.value("conic_constant", 0.0f);
        if (j_form.contains("terms") && j_form["terms"].is_array())
        {
            const json &terms = j_form["terms"];
            int n = (int)terms.size();
            if (n > MAX_ASPHERE_TERMS)
            {
                fprintf(stderr,
                        "WARNING: %s: asphere has %d terms; truncating to %d.\n",
                        path, n, MAX_ASPHERE_TERMS);
                n = MAX_ASPHERE_TERMS;
            }
            for (int i = 0; i < n; ++i)
                s.asphere_terms[i] = terms[i].get<float>();
            s.n_asphere_terms = n;
        }
        return true;
    }
    if (type == "cylindrical")
    {
        s.form   = FORM_CYLINDRICAL;
        s.radius = j_form.value("radius", 0.0f);
        const std::string ax = j_form.value("axis", std::string("x"));
        s.cyl_axis = (ax == "y" || ax == "Y") ? CYL_AXIS_Y : CYL_AXIS_X;
        return true;
    }

    fprintf(stderr, "ERROR: %s: unknown surface form '%s'\n", path, type.c_str());
    return false;
}

// ---- Material -> Surface IOR fields --------------------------------------

void apply_air(Surface &s)
{
    s.ior            = 1.0f;
    s.abbe_v         = 0.0f;
    s.disp_model     = DISP_AIR;
    s.sellmeier_B[0] = s.sellmeier_B[1] = s.sellmeier_B[2] = 0.0f;
    s.sellmeier_C[0] = s.sellmeier_C[1] = s.sellmeier_C[2] = 0.0f;
}

bool apply_glass(const std::string &glass_name,
                 const std::map<std::string, GlassEntry> &catalogue,
                 Surface &s, const char *path)
{
    if (glass_name.empty() || glass_name == "air" || glass_name == "AIR")
    {
        apply_air(s);
        return true;
    }
    auto it = catalogue.find(glass_name);
    if (it == catalogue.end())
    {
        fprintf(stderr,
                "ERROR: %s: glass '%s' not present in glass_catalogue.\n",
                path, glass_name.c_str());
        return false;
    }
    const GlassEntry &g = it->second;
    s.ior        = g.n_d;
    s.abbe_v     = g.V_d;
    s.disp_model = g.model;
    for (int i = 0; i < 3; ++i)
    {
        s.sellmeier_B[i] = g.B[i];
        s.sellmeier_C[i] = g.C[i];
    }
    return true;
}

// ---- Element flatten ------------------------------------------------------

// Per-element bookkeeping captured during the walk so we can patch inter-
// element air gaps in a second pass using nominal (pre-tilt) axial z values.
struct ElementEmit
{
    float z_nominal;           // nominal axial z of element origin (transform.pos.z only)
    int   first_surf;          // index into OpticalSystem.surfaces
    int   n_surf;              // count of surfaces this element emitted
    float last_surf_z_nominal; // nominal z of the element's final surface vertex
};

// Flatten one element into out.surfaces.  M is the world transform for the
// element (already includes any pivot composition); nominal_z is the
// element origin's nominal axial position (z of M.pos), used for air-gap
// bookkeeping.  out_last_z_nominal receives the nominal z of the last
// surface vertex.
bool flatten_element(const json &j_el,
                     const Transform &M,
                     float nominal_z,
                     const std::map<std::string, GlassEntry> &catalogue,
                     OpticalSystem &out,
                     float &out_last_z_nominal,
                     const char *path)
{
    if (!j_el.contains("surfaces") || !j_el["surfaces"].is_array())
    {
        fprintf(stderr, "ERROR: %s: element missing 'surfaces' array\n", path);
        return false;
    }
    if (!j_el.contains("materials") || !j_el["materials"].is_array())
    {
        fprintf(stderr, "ERROR: %s: element missing 'materials' array\n", path);
        return false;
    }
    const json &j_surfs = j_el["surfaces"];
    const json &j_mats  = j_el["materials"];
    const int   n_surf  = (int)j_surfs.size();
    const int   n_mat   = (int)j_mats.size();
    if (n_surf != n_mat + 1)
    {
        fprintf(stderr,
                "ERROR: %s: element '%s' has %d surfaces and %d materials "
                "(surfaces must equal materials + 1).\n",
                path, j_el.value("name", std::string()).c_str(),
                n_surf, n_mat);
        return false;
    }
    if (n_surf < 1)
    {
        fprintf(stderr, "ERROR: %s: element has no surfaces\n", path);
        return false;
    }

    // cum_z_local: local z offset from this element's origin (starts at 0),
    // used for computing world vertex positions via the element transform M.
    // cum_z_nominal: global z accumulated from nominal_z, used only for
    // out_last_z_nominal so the caller can patch inter-element air gaps.
    float cum_z_local   = 0.0f;
    float cum_z_nominal = nominal_z;
    float last_nominal  = nominal_z;

    for (int i = 0; i < n_surf; ++i)
    {
        const json &js = j_surfs[i];

        Surface s{};
        s.is_stop         = js.value("is_stop", false);
        // Missing `is_active` values default to active.
        s.is_active       = js.value("is_active", true);
        s.semi_aperture   = js.value("semi_aperture", 0.0f);
        CoatingTables coating_record;
        if (!extract_coating(js.value("modifiers", json::array()),
                             s.coating, coating_record, path))
            return false;
        s.n_asphere_terms = 0;
        s.cyl_axis        = CYL_AXIS_X;
        s.conic_k         = 0.0f;
        for (int k = 0; k < MAX_ASPHERE_TERMS; ++k) s.asphere_terms[k] = 0.0f;

        if (!parse_form(js.value("form", json::object()), s, path))
            return false;

        // Aperture shape modifier (defaults already set on Surface{}: CIRCLE,
        // aspect 1.0).  An image-aperture modifier returns the source path +
        // bounding semi_diameter on `image_record`; that record is then
        // appended to OpticalSystem::aperture_images parallel to the surface.
        ApertureImage image_record;
        apply_aperture_modifiers(js.value("modifiers", json::array()), s,
                                 image_record, path);

        // Within-element thickness: gap to next surface inside this element
        // (or, for the last surface, the gap to the next element / sensor).
        // For non-final elements, the last surface's value is overwritten by
        // the air-gap patching pass in flatten_optical_system. For the final
        // element, the loaded value is the back focal distance (gap to sensor
        // at z=0).
        s.thickness = js.value("thickness", 0.0f);

        // Map the local position (0, 0, cum_z_local) through M to get the world
        // vertex: V = M.pos + M.piv_corr + M.rot * (0, 0, cum_z_local).
        // M.rot * (0,0,z) = (rot[2]*z, rot[5]*z, rot[8]*z)  — third column * z.
        // M.piv_corr is zero unless the element declares a centre of rotation;
        // it is deliberately absent from the nominal-z bookkeeping below.
        s.decenter_x = M.pos[0] + M.piv_corr[0] + M.rot[2] * cum_z_local;
        s.decenter_y = M.pos[1] + M.piv_corr[1] + M.rot[5] * cum_z_local;
        s.z          = M.pos[2] + M.piv_corr[2] + M.rot[8] * cum_z_local;
        for (int k = 0; k < 9; ++k) s.rot[k] = M.rot[k];

        // Medium AFTER this surface:
        //   i  < n_surf - 1 → materials[i]
        //   i == n_surf - 1 → air (between elements)
        if (i < n_surf - 1)
        {
            const std::string glass = j_mats[i].value("glass", std::string());
            if (!apply_glass(glass, catalogue, s, path))
                return false;
        }
        else
        {
            apply_air(s);
        }

        if (s.semi_aperture <= 0.0f)
        {
            fprintf(stderr,
                    "WARNING: %s: element '%s' surface %d has semi_aperture <= 0; "
                    "the tracer will reject all rays.\n",
                    path, j_el.value("name", std::string()).c_str(), i);
        }

        out.surfaces.push_back(s);
        // UUIDs must be non-empty because Element::resolve_surfaces() maps them
        // to unique surface indices.
        std::string sid = js.value("id", std::string());
        if (sid.empty())
            sid = "auto-surface-" + std::to_string(out.surfaces.size() - 1);
        out.surface_ids.push_back(std::move(sid));
        out.aperture_images.push_back(std::move(image_record));
        out.coating_tables.push_back(std::move(coating_record));

        last_nominal  = cum_z_nominal;
        cum_z_local   += s.thickness;
        cum_z_nominal += s.thickness;
    }

    out_last_z_nominal = last_nominal;
    return true;
}

// ---- Pivot rig ------------------------------------------------------------

struct PivotPointMode { enum : int { CENTROID = 0, MANUAL = 1 }; };

struct Pivot
{
    std::string              id;
    std::string              name;
    std::vector<std::string> element_ids;
    int                      point_mode = PivotPointMode::CENTROID;
    float                    pivot_xyz[3] = {0,0,0}; // manual mode override
    float                    off_pos[3]   = {0,0,0};
    float                    off_rot[3]   = {0,0,0}; // tilt_x, tilt_y, roll (deg)
};

bool parse_pivots(const json &j_pivots, std::vector<Pivot> &out, const char *path)
{
    if (!j_pivots.is_array())
    {
        fprintf(stderr, "ERROR: %s: pivots must be an array\n", path);
        return false;
    }
    for (const auto &jp : j_pivots)
    {
        if (!jp.is_object())
        {
            fprintf(stderr, "ERROR: %s: pivot entry must be an object\n", path);
            return false;
        }
        Pivot p;
        p.id   = jp.value("id",   std::string());
        p.name = jp.value("name", std::string());

        if (!jp.contains("elements") || !jp["elements"].is_array())
        {
            fprintf(stderr,
                    "ERROR: %s: pivot '%s' missing 'elements' array\n",
                    path, p.name.c_str());
            return false;
        }
        for (const auto &eid : jp["elements"])
            p.element_ids.push_back(eid.get<std::string>());

        if (jp.contains("pivot_point") && jp["pivot_point"].is_object())
        {
            const json &pp = jp["pivot_point"];
            const std::string mode = pp.value("mode", std::string("centroid"));
            p.point_mode = (mode == "manual") ? PivotPointMode::MANUAL
                                              : PivotPointMode::CENTROID;
            p.pivot_xyz[0] = pp.value("x", 0.0f);
            p.pivot_xyz[1] = pp.value("y", 0.0f);
            p.pivot_xyz[2] = pp.value("z", 0.0f);
        }

        if (jp.contains("offset") && jp["offset"].is_object())
        {
            const json &off = jp["offset"];
            if (off.contains("position") && off["position"].is_object())
            {
                const json &op = off["position"];
                p.off_pos[0] = op.value("x", 0.0f);
                p.off_pos[1] = op.value("y", 0.0f);
                p.off_pos[2] = op.value("z", 0.0f);
            }
            if (off.contains("rotation") && off["rotation"].is_object())
            {
                const json &orot = off["rotation"];
                p.off_rot[0] = orot.value("tilt_x", 0.0f);
                p.off_rot[1] = orot.value("tilt_y", 0.0f);
                p.off_rot[2] = orot.value("roll",   0.0f);
            }
        }
        // `exposed` is metadata for editors only; we don't load it here.

        out.push_back(std::move(p));
    }
    return true;
}

// Apply one pivot to one element transform:
//   P_e' = R_off · (P_e − P_p) + P_p + Δp
//   R_e' = R_off · R_e
static void apply_pivot_to_transform(const Pivot &p,
                                     const float pivot_point[3],
                                     Transform &t)
{
    float R_off[9];
    make_rotation(p.off_rot[0], p.off_rot[1], p.off_rot[2], R_off);

    float rel[3] = { t.pos[0] - pivot_point[0],
                     t.pos[1] - pivot_point[1],
                     t.pos[2] - pivot_point[2] };
    float rotated[3];
    mat3_mul_vec(R_off, rel, rotated);
    t.pos[0] = rotated[0] + pivot_point[0] + p.off_pos[0];
    t.pos[1] = rotated[1] + pivot_point[1] + p.off_pos[1];
    t.pos[2] = rotated[2] + pivot_point[2] + p.off_pos[2];

    float R_new[9];
    mat3_mul_mat(R_off, t.rot, R_new);
    for (int i = 0; i < 9; ++i) t.rot[i] = R_new[i];

    // The element's own pivot correction is a world-space vector in the
    // element's pre-rig frame, so the rig's rotation carries it along. Skipping
    // this would leave a pivoted element's vertices behind when a rig tilts it.
    float corr[3];
    mat3_mul_vec(R_off, t.piv_corr, corr);
    t.piv_corr[0] = corr[0];
    t.piv_corr[1] = corr[1];
    t.piv_corr[2] = corr[2];
}

// ---- Optical system flatten ----------------------------------------------

bool flatten_optical_system(const json       &j_root,
                            const std::map<std::string, GlassEntry> &catalogue,
                            OpticalSystem    &out,
                            const char       *path)
{
    if (!j_root.contains("optical_system") || !j_root["optical_system"].is_array())
    {
        fprintf(stderr, "ERROR: %s: optical_system must be an array\n", path);
        return false;
    }
    const json &j_sys = j_root["optical_system"];

    // ---- Pass 1: build resolved transforms per element -------------------
    // Each element gets a baseline Transform from its `transform` block, with
    // position.mode == "relative_to_preceding" rewriting z to a delta from
    // the previously-resolved element's resolved-absolute z. We also stash
    // each element's id and the json node for the second pass.
    struct ElementRecord
    {
        std::string id;
        const json *node;
        Transform   xform;        // mutated by pivots
        float       pre_pivot_z;  // resolved-absolute z BEFORE any pivot composition
    };
    std::vector<ElementRecord> records;
    records.reserve(j_sys.size());

    float prev_resolved_z = 0.0f;
    bool  have_prev       = false;

    for (const auto &item : j_sys)
    {
        const std::string type = item.value("type", std::string());
        if (type != "element")
        {
            fprintf(stderr,
                    "ERROR: %s: optical_system item has unknown type '%s' "
                    "(expected 'element').\n",
                    path, type.c_str());
            return false;
        }

        Transform t = parse_transform(item.value("transform", json::object()));

        // Apply position.mode if present.
        const json &j_xform = item.value("transform", json::object());
        std::string pos_mode = "absolute";
        if (j_xform.is_object() && j_xform.contains("position")
            && j_xform["position"].is_object())
        {
            pos_mode = j_xform["position"].value("mode", std::string("absolute"));
        }
        if (pos_mode == "relative_to_preceding")
        {
            if (!have_prev)
            {
                fprintf(stderr,
                        "WARNING: %s: first element '%s' has "
                        "position.mode='relative_to_preceding'; "
                        "treating as absolute.\n",
                        path, item.value("name", std::string()).c_str());
            }
            else
            {
                t.pos[2] += prev_resolved_z;
            }
        }
        else if (pos_mode != "absolute")
        {
            fprintf(stderr,
                    "WARNING: %s: element '%s' has unknown "
                    "position.mode='%s'; treating as absolute.\n",
                    path, item.value("name", std::string()).c_str(),
                    pos_mode.c_str());
        }

        ElementRecord rec;
        rec.id          = item.value("id", std::string());
        rec.node        = &item;
        rec.xform       = t;
        rec.pre_pivot_z = t.pos[2];
        records.push_back(rec);

        prev_resolved_z = t.pos[2];
        have_prev = true;
    }

    // ---- Pass 2: apply pivots --------------------------------------------
    // Pivots act on resolved-absolute element transforms. Multiple pivots on
    // the same element stack in array order.
    if (j_root.contains("pivots"))
    {
        std::vector<Pivot> pivots;
        if (!parse_pivots(j_root["pivots"], pivots, path))
            return false;

        // Build id -> record index lookup once.
        std::map<std::string, int> id_to_idx;
        for (int i = 0; i < (int)records.size(); ++i)
        {
            if (!records[i].id.empty())
                id_to_idx.emplace(records[i].id, i);
        }

        for (const Pivot &p : pivots)
        {
            // Validate element refs and gather target indices.
            std::vector<int> targets;
            targets.reserve(p.element_ids.size());
            for (const std::string &eid : p.element_ids)
            {
                auto it = id_to_idx.find(eid);
                if (it == id_to_idx.end())
                {
                    fprintf(stderr,
                            "ERROR: %s: pivot '%s' references unknown "
                            "element id '%s'.\n",
                            path, p.name.c_str(), eid.c_str());
                    return false;
                }
                targets.push_back(it->second);
            }
            if (targets.empty()) continue;

            // Compute the pivot point.
            float pivot_point[3] = {0,0,0};
            if (p.point_mode == PivotPointMode::MANUAL)
            {
                pivot_point[0] = p.pivot_xyz[0];
                pivot_point[1] = p.pivot_xyz[1];
                pivot_point[2] = p.pivot_xyz[2];
            }
            else
            {
                // Centroid: mean of the (current) resolved element origins.
                // We compute on the current pivot-stack state so successive
                // pivots see the accumulated motion of preceding pivots.
                double sx = 0, sy = 0, sz = 0;
                for (int idx : targets)
                {
                    sx += records[idx].xform.pos[0];
                    sy += records[idx].xform.pos[1];
                    sz += records[idx].xform.pos[2];
                }
                const double n = (double)targets.size();
                pivot_point[0] = (float)(sx / n);
                pivot_point[1] = (float)(sy / n);
                pivot_point[2] = (float)(sz / n);
            }

            // Apply offset to each targeted element.
            for (int idx : targets)
                apply_pivot_to_transform(p, pivot_point, records[idx].xform);
        }
    }

    // ---- Pass 3: emit surfaces -------------------------------------------
    std::vector<ElementEmit> emits;
    emits.reserve(records.size());

    for (const ElementRecord &rec : records)
    {
        const float nominal_z = rec.xform.pos[2];

        ElementEmit e;
        e.z_nominal  = nominal_z;
        e.first_surf = (int)out.surfaces.size();
        if (!flatten_element(*rec.node, rec.xform, nominal_z, catalogue, out,
                             e.last_surf_z_nominal, path))
            return false;
        e.n_surf = (int)out.surfaces.size() - e.first_surf;
        emits.push_back(e);
    }

    // Patch inter-element air gaps using nominal axial z so tilt-induced
    // axial shifts don't corrupt the sequential thickness chain.
    for (size_t k = 0; k + 1 < emits.size(); ++k)
    {
        const int   last_idx = emits[k].first_surf + emits[k].n_surf - 1;
        const float next_z   = emits[k + 1].z_nominal;
        out.surfaces[last_idx].thickness = next_z - emits[k].last_surf_z_nominal;
    }

    // Rebase to sensor=0 convention: shift every surface so the chain ends at
    // z=0 under the **pre-pivot** authored layout. Using pre-pivot z makes
    // pivots additive on top of a stable sensor anchor — i.e. when a pivot
    // shifts the rear element by +Δz, the rear element visibly moves +Δz
    // relative to the sensor instead of getting cancelled by a re-anchor pass.
    if (!emits.empty() && !records.empty())
    {
        const json &j_last_surfs = (*records.back().node)["surfaces"];
        float internal_total = 0.0f;
        for (const auto &js : j_last_surfs)
            internal_total += js.value("thickness", 0.0f);
        const float total = records.back().pre_pivot_z + internal_total;
        for (auto &s : out.surfaces)
            s.z -= total;
    }

    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

OpticalSystem::OpticalSystem(const OpticalSystem& other)
    : name(other.name), focal_length(other.focal_length),
      surfaces(other.surfaces), surface_ids(other.surface_ids),
      aperture_images(other.aperture_images), coating_tables(other.coating_tables)
{
    sync_coating_pointers();
}

OpticalSystem& OpticalSystem::operator=(const OpticalSystem& other)
{
    if (this == &other) return *this;
    name = other.name;
    focal_length = other.focal_length;
    surfaces = other.surfaces;
    surface_ids = other.surface_ids;
    aperture_images = other.aperture_images;
    coating_tables = other.coating_tables;
    sync_coating_pointers();
    return *this;
}

OpticalSystem::OpticalSystem(OpticalSystem&& other) noexcept
    : name(std::move(other.name)), focal_length(other.focal_length),
      surfaces(std::move(other.surfaces)), surface_ids(std::move(other.surface_ids)),
      aperture_images(std::move(other.aperture_images)),
      coating_tables(std::move(other.coating_tables))
{
    sync_coating_pointers();
}

OpticalSystem& OpticalSystem::operator=(OpticalSystem&& other) noexcept
{
    if (this == &other) return *this;
    name = std::move(other.name);
    focal_length = other.focal_length;
    surfaces = std::move(other.surfaces);
    surface_ids = std::move(other.surface_ids);
    aperture_images = std::move(other.aperture_images);
    coating_tables = std::move(other.coating_tables);
    sync_coating_pointers();
    return *this;
}

bool OpticalSystem::load(const char *filename)
{
    OpticalSystem parsed;
    if (!parsed.load_in_place(filename))
        return false;
    *this = std::move(parsed);
    return true;
}

bool OpticalSystem::load_in_place(const char *filename)
{
    surfaces.clear();
    surface_ids.clear();
    aperture_images.clear();
    coating_tables.clear();
    name.clear();
    focal_length = 0.0f;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        fprintf(stderr, "ERROR: cannot open lens file: %s\n", filename);
        return false;
    }

    json j;
    try
    {
        file >> j;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR: %s: invalid JSON: %s\n", filename, e.what());
        return false;
    }

    // ---- Format / version check ----
    const std::string fmt = j.value("format", std::string());
    if (fmt != "ghostlight-optical")
    {
        fprintf(stderr,
                "ERROR: %s: unexpected format '%s' (expected 'ghostlight-optical')\n",
                filename, fmt.c_str());
        return false;
    }
    // Loading requires the current major version; migration is explicit.
    int major = 0, minor = 0;
    if (j.contains("version") && j["version"].is_object())
    {
        major = j["version"].value("major", 0);
        minor = j["version"].value("minor", 0);
    }
    if (major != LENS_FORMAT_MAJOR)
    {
        fprintf(stderr,
                "ERROR: %s: unsupported lens format major version %d "
                "(this build reads version %d only).\n",
                filename, major, LENS_FORMAT_MAJOR);
        if (major == 2)
            fprintf(stderr,
                    "       This file predates the format consolidation. Run:\n"
                    "         python lenses/migrate_lens_files.py %s\n",
                    filename);
        return false;
    }
    if (minor > LENS_FORMAT_MINOR)
    {
        fprintf(stderr,
                "WARNING: %s: lens format minor version %d is newer than this "
                "build understands (%d); unknown fields will be ignored.\n",
                filename, minor, LENS_FORMAT_MINOR);
    }

    // ---- Metadata ----
    if (j.contains("metadata") && j["metadata"].is_object())
    {
        const json &m = j["metadata"];
        name = m.value("name", std::string());
        // Focal length is not geometry (it is derived by tracing);
        // accept an optional metadata hint so callers that need a value before
        // a paraxial trace lands (e.g. Ghostlight's sensor-half-dimension calc)
        // can still get something sensible.
        focal_length = m.value("focal_length_mm", 0.0f);
    }

    // ---- Glass catalogue ----
    std::map<std::string, GlassEntry> catalogue;
    if (j.contains("glass_catalogue"))
    {
        if (!parse_glass_catalogue(j["glass_catalogue"], catalogue, filename))
            return false;
    }

    // ---- Walk optical_system + pivots and flatten ----
    if (!j.contains("optical_system"))
    {
        fprintf(stderr, "ERROR: %s: missing optical_system array\n", filename);
        return false;
    }
    if (!flatten_optical_system(j, catalogue, *this, filename))
        return false;

    if (surfaces.empty())
    {
        fprintf(stderr, "ERROR: %s: optical_system produced no surfaces\n", filename);
        return false;
    }

    // Post-flatten coating pass: surface IORs are final here, so physical
    // layer stacks can be baked to lookup tables, then every surface's
    // coating pointers are patched into this system's coating_tables.
    for (int i = 0; i < (int)surfaces.size(); ++i)
    {
        std::string bake_err;
        if (!bake_coating_layers(i, &bake_err))
            fprintf(stderr, "WARNING: %s: surface %d: %s; treating as "
                    "uncoated.\n", filename, i, bake_err.c_str());
    }
    sync_coating_pointers();
    sync_aperture_profiles();

    // Surface z's are rebased by flatten_optical_system so the chain ends at
    // z=0 (the sensor).
    return true;
}

bool OpticalSystem::finalize()
{
    if (surfaces.empty())
        return false;

    // Walk backward from sensor=0: sum thicknesses to get total axial extent,
    // then lay surfaces starting at -total. The last surface's thickness is
    // the back focal distance and brings the chain end to exactly 0.
    float total = 0.0f;
    for (const auto& s : surfaces)
        total += s.thickness;

    float z = -total;
    for (auto& s : surfaces)
    {
        s.z = z;
        z  += s.thickness;
    }

    // Resize the parallel vectors so programmatic builders don't have to.
    surface_ids.resize(surfaces.size());
    aperture_images.resize(surfaces.size());
    coating_tables.resize(surfaces.size());
    sync_coating_pointers();
    sync_aperture_profiles();
    return true;
}

// ---------------------------------------------------------------------------
// Coating table pointer sync / TMM bake / state hash
// ---------------------------------------------------------------------------

void OpticalSystem::sync_coating_pointers()
{
    if (coating_tables.size() < surfaces.size())
        coating_tables.resize(surfaces.size());

    for (size_t i = 0; i < surfaces.size(); ++i)
    {
        Coating &c            = surfaces[i].coating;
        const CoatingTables &t = coating_tables[i];

        c.table       = t.table.empty() ? nullptr : t.table.data();
        c.table_count = (int)t.table.size();

        const bool sa_ok = !t.sa_r.empty()
                        && !t.sa_wavelengths.empty()
                        && !t.sa_angles.empty()
                        && t.sa_r.size() == t.sa_wavelengths.size()
                                          * t.sa_angles.size();
        c.sa_wavelengths   = sa_ok ? t.sa_wavelengths.data() : nullptr;
        c.sa_angles        = sa_ok ? t.sa_angles.data()      : nullptr;
        c.sa_r             = sa_ok ? t.sa_r.data()           : nullptr;
        c.sa_n_wavelengths = sa_ok ? (int)t.sa_wavelengths.size() : 0;
        c.sa_n_angles      = sa_ok ? (int)t.sa_angles.size()      : 0;
    }
}

bool OpticalSystem::bake_coating_layers(int i, std::string *err)
{
    if (i < 0 || i >= (int)surfaces.size())
    {
        if (err) *err = "bake_coating_layers: surface index out of range";
        return false;
    }
    if (coating_tables.size() < surfaces.size())
        coating_tables.resize(surfaces.size());

    CoatingTables &tabs = coating_tables[i];
    if (tabs.layers.empty())
        return true; // nothing to bake

    // Validate specs before evaluating anything.
    for (const CoatingLayerSpec &spec : tabs.layers)
    {
        if (spec.nk_lambda_um.empty() || spec.thickness_nm <= 0.0f)
        {
            if (err) *err = "coating layer stack has a layer with no nk data "
                            "or non-positive thickness";
            tabs.layers.clear();
            return false;
        }
    }

    // Interpolate a layer's complex index at lambda (nm; nk tables are μm).
    auto layer_n_at = [](const CoatingLayerSpec &spec, float lambda_nm)
        -> std::complex<float>
    {
        const float lu = lambda_nm * 1e-3f;
        const std::vector<float> &xs = spec.nk_lambda_um;
        const size_t n = xs.size();
        if (lu <= xs.front())
            return { spec.nk_n.front(), spec.nk_k.front() };
        if (lu >= xs.back())
            return { spec.nk_n.back(), spec.nk_k.back() };
        for (size_t k = 1; k < n; ++k)
        {
            if (lu <= xs[k])
            {
                float span = xs[k] - xs[k - 1];
                float f    = (span > 0.0f) ? (lu - xs[k - 1]) / span : 0.0f;
                return { spec.nk_n[k - 1] + f * (spec.nk_n[k] - spec.nk_n[k - 1]),
                         spec.nk_k[k - 1] + f * (spec.nk_k[k] - spec.nk_k[k - 1]) };
            }
        }
        return { spec.nk_n.back(), spec.nk_k.back() };
    };

    // Bake grid: λ 400–700 nm step 10 (matches the renderer's hardcoded
    // spectral range), AOI 0–85° step 5 in the reference (ambient) medium.
    constexpr float kLambdaMin = 400.0f, kLambdaMax = 700.0f, kLambdaStep = 10.0f;
    constexpr float kAngleMin  = 0.0f,   kAngleMax  = 85.0f,  kAngleStep  = 5.0f;
    const int n_wl  = (int)((kLambdaMax - kLambdaMin) / kLambdaStep) + 1; // 31
    const int n_ang = (int)((kAngleMax - kAngleMin) / kAngleStep) + 1;    // 18

    tabs.sa_wavelengths.resize(n_wl);
    tabs.sa_angles.resize(n_ang);
    tabs.sa_r.assign((size_t)n_wl * n_ang, 0.0f);

    for (int a = 0; a < n_ang; ++a)
        tabs.sa_angles[a] = kAngleMin + a * kAngleStep;

    std::vector<tmm::Layer> layers(tabs.layers.size());
    for (int w = 0; w < n_wl; ++w)
    {
        const float lambda = kLambdaMin + w * kLambdaStep;
        tabs.sa_wavelengths[w] = lambda;

        const float n_amb = ior_before(i, lambda);
        const float n_sub = surfaces[i].ior_at(lambda);
        for (size_t l = 0; l < tabs.layers.size(); ++l)
        {
            layers[l].n            = layer_n_at(tabs.layers[l], lambda);
            layers[l].thickness_nm = tabs.layers[l].thickness_nm;
        }

        for (int a = 0; a < n_ang; ++a)
            tabs.sa_r[(size_t)w * n_ang + a] =
                tmm::reflectance(layers, { n_amb, 0.0f }, { n_sub, 0.0f },
                                 lambda, tabs.sa_angles[a]);
    }

    Coating &c = surfaces[i].coating;
    c.model                = CoatingModel::SPECTRAL_ANGULAR;
    c.out_of_range_discard = false;
    // The bake's angle axis is measured in the actual ambient medium of this
    // surface, so lookups convert against that medium's d-line IOR.
    c.angle_ref_ior        = ior_before(i, 587.56f);

    sync_coating_pointers();
    return true;
}

uint64_t OpticalSystem::coating_state_hash() const
{
    // FNV-1a, 64-bit.
    uint64_t h = 1469598103934665603ull;
    auto mix_bytes = [&h](const void *p, size_t n)
    {
        const unsigned char *b = (const unsigned char *)p;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 1099511628211ull;
        }
    };
    auto mix_f = [&](float v)         { mix_bytes(&v, sizeof(v)); };
    auto mix_i = [&](int v)           { mix_bytes(&v, sizeof(v)); };

    for (size_t i = 0; i < surfaces.size(); ++i)
    {
        const Coating &c = surfaces[i].coating;
        mix_i((int)c.model);
        mix_i(c.ar_layers);
        mix_i(c.out_of_range_discard ? 1 : 0);
        mix_f(c.gauss_sigma);      mix_f(c.gauss_background);
        mix_f(c.gauss_peak);       mix_f(c.gauss_decenter_x);
        mix_f(c.gauss_decenter_y);
        mix_f(c.angle_ref_ior);
        mix_f(c.tint_r); mix_f(c.tint_g); mix_f(c.tint_b);
        mix_f(c.tint_strength);

        if (i < coating_tables.size())
        {
            const CoatingTables &t = coating_tables[i];
            mix_i((int)t.table.size());
            if (!t.table.empty())
                mix_bytes(t.table.data(),
                          t.table.size() * sizeof(CoatingTable1D));
            mix_i((int)t.sa_wavelengths.size());
            mix_i((int)t.sa_angles.size());
            if (!t.sa_r.empty())
                mix_bytes(t.sa_r.data(), t.sa_r.size() * sizeof(float));
            mix_i((int)t.layers.size());
            for (const CoatingLayerSpec &spec : t.layers)
            {
                mix_bytes(spec.material.data(), spec.material.size());
                mix_f(spec.thickness_nm);
                if (!spec.nk_lambda_um.empty())
                    mix_bytes(spec.nk_lambda_um.data(),
                              spec.nk_lambda_um.size() * sizeof(float));
                if (!spec.nk_n.empty())
                    mix_bytes(spec.nk_n.data(), spec.nk_n.size() * sizeof(float));
                if (!spec.nk_k.empty())
                    mix_bytes(spec.nk_k.data(), spec.nk_k.size() * sizeof(float));
            }
        }
    }
    return h;
}

uint64_t OpticalSystem::aperture_image_state_hash() const
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* data, size_t size)
    {
        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            h ^= bytes[i];
            h *= 1099511628211ull;
        }
    };
    for (const ApertureImage& image : aperture_images)
    {
        mix(&image.width, sizeof(image.width));
        mix(&image.height, sizeof(image.height));
        mix(&image.semi_diameter, sizeof(image.semi_diameter));
        const size_t count = image.pixels.size();
        mix(&count, sizeof(count));
        if (!image.pixels.empty())
            mix(image.pixels.data(), image.pixels.size() * sizeof(float));
    }
    return h;
}

void OpticalSystem::insert_surface(int index, const Surface& surface,
                                   const std::string& surface_id)
{
    if (index < 0 || index > (int)surfaces.size())
        throw std::out_of_range("surface index out of range");
    surface_ids.resize(surfaces.size());
    aperture_images.resize(surfaces.size());
    coating_tables.resize(surfaces.size());
    surfaces.insert(surfaces.begin() + index, surface);
    surface_ids.insert(surface_ids.begin() + index, surface_id);
    aperture_images.insert(aperture_images.begin() + index, ApertureImage{});
    coating_tables.insert(coating_tables.begin() + index, CoatingTables{});
    sync_coating_pointers();
}

void OpticalSystem::erase_surface(int index)
{
    if (index < 0 || index >= (int)surfaces.size())
        throw std::out_of_range("surface index out of range");
    surface_ids.resize(surfaces.size());
    aperture_images.resize(surfaces.size());
    coating_tables.resize(surfaces.size());
    surfaces.erase(surfaces.begin() + index);
    surface_ids.erase(surface_ids.begin() + index);
    aperture_images.erase(aperture_images.begin() + index);
    coating_tables.erase(coating_tables.begin() + index);
    sync_coating_pointers();
}

void OpticalSystem::print_summary() const
{
    printf("OpticalSystem: %s\n", name.c_str());
    printf("  focal_length: %.1f mm\n", focal_length);
    printf("  surfaces: %d\n", num_surfaces());
    printf("  sensor at z=0 (convention)\n");
    printf("  %-4s  %-10s  %10s  %8s  %6s  %6s  %8s  %4s  %s\n",
           "Idx", "Form", "Radius", "Thick", "IOR", "Abbe", "SemiAp", "Coat", "");

    auto form_name = [](int f) {
        switch (f)
        {
            case FORM_SPHERE:      return "sphere";
            case FORM_ASPHERE:     return "asphere";
            case FORM_CYLINDRICAL: return "cylindrical";
            default:               return "?";
        }
    };

    for (int i = 0; i < num_surfaces(); ++i)
    {
        const Surface &s = surfaces[i];
        if (s.form == FORM_SPHERE && s.radius == 0.0f)
        {
            printf("  %-4d  %-10s  %10s  %8.3f  %6.3f  %6.1f  %8.2f  %4d  %s\n",
                   i, form_name(s.form),
                   s.is_stop ? "STOP" : "flat",
                   s.thickness, s.ior, s.abbe_v, s.semi_aperture,
                   s.coating.ar_layers,
                   s.is_stop ? "<-- aperture stop" : "");
        }
        else
        {
            printf("  %-4d  %-10s  %10.3f  %8.3f  %6.3f  %6.1f  %8.2f  %4d%s\n",
                   i, form_name(s.form),
                   s.radius, s.thickness, s.ior, s.abbe_v,
                   s.semi_aperture, s.coating.ar_layers,
                   s.is_stop ? "  <-- aperture stop" : "");
        }
    }
}
