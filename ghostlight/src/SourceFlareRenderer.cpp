// ============================================================================
// SourceFlareRenderer.cpp — Extended (area) source ghost renderer
//
// Generalizes the single-point renderer: the source is a set of collimated
// directions (angular offsets around the screen-space center) whose weighted
// contributions accumulate in one launch_ghost_render call.  The point
// renderer delegates here with a single zero-offset row.
// ============================================================================

#include "SourceFlareRenderer.h"
#include "diffraction.h"
#include "veil_render.h"
#include "gate_render.h"
#include "ghost_render.h"
#include "source_map.h"
#include "spectral.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

bool render_source_flare(int width, int height,
                         const float*            offsets,
                         int                     n_offsets,
                         const OpticalSystem&    lens,
                         const LensCalibration&  calib,
                         const PointFlareConfig& cfg,
                         FlareBuffers&           out)
{
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "SourceFlareRenderer: width and height must be positive\n");
        return false;
    }
    if (lens.num_surfaces() <= 0) {
        fprintf(stderr, "SourceFlareRenderer: lens has no surfaces\n");
        return false;
    }
    if (n_offsets < 1 || offsets == nullptr) {
        fprintf(stderr, "SourceFlareRenderer: no source offsets\n");
        return false;
    }

    const int    w   = width;
    const int    h   = height;
    if ((size_t)w > std::numeric_limits<size_t>::max() / (size_t)h) {
        fprintf(stderr, "SourceFlareRenderer: image dimensions overflow\n");
        return false;
    }
    const size_t npx = (size_t)w * h;
    if (npx > (size_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "SourceFlareRenderer: image has too many pixels\n");
        return false;
    }
    FlareBuffers next;

    std::vector<float> ghost_r(npx, 0.0f), ghost_g(npx, 0.0f), ghost_b(npx, 0.0f);

    // source_x/y are screen/image-space coordinates ([0,1], 0.5 = optical axis).
    // Solved to a field angle by tracing (source_map.h): the angle whose beam
    // actually images at that point on this lens. This keeps traced and
    // splatted layers aligned on distorting or anamorphic lenses.
    const float eff_half_w_pf = (cfg.sensor_half_w > 0.0f) ? cfg.sensor_half_w : calib.sensor_half_w;
    const float eff_half_h_pf = (cfg.sensor_half_h > 0.0f) ? cfg.sensor_half_h : calib.sensor_half_h;

    const SourceMapSolve smap = solve_source_angle(lens, calib,
                                                   (cfg.source_x - 0.5f) * 2.0f,
                                                   (cfg.source_y - 0.5f) * 2.0f,
                                                   eff_half_w_pf, eff_half_h_pf);
    const float angle_x0 = smap.angle_x;
    const float angle_y0 = smap.angle_y;

    // Convert source color to output color space (input_cs -> XYZ -> output_cs).
    float M_in[3][3], M_out[3][3];
    resolve_input_matrix(cfg.input_cs, cfg.custom_input_to_xyz, M_in);
    resolve_output_matrix(cfg.output_cs, cfg.custom_xyz_to_output, M_out);

    const float r = cfg.source_r, g = cfg.source_g, b = cfg.source_b;
    const float xyz_r = M_in[0][0]*r + M_in[0][1]*g + M_in[0][2]*b;
    const float xyz_g = M_in[1][0]*r + M_in[1][1]*g + M_in[1][2]*b;
    const float xyz_b = M_in[2][0]*r + M_in[2][1]*g + M_in[2][2]*b;

    const float src_r = M_out[0][0]*xyz_r + M_out[0][1]*xyz_g + M_out[0][2]*xyz_b;
    const float src_g = M_out[1][0]*xyz_r + M_out[1][1]*xyz_g + M_out[1][2]*xyz_b;
    const float src_b = M_out[2][0]*xyz_r + M_out[2][1]*xyz_g + M_out[2][2]*xyz_b;

    // Each source's sensor position in buffer pixels, for the starburst splat,
    // so the starburst sits exactly on the source's direct image.
    //
    // Taken from the solve's forward map, which is the same relation the angle
    // above came out of — so the splat lands on the traced image rather than on
    // an independent formula that only agrees with it near the axis. For the
    // zero-offset sample it reproduces the solved landing exactly; for the rest
    // it is the local Jacobian, which costs no tracing and therefore cannot
    // make a chunked render differ from a single one.
    std::vector<FlareSource> sources(n_offsets);
    std::vector<float>       src_px(n_offsets), src_py(n_offsets);
    for (int i = 0; i < n_offsets; ++i)
    {
        const float dax    = offsets[i * 3 + 0];
        const float day    = offsets[i * 3 + 1];
        const float weight = offsets[i * 3 + 2];
        if (!std::isfinite(dax) || !std::isfinite(day) || !std::isfinite(weight)) {
            fprintf(stderr, "SourceFlareRenderer: source offsets must be finite\n");
            return false;
        }

        FlareSource& bp = sources[i];
        bp.angle_x = angle_x0 + dax;
        bp.angle_y = angle_y0 + day;
        bp.r = src_r * weight;
        bp.g = src_g * weight;
        bp.b = src_b * weight;

        float sx = 0.0f, sy = 0.0f;
        source_map_screen(smap, bp.angle_x, bp.angle_y, &sx, &sy);
        src_px[i] = sx * w;
        src_py[i] = sy * h;
    }

    std::vector<FlareAovLayer>* aov_ptr =
        (cfg.aov_mode != GhostAovMode::NONE) ? &next.aov_layers : nullptr;

    std::string    cuda_error;
    next.has_stats = cfg.collect_stats;
    GhostRenderStats* stats_ptr = cfg.collect_stats ? &next.stats : nullptr;
    if (!render_ghost_pipeline(lens, calib, sources,
                               w, h, w, h, 0, 0,
                               ghost_r.data(), ghost_g.data(), ghost_b.data(),
                               cfg, &cuda_error, aov_ptr, stats_ptr)) {
        fprintf(stderr, "SourceFlareRenderer: CUDA error: %s\n", cuda_error.c_str());
        return false;
    }

    next.ghost_r = std::move(ghost_r);
    next.ghost_g = std::move(ghost_g);
    next.ghost_b = std::move(ghost_b);

    // Starburst diffraction layer, returned as a separate additive buffer.
    if (cfg.diffraction.starburst)
    {
        std::vector<float> sb_r(npx, 0.0f), sb_g(npx, 0.0f), sb_b(npx, 0.0f);
        std::string sb_err;
        if (render_starburst(lens, calib, sources, src_px, src_py,
                             w, h, w, h, eff_half_w_pf, eff_half_h_pf, cfg,
                             sb_r.data(), sb_g.data(), sb_b.data(), &sb_err))
        {
            next.starburst_r = std::move(sb_r);
            next.starburst_g = std::move(sb_g);
            next.starburst_b = std::move(sb_b);
        }
        else
        {
            fprintf(stderr, "SourceFlareRenderer: starburst error: %s\n", sb_err.c_str());
            return false;
        }
    }

    // Veiling-glare layer, returned separately for metering.
    if (cfg.diffraction.veil)
    {
        std::vector<float> ve_r(npx, 0.0f), ve_g(npx, 0.0f), ve_b(npx, 0.0f);
        std::string ve_err;
        if (render_veil(lens, calib, sources, src_px, src_py,
                        w, h, w, h, eff_half_w_pf, eff_half_h_pf, cfg,
                        ve_r.data(), ve_g.data(), ve_b.data(), &ve_err))
        {
            next.veil_r = std::move(ve_r);
            next.veil_g = std::move(ve_g);
            next.veil_b = std::move(ve_b);
        }
        else
        {
            fprintf(stderr, "SourceFlareRenderer: veil error: %s\n", ve_err.c_str());
            return false;
        }
    }

    // Film-gate scatter layer, returned as a separate additive buffer.
    //
    // eff_half_*_pf, not calib.sensor_half_*: the gate opening derives from the
    // rendered frame, so under a sensor override the raw calibrated value would
    // stand the wall off the frame edge and the fold would land out of view.
    if (cfg.gate.enabled)
    {
        std::vector<float> ga_r(npx, 0.0f), ga_g(npx, 0.0f), ga_b(npx, 0.0f);
        std::string ga_err;
        if (render_gate(lens, calib, sources,
                        w, h, w, h, 0, 0, eff_half_w_pf, eff_half_h_pf, cfg,
                        ga_r.data(), ga_g.data(), ga_b.data(), &ga_err))
        {
            next.gate_r = std::move(ga_r);
            next.gate_g = std::move(ga_g);
            next.gate_b = std::move(ga_b);
        }
        else
        {
            fprintf(stderr, "SourceFlareRenderer: gate error: %s\n", ga_err.c_str());
            return false;
        }
    }

    next.width = w;
    next.height = h;
    out = std::move(next);
    return true;
}
