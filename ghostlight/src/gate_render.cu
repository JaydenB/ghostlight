// ============================================================================
// gate_render.cu — GPU film-gate scatter kernel and CPU launcher.
//
// Architecturally the slim cousin of psf_render.cu: same aperture sampler, same
// spectral table, same primary trace, same bilinear splat — minus the tiles, and
// with the gate band test / mirror fold / scatter lobe (gate.h) applied to every
// ray that reaches the sensor plane outside the opening.
//
// One thread owns one (source, pupil sample) and loops over wavelengths. The
// four wall tests sit outside the scatter loop so a ray that scrapes nothing
// costs four compares and no draws.
// ============================================================================

#include "gate_render.h"

#include "aperture_sampler.h"
#include "aperture_textures.h"
#include "coating_upload.h"
#include "spawn_plane.h"  // SPAWN_OFFSET, spawn_shift
#include "spectral.h"
#include "trace_cuda.h"   // d_trace_primary_ray

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int BLOCK = 256;

// Per-source payload: the field angle the beam arrives at, and the source's
// colour in the output space (weight already folded in by the caller).
struct GPUGateSource { float angle_x, angle_y, r, g, b; };

// Persistent device scratch (leaked singleton + mutex, mirroring VeilGpu in
// veil_render.cu: never destroyed, so no cudaFree races static CUDA teardown).
struct GateGpu {
    Surface*           d_surfs   = nullptr;  size_t surfs_bytes = 0;
    void*              d_sources = nullptr;  size_t src_bytes   = 0;
    void*              d_grid    = nullptr;  size_t grid_bytes  = 0;
    void*              d_spec    = nullptr;  size_t spec_bytes  = 0;
    float*             d_out_r   = nullptr;
    float*             d_out_g   = nullptr;
    float*             d_out_b   = nullptr;
    float*             d_dir_r   = nullptr;  // direct-image capture (debug only)
    float*             d_dir_g   = nullptr;
    float*             d_dir_b   = nullptr;
    size_t             out_floats = 0;
    size_t             dir_floats = 0;
    unsigned long long* d_counters = nullptr;   // [traces, scrapes] (debug only)
    unsigned int*      d_reach    = nullptr;    // max reach, float bits (debug only)
};

std::mutex g_gate_mutex;
GateGpu&   gate_gpu() { static GateGpu* c = new GateGpu(); return *c; }

bool ensure_bytes(void*& ptr, size_t& cap, size_t need) {
    if (need <= cap) return true;
    cudaFree(ptr); ptr = nullptr; cap = 0;
    if (cudaMalloc(&ptr, need) != cudaSuccess) return false;
    cap = need; return true;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK) void gate_kernel(
    const Surface*             d_surfs,
    int                        n_surfs,
    const GPUGateSource*       d_sources,
    int                        n_sources,
    const ApertureSample*      d_grid,
    int                        n_grid,
    float                      front_R,
    float                      start_z,
    const GPUSpectralSample*   d_spec,
    int                        n_spec,
    GpuGate                    gate,
    unsigned int               seed_base,
    float                      ray_weight,
    float                      sensor_half_w,
    float                      sensor_half_h,
    int                        fmt_w,
    int                        fmt_h,
    int                        fmt_x0_in_buf,
    int                        fmt_y0_in_buf,
    int                        width,
    int                        height,
    float*                     d_out_r,
    float*                     d_out_g,
    float*                     d_out_b,
    float*                     d_dir_r,        // nullable: direct-image capture
    float*                     d_dir_g,
    float*                     d_dir_b,
    unsigned long long*        d_counters,     // nullable: [traces, scrapes]
    unsigned int*              d_reach,        // nullable: max realised reach
    const cudaTextureObject_t* d_aperture_textures)
{
    const int src_idx = (int)blockIdx.x;
    if (src_idx >= n_sources) return;

    const int grid_idx = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (grid_idx >= n_grid) return;

    const GPUGateSource&  src = d_sources[src_idx];
    const ApertureSample& gs  = d_grid[grid_idx];

    const float bx = tanf(src.angle_x);
    const float by = tanf(src.angle_y);
    const Vec3f beam_dir = Vec3f(bx, by, 1.0f).normalized();

    // Track the beam off axis (spawn_plane.h); exactly zero on axis.
    float sdx, sdy;
    spawn_shift(bx, by, sdx, sdy);

    Ray base_ray;
    base_ray.origin = Vec3f(gs.u * front_R + sdx, gs.v * front_R + sdy, start_z);
    base_ray.dir    = beam_dir;
    base_ray.lambda = 0.0f;

    // Use source angle rather than chunk-local src_idx so the random stream is
    // invariant to source batching.
    const unsigned int ray_seed =
          __float_as_uint(src.angle_x) * 0x9E3779B9u
        + __float_as_uint(src.angle_y) * 0x85EBCA6Bu
        + (unsigned int)grid_idx       * 0xC2B2AE35u
        + seed_base;

    // A mirror wall needs no draws, and n_scatter copies of one deterministic
    // landing would be pure waste, so collapse to a single full-weight sample.
    const bool  stochastic = (gate.sig_wide > 0.0f);
    const int   n_k        = stochastic ? gate.n_scatter : 1;
    const float inv_k      = stochastic ? gate.inv_scatter : 1.0f;

    for (int si = 0; si < n_spec; ++si)
    {
        const GPUSpectralSample& spec = d_spec[si];

        Ray r    = base_ray;
        r.lambda = spec.lambda;

        Vec3f dir;
        TraceResult res = d_trace_primary_ray(r, d_surfs, n_surfs,
                                              d_aperture_textures, &dir);
        if (d_counters) atomicAdd(&d_counters[0], 1ull);
        if (res.status != TraceStatus::OK) continue;
        if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;
        if (!(fabsf(dir.z) > 1.0e-12f)) continue;

        const float w_base = res.weight * ray_weight;

        // The direct image, from the same traces. Only the debug capture wants
        // it — the flare renderers never draw the source itself — but it is the
        // only radiometric baseline the gate's peak can be quoted against.
        if (d_dir_r)
        {
            const float dpx = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
            const float dpy = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;
            const int dxi = (int)floorf(dpx);
            const int dyi = (int)floorf(dpy);
            if (dxi >= 0 && dxi < width && dyi >= 0 && dyi < height)
            {
                const int pix = dyi * width + dxi;
                atomicAdd(&d_dir_r[pix], src.r * spec.cmf_r * w_base);
                atomicAdd(&d_dir_g[pix], src.g * spec.cmf_g * w_base);
                atomicAdd(&d_dir_b[pix], src.b * spec.cmf_b * w_base);
            }
        }

        const float sx = dir.x / dir.z;
        const float sy = dir.y / dir.z;

        // Walls, in order: +x, -x, +y, -y.
        for (int wall = 0; wall < 4; ++wall)
        {
            const int   axis = wall >> 1;
            const float sgn  = (wall & 1) ? -1.0f : 1.0f;

            float edge, p_axis, s_axis, p_other, s_other, o_lo, o_hi, n_axis;
            if (axis == 0) {
                edge   = (sgn > 0.0f) ? gate.x_pos : gate.x_neg;
                p_axis = res.position.x; s_axis = sx; n_axis = dir.x;
                p_other = res.position.y; s_other = sy;
                o_lo = gate.y_neg; o_hi = gate.y_pos;
            } else {
                edge   = (sgn > 0.0f) ? gate.y_pos : gate.y_neg;
                p_axis = res.position.y; s_axis = sy; n_axis = dir.y;
                p_other = res.position.x; s_other = sx;
                o_lo = gate.x_neg; o_hi = gate.x_pos;
            }

            float back, fold;
            if (!gate_wall_scrape(gate, p_axis, s_axis, edge, sgn, back, fold)) continue;

            // Where it meets the wall along the wall's own length. Outside the
            // opening on that axis it misses the plate entirely, so a ray
            // diagonally beyond two walls strikes neither. That under-counts the
            // four corners; a genuine corner double-bounce lands elsewhere and
            // is not modelled.
            const float hit_other = p_other - s_other * back;
            if (hit_other <= o_lo || hit_other >= o_hi) continue;

            if (d_counters) atomicAdd(&d_counters[1], 1ull);
            if (d_reach) {
                const float reach = fabsf(fold - edge);
                atomicMax(d_reach, __float_as_uint(reach));
            }

            // dir is normalised, so the component along the wall normal IS the
            // cosine to it. Grazing rays have cos_i -> 0 and Schlick -> 1.
            const float R = gate_schlick(fabsf(n_axis), gate.r0);
            const float w_hit = w_base * R * gate.gain * inv_k;
            if (!(w_hit > 0.0f)) continue;

            for (int k = 0; k < n_k; ++k)
            {
                // Every draw is a pure function of stable identities with no
                // stream carried between them, so a ray that skips a wall cannot
                // desynchronise any other ray — a stronger guarantee than HURB's
                // skip-without-drawing discipline, and what makes a chunked
                // render match an unchunked one exactly.
                unsigned int state = gate_wang_hash(
                      ray_seed
                    ^ ((unsigned int)si   * 0x9E3779B9u)
                    ^ ((unsigned int)wall * 0x85EBCA6Bu)
                    ^ ((unsigned int)k    * 0x27D4EB2Fu));

                float k_axis = 0.0f, k_other = 0.0f, k_z = 0.0f;
                if (stochastic) gate_sample_lobe(gate, state, k_axis, k_other, k_z);

                const float d_ax = -sgn * fabsf(s_axis) + k_axis;
                const float d_ot = s_other + k_other;
                const float d_z  = 1.0f + k_z;
                if (!(d_z > 1.0e-6f)) continue;   // scattered back toward the lens

                const float land_axis  = edge      + d_ax / d_z * back;
                const float land_other = hit_other + d_ot / d_z * back;

                const float lx = (axis == 0) ? land_axis  : land_other;
                const float ly = (axis == 0) ? land_other : land_axis;

                float px = (lx / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
                float py = (ly / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;
                if (!isfinite(px) || !isfinite(py)) continue;
                px = fmaxf(-2.0e9f, fminf(2.0e9f, px));
                py = fmaxf(-2.0e9f, fminf(2.0e9f, py));

                const int   x0 = (int)floorf(px - 0.5f);
                const int   y0 = (int)floorf(py - 0.5f);
                const float fx = (px - 0.5f) - (float)x0;
                const float fy = (py - 0.5f) - (float)y0;

                // Per-channel accumulation, as psf_kernel does rather than the
                // ghost kernel's luminance-weighted upsample: the wall is
                // achromatic, so a white source must integrate to a neutral
                // layer. The colour that does show is the lens's own dispersion
                // moving each wavelength's landing across the band.
                const float cr = src.r * spec.cmf_r * w_hit;
                const float cg = src.g * spec.cmf_g * w_hit;
                const float cb = src.b * spec.cmf_b * w_hit;

#define GATE_SPLAT(xi, yi, wt)                                                   \
                if ((xi) >= 0 && (xi) < width && (yi) >= 0 && (yi) < height) {   \
                    const int _p = (yi) * width + (xi);                          \
                    if (fabsf(cr) > 1e-16f) atomicAdd(&d_out_r[_p], cr * (wt));  \
                    if (fabsf(cg) > 1e-16f) atomicAdd(&d_out_g[_p], cg * (wt));  \
                    if (fabsf(cb) > 1e-16f) atomicAdd(&d_out_b[_p], cb * (wt));  \
                }
                GATE_SPLAT(x0,     y0,     (1.0f - fx) * (1.0f - fy))
                GATE_SPLAT(x0 + 1, y0,     fx          * (1.0f - fy))
                GATE_SPLAT(x0,     y0 + 1, (1.0f - fx) * fy)
                GATE_SPLAT(x0 + 1, y0 + 1, fx          * fy)
#undef GATE_SPLAT
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
bool render_gate(const OpticalSystem&            lens,
                 const LensCalibration&          calib,
                 const std::vector<FlareSource>& sources,
                 int                             width,
                 int                             height,
                 int                             fmt_w,
                 int                             fmt_h,
                 int                             fmt_x0_in_buf,
                 int                             fmt_y0_in_buf,
                 float                           sensor_half_w,
                 float                           sensor_half_h,
                 const FlareConfig&              cfg,
                 float*                          out_r,
                 float*                          out_g,
                 float*                          out_b,
                 std::string*                    out_error,
                 GateDebug*                      dbg)
{
    if (!cfg.gate.enabled || sources.empty()) return true;

    auto fail = [&](const char* msg) {
        if (out_error && out_error->empty()) *out_error = msg;
        fprintf(stderr, "render_gate: %s\n", msg);
        return false;
    };

    if (width <= 0 || height <= 0 || fmt_w <= 0 || fmt_h <= 0)
        return fail("degenerate output dimensions");
    if (lens.num_surfaces() <= 0) return fail("lens has no surfaces");
    if (!(sensor_half_w > 0.0f) || !(sensor_half_h > 0.0f))
        return fail("degenerate sensor half-size");

    const GpuGate gate = build_gpu_gate(cfg.gate, sensor_half_w, sensor_half_h);

    // ---- Entrance-pupil grid (the shared sampler, so the silhouette cannot
    // drift from the ghost / starburst view of the same lens) ----------------
    ApertureSamplerParams sp;
    sp.ray_grid        = cfg.ray_grid;
    sp.pupil_jitter    = cfg.pupil_jitter;
    sp.jitter_seed     = cfg.jitter_seed;
    sp.blades_override = cfg.aperture_blades;
    sp.rotation_deg    = cfg.aperture_rotation;
    const std::vector<ApertureSample> grid_samples = build_aperture_samples(lens, sp);
    const int n_grid = (int)grid_samples.size();
    if (n_grid == 0) return true;          // nothing passes the stop: no flare

    const float ray_weight = 1.0f / (float)n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - SPAWN_OFFSET;

    // ---- Spectral table (the shared builder, pre-multiplied by dlambda so the
    // kernel's plain sum is the Riemann sum and brightness does not scale with
    // the sample count — the same correction psf_render.cu applies) ----------
    const int ns_cfg = (cfg.gate.spectral_samples > 0) ? cfg.gate.spectral_samples
                                                       : cfg.spectral_samples;
    const int ns = std::max(3, ns_cfg);
    std::vector<GPUSpectralSample> spectral_cpu;
    if (ns == 3)
    {
        spectral_cpu = { { 650.0f, 1.0f, 0.0f, 0.0f },
                         { 550.0f, 0.0f, 1.0f, 0.0f },
                         { 450.0f, 0.0f, 0.0f, 1.0f } };
    }
    else
    {
        const SensorProfile& prof = get_sensor_profile(cfg.sensor_model);
        float M_out[3][3];
        resolve_output_matrix(cfg.output_cs, cfg.custom_xyz_to_output, M_out);
        spectral_cpu = build_spectral_table(ns, 400.0f, 700.0f, prof, M_out);
        const float d_lambda = (700.0f - 400.0f) / (float)ns;
        for (auto& s : spectral_cpu) {
            s.cmf_r *= d_lambda;
            s.cmf_g *= d_lambda;
            s.cmf_b *= d_lambda;
        }
    }
    const int n_spec = (int)spectral_cpu.size();

    // ---- Sources ------------------------------------------------------------
    const int n_src = (int)sources.size();
    std::vector<GPUGateSource> host_src(n_src);
    for (int i = 0; i < n_src; ++i)
        host_src[i] = { sources[i].angle_x, sources[i].angle_y,
                        sources[i].r, sources[i].g, sources[i].b };

    std::lock_guard<std::mutex> lock(g_gate_mutex);
    GateGpu& g = gate_gpu();

    const int    n_surfs = lens.num_surfaces();
    const size_t n_px    = (size_t)width * height;

    // Table-backed coatings and image apertures own their own device memory with
    // their own lifetime, so they stay function-local RAII (as psf_render does)
    // rather than joining the persistent scratch.
    CoatingDeviceArena   coating_arena;
    std::vector<Surface> coated_surfs;
    if (!upload_coating_tables(lens, coating_arena, coated_surfs, out_error))
        return fail("coating table upload failed");
    const Surface* surfs_src = coated_surfs.empty() ? lens.surfaces.data()
                                                    : coated_surfs.data();

    ApertureTexturePack apertures;
    if (!upload_aperture_textures(lens, apertures, out_error))
        return fail("aperture texture upload failed");

    void* p_surfs = (void*)g.d_surfs;
    if (!ensure_bytes(p_surfs,    g.surfs_bytes, (size_t)n_surfs * sizeof(Surface)) ||
        !ensure_bytes(g.d_sources, g.src_bytes,  (size_t)n_src   * sizeof(GPUGateSource)) ||
        !ensure_bytes(g.d_grid,    g.grid_bytes, (size_t)n_grid  * sizeof(ApertureSample)) ||
        !ensure_bytes(g.d_spec,    g.spec_bytes, (size_t)n_spec  * sizeof(GPUSpectralSample)))
        return fail("cudaMalloc failed (gate inputs)");
    g.d_surfs = (Surface*)p_surfs;

    if (n_px > g.out_floats) {
        cudaFree(g.d_out_r); cudaFree(g.d_out_g); cudaFree(g.d_out_b);
        g.d_out_r = g.d_out_g = g.d_out_b = nullptr;
        g.out_floats = 0;
        if (cudaMalloc(&g.d_out_r, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_g, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_b, n_px * sizeof(float)) != cudaSuccess)
            return fail("cudaMalloc failed (gate output)");
        g.out_floats = n_px;
    }

    cudaMemcpy(g.d_surfs,   surfs_src,          n_surfs * sizeof(Surface),           cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_sources, host_src.data(),    n_src   * sizeof(GPUGateSource),     cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_grid,    grid_samples.data(),n_grid  * sizeof(ApertureSample),    cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_spec,    spectral_cpu.data(),n_spec  * sizeof(GPUSpectralSample), cudaMemcpyHostToDevice);
    cudaMemset(g.d_out_r, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_g, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_b, 0, n_px * sizeof(float));

    // ---- Debug-only captures ------------------------------------------------
    float* d_dir_r = nullptr; float* d_dir_g = nullptr; float* d_dir_b = nullptr;
    unsigned long long* d_counters = nullptr;
    unsigned int*       d_reach    = nullptr;
    if (dbg)
    {
        if (n_px > g.dir_floats) {
            cudaFree(g.d_dir_r); cudaFree(g.d_dir_g); cudaFree(g.d_dir_b);
            g.d_dir_r = g.d_dir_g = g.d_dir_b = nullptr;
            g.dir_floats = 0;
            if (cudaMalloc(&g.d_dir_r, n_px * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g.d_dir_g, n_px * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&g.d_dir_b, n_px * sizeof(float)) != cudaSuccess)
                return fail("cudaMalloc failed (gate direct capture)");
            g.dir_floats = n_px;
        }
        if (!g.d_counters && cudaMalloc(&g.d_counters, 2 * sizeof(unsigned long long)) != cudaSuccess)
            return fail("cudaMalloc failed (gate counters)");
        if (!g.d_reach && cudaMalloc(&g.d_reach, sizeof(unsigned int)) != cudaSuccess)
            return fail("cudaMalloc failed (gate reach)");
        cudaMemset(g.d_dir_r, 0, n_px * sizeof(float));
        cudaMemset(g.d_dir_g, 0, n_px * sizeof(float));
        cudaMemset(g.d_dir_b, 0, n_px * sizeof(float));
        cudaMemset(g.d_counters, 0, 2 * sizeof(unsigned long long));
        cudaMemset(g.d_reach, 0, sizeof(unsigned int));
        d_dir_r = g.d_dir_r; d_dir_g = g.d_dir_g; d_dir_b = g.d_dir_b;
        d_counters = g.d_counters; d_reach = g.d_reach;
    }

    // ---- Launch -------------------------------------------------------------
    const int  grid_blocks = (n_grid + BLOCK - 1) / BLOCK;
    const dim3 block(BLOCK, 1, 1);
    const dim3 grid_dim((unsigned)n_src, (unsigned)grid_blocks, 1);
    const unsigned int seed_base = (unsigned int)cfg.jitter_seed * 1000003u;

    if (cfg.verbose) {
        printf("Ghostlight gate: %d sources x %d pupil x %d spectral -> grid (%u, %u)\n",
               n_src, n_grid, n_spec, grid_dim.x, grid_dim.y);
        fflush(stdout);
    }

    gate_kernel<<<grid_dim, block>>>(
        g.d_surfs, n_surfs,
        (const GPUGateSource*)g.d_sources, n_src,
        (const ApertureSample*)g.d_grid, n_grid,
        front_R, start_z,
        (const GPUSpectralSample*)g.d_spec, n_spec,
        gate, seed_base, ray_weight,
        sensor_half_w, sensor_half_h,
        fmt_w, fmt_h, fmt_x0_in_buf, fmt_y0_in_buf,
        width, height,
        g.d_out_r, g.d_out_g, g.d_out_b,
        d_dir_r, d_dir_g, d_dir_b,
        d_counters, d_reach,
        static_cast<cudaTextureObject_t*>(apertures.d_textures));

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail(cudaGetErrorString(err));

    cudaMemcpy(out_r, g.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_g, g.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b, g.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost);

    if (dbg)
    {
        dbg->gate = gate;
        unsigned long long counters[2] = { 0ull, 0ull };
        unsigned int reach_bits = 0u;
        cudaMemcpy(counters, g.d_counters, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&reach_bits, g.d_reach, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        dbg->traces  = (long long)counters[0];
        dbg->scrapes = (long long)counters[1];
        float reach = 0.0f;
        std::memcpy(&reach, &reach_bits, sizeof(float));
        dbg->reach_mm = reach;

        dbg->direct_r.assign(n_px, 0.0f);
        dbg->direct_g.assign(n_px, 0.0f);
        dbg->direct_b.assign(n_px, 0.0f);
        cudaMemcpy(dbg->direct_r.data(), g.d_dir_r, n_px * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg->direct_g.data(), g.d_dir_g, n_px * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg->direct_b.data(), g.d_dir_b, n_px * sizeof(float), cudaMemcpyDeviceToHost);

        double e = 0.0;
        for (size_t i = 0; i < n_px; ++i) e += out_r[i] + out_g[i] + out_b[i];
        dbg->energy = e;

        const float nx = (calib.f_number_x > 0.0f) ? calib.f_number_x : 0.0f;
        const float ny = (calib.f_number_y > 0.0f) ? calib.f_number_y : 0.0f;
        dbg->band_x_mm = (nx > 0.0f) ? gate.t_mm / (2.0f * nx) : 0.0f;
        dbg->band_y_mm = (ny > 0.0f) ? gate.t_mm / (2.0f * ny) : 0.0f;
    }

    return true;
}
