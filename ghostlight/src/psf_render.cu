// ============================================================================
// psf_render.cu — GPU PSF scatter kernel and CPU launcher.
//
// Architecturally the slim cousin of ghost_render.cu: same aperture sampler,
// same spectral table, same bilinear splat — minus the ghost-pair iteration
// and reflected ghost legs. Each thread traces a primary ray (no reflections) and
// splats its hit, relative to the chief-ray landing, into the tile owned by
// its source.
// ============================================================================

#include "psf_render.h"

#include "aperture_sampler.h"
#include "aperture_textures.h"
#include "coating_upload.h"
#include "optical_system.h"
#include "spawn_plane.h"  // SPAWN_OFFSET, spawn_shift
#include "spectral.h"
#include "trace_cuda.h"   // d_trace_primary_ray

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ===========================================================================
// Scatter kernel
// ===========================================================================

static constexpr int BLOCK_SIZE = 256;

__global__ __launch_bounds__(BLOCK_SIZE) void psf_kernel(
    const Surface*           d_surfs,
    int                      n_surfs,
    const PSFGpuSource*      d_sources,
    int                      n_sources,
    const ApertureSample*    d_grid,
    int                      n_grid,
    float                    front_R,
    float                    start_z,
    float                    mm_per_pixel,
    int                      tile_w,
    int                      tile_h,
    int                      composite_w,
    int                      composite_h,
    float*                   d_out_r,
    float*                   d_out_g,
    float*                   d_out_b,
    float                    ray_weight,
    int                      monochromatic,        // 0/1 — kernel ABI is plain int
    const GPUSpectralSample* d_spec,
    int                      n_spec,
    float                    splat_sigma_px,       // <=0 = bilinear; >0 = Gaussian splat
    int                      splat_radius_px,      // splat footprint half-width (cells)
    const cudaTextureObject_t* d_aperture_textures)
{
    const int src_idx = (int)blockIdx.x;
    if (src_idx >= n_sources) return;

    const int grid_idx = (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;
    if (grid_idx >= n_grid) return;

    const PSFGpuSource&   src = d_sources[src_idx];
    const ApertureSample& gs  = d_grid[grid_idx];
    const float u = gs.u;
    const float v = gs.v;

    const float bx = tanf(src.angle_x);
    const float by = tanf(src.angle_y);
    const Vec3f beam_dir = Vec3f(bx, by, 1.0f).normalized();

    // Track the beam off axis (spawn_plane.h); exactly zero on axis.
    float sdx, sdy;
    spawn_shift(bx, by, sdx, sdy);

    Ray base_ray;
    base_ray.origin = Vec3f(u * front_R + sdx, v * front_R + sdy, start_z);
    base_ray.dir    = beam_dir;
    base_ray.lambda = 0.0f;

    // Tile-local bounds in composite-buffer coordinates.  Splats outside
    // these bounds are dropped so an off-centre PSF doesn't bleed into the
    // neighbouring tile (and the chief-ray-centred design means correctly
    // imaged rays land in [0, tile_w) x [0, tile_h) tile-local space).
    const int tx_min = src.tile_x0;
    const int ty_min = src.tile_y0;
    const int tx_max = src.tile_x0 + tile_w;
    const int ty_max = src.tile_y0 + tile_h;

    for (int si = 0; si < n_spec; ++si)
    {
        const GPUSpectralSample& spec = d_spec[si];

        Ray r    = base_ray;
        r.lambda = spec.lambda;

        TraceResult res = d_trace_primary_ray(r, d_surfs, n_surfs,
                                              d_aperture_textures);
        if (res.status != TraceStatus::OK) continue;
        if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;

        // Centre on chief ray, convert mm offset to tile-local pixel.
        const float dx_mm = res.position.x - src.chief_x_mm;
        const float dy_mm = res.position.y - src.chief_y_mm;
        const float tx    = dx_mm / mm_per_pixel + (float)tile_w * 0.5f;
        const float ty    = dy_mm / mm_per_pixel + (float)tile_h * 0.5f;

        // Early reject — widen the in-tile bounds by the splat radius so
        // edge rays whose Gaussian footprint still touches the tile aren't
        // dropped.  For pure bilinear (sigma<=0) splat_radius_px==0 and the
        // ±1 margin below still covers the bilinear footprint.
        const float margin = (float)(splat_radius_px > 0 ? splat_radius_px + 1 : 1);
        if (tx < -margin || tx > (float)tile_w + margin
         || ty < -margin || ty > (float)tile_h + margin) continue;

        const float px = tx + (float)src.tile_x0;
        const float py = ty + (float)src.tile_y0;

        // Per-channel accumulation: each input channel multiplies its own CMF,
        // not a luminance-weighted spectral upsample (which is what the *ghost*
        // kernel does — appropriate for colourful flares but introduces a
        // green-yellow bias for white sources because the CIE CMFs aren't
        // orthogonal).  This way a white source (r=g=b=1) integrates to a
        // white PSF in the output colour space; chromatic aberration shows
        // as legitimate fringes around the spot, not a global colour cast.
        const float w = res.weight * ray_weight;
        const float src_avg = (src.r + src.g + src.b) * (1.0f / 3.0f);
        const float lum = src_avg * spec.cmf_g * w;         // monochromatic value
        const float cr  = src.r * spec.cmf_r * w;
        const float cg  = src.g * spec.cmf_g * w;
        const float cb  = src.b * spec.cmf_b * w;

        if (splat_sigma_px <= 0.0f)
        {
            // ---- Bilinear single-pixel splat (4 cells) ----
            const int   x0 = (int)floorf(px - 0.5f);
            const int   y0 = (int)floorf(py - 0.5f);
            const float fx = (px - 0.5f) - (float)x0;
            const float fy = (py - 0.5f) - (float)y0;
            const float bw[4] = {
                (1.0f - fx) * (1.0f - fy),
                fx          * (1.0f - fy),
                (1.0f - fx) * fy,
                fx          * fy,
            };
            const int xs[4] = { x0, x0+1, x0,   x0+1 };
            const int ys[4] = { y0, y0,   y0+1, y0+1 };
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const int xi = xs[k], yi = ys[k];
                if (xi < tx_min || xi >= tx_max || yi < ty_min || yi >= ty_max
                 || xi < 0 || xi >= composite_w || yi < 0 || yi >= composite_h) continue;
                const int pix = yi * composite_w + xi;
                const float wt = bw[k];
                if (monochromatic) {
                    if (fabsf(lum) > 1e-14f) atomicAdd(&d_out_r[pix], lum * wt);
                } else {
                    if (fabsf(cr) > 1e-14f) atomicAdd(&d_out_r[pix], cr * wt);
                    if (fabsf(cg) > 1e-14f) atomicAdd(&d_out_g[pix], cg * wt);
                    if (fabsf(cb) > 1e-14f) atomicAdd(&d_out_b[pix], cb * wt);
                }
            }
        }
        else
        {
            // ---- Gaussian splat (footprint = (2r+1) × (2r+1)) ----
            // Centre cell index: the cell containing px is floor(px).  We
            // splat into cells (cx ± dx, cy ± dy) with weights from a
            // continuous Gaussian sampled at each cell centre — then
            // normalise so the total weight over splatted cells sums to 1.
            const float inv_2sig2 = 1.0f / (2.0f * splat_sigma_px * splat_sigma_px);
            const int cx = (int)floorf(px);
            const int cy = (int)floorf(py);
            const float ox = px - ((float)cx + 0.5f);   // (-0.5, 0.5)
            const float oy = py - ((float)cy + 0.5f);

            // Compute total weight first so we can normalise.  Splat radius
            // is small (typically 1-4), so a 2-pass loop is cheap.
            float wsum = 0.0f;
            for (int dy = -splat_radius_px; dy <= splat_radius_px; ++dy)
            {
                const float dyr = (float)dy - oy;
                for (int dx = -splat_radius_px; dx <= splat_radius_px; ++dx)
                {
                    const float dxr = (float)dx - ox;
                    wsum += expf(-(dxr*dxr + dyr*dyr) * inv_2sig2);
                }
            }
            const float inv_wsum = (wsum > 1e-20f) ? (1.0f / wsum) : 0.0f;

            for (int dy = -splat_radius_px; dy <= splat_radius_px; ++dy)
            {
                const int yi = cy + dy;
                if (yi < ty_min || yi >= ty_max || yi < 0 || yi >= composite_h) continue;
                const float dyr = (float)dy - oy;
                for (int dx = -splat_radius_px; dx <= splat_radius_px; ++dx)
                {
                    const int xi = cx + dx;
                    if (xi < tx_min || xi >= tx_max || xi < 0 || xi >= composite_w) continue;
                    const float dxr = (float)dx - ox;
                    const float wt = expf(-(dxr*dxr + dyr*dyr) * inv_2sig2) * inv_wsum;
                    if (wt < 1e-6f) continue;
                    const int pix = yi * composite_w + xi;
                    if (monochromatic) {
                        if (fabsf(lum) > 1e-14f) atomicAdd(&d_out_r[pix], lum * wt);
                    } else {
                        if (fabsf(cr) > 1e-14f) atomicAdd(&d_out_r[pix], cr * wt);
                        if (fabsf(cg) > 1e-14f) atomicAdd(&d_out_g[pix], cg * wt);
                        if (fabsf(cb) > 1e-14f) atomicAdd(&d_out_b[pix], cb * wt);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// CPU launcher: allocate device buffers, upload inputs, run the kernel, and
// copy results back. Buffers are function-local.
// ===========================================================================

void launch_psf_render(const OpticalSystem&             lens,
                       const std::vector<PSFGpuSource>& gpu_sources,
                       int   tile_w,
                       int   tile_h,
                       int   composite_w,
                       int   composite_h,
                       float mm_per_pixel,
                       bool  monochromatic,
                       float* out_r,
                       float* out_g,
                       float* out_b,
                       const PSFConfig& config,
                       std::string*     out_error)
{
    if (gpu_sources.empty()) return;

    const int n_surfs   = lens.num_surfaces();
    const int n_sources = (int)gpu_sources.size();
    if (n_surfs <= 0) return;

    // ---- GPU availability check (mirrors ghost_render.cu wording) ----
    {
        int device_count = 0;
        cudaError_t ce = cudaGetDeviceCount(&device_count);
        if (ce == cudaErrorInsufficientDriver) {
            if (out_error)
                *out_error = "Ghostlight requires an NVIDIA GPU with an up-to-date driver. "
                             "The installed CUDA driver is too old — please update your "
                             "NVIDIA driver (525 or newer required).";
            return;
        }
        if (ce == cudaErrorNoDevice || device_count == 0) {
            if (out_error)
                *out_error = "Ghostlight requires an NVIDIA CUDA GPU — no compatible GPU "
                             "was detected on this system. Ghostlight will produce black output.";
            return;
        }
        if (ce != cudaSuccess) {
            if (out_error)
                *out_error = std::string("Ghostlight: CUDA initialisation failed (")
                             + cudaGetErrorString(ce)
                             + "). Check that your NVIDIA driver is installed correctly.";
            return;
        }
    }

    // Clear any prior sticky error so the post-launch error check is meaningful.
    {
        cudaError_t prev_err = cudaGetLastError();
        if (prev_err != cudaSuccess) {
            fprintf(stderr, "Ghostlight: clearing CUDA sticky error: %s\n",
                    cudaGetErrorString(prev_err));
        }
    }

    // ---- Build entrance-pupil grid (shared with ghost renderer) ----
    ApertureSamplerParams sampler_p;
    sampler_p.ray_grid         = config.ray_grid;
    sampler_p.pupil_jitter     = config.pupil_jitter;
    sampler_p.jitter_seed      = config.jitter_seed;
    sampler_p.blades_override  = config.aperture_blades;
    sampler_p.rotation_deg     = config.aperture_rotation;
    const std::vector<ApertureSample> grid_samples = build_aperture_samples(lens, sampler_p);

    const int n_grid = (int)grid_samples.size();
    if (n_grid == 0) return;
    const float ray_weight = 1.0f / n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - SPAWN_OFFSET;

    // ---- Build spectral table (same path as ghost) ----
    // Monochromatic mode collapses to one wavelength at the d-line.  This
    // is the genuine "single wavelength PSF" — no lateral chromatic spatial
    // separation in the output (which a multi-wavelength luminance sum
    // would produce as a row of N spots strung along the chromatic axis).
    std::vector<GPUSpectralSample> spectral_cpu;
    int n_spec;
    if (monochromatic)
    {
        // cmf_g = 1, cmf_r = cmf_b = 0 → kernel's mono branch sums
        // src_avg * cmf_g * w = src_avg * w into out_r.
        spectral_cpu = { { 587.56f, 0.0f, 1.0f, 0.0f } };
        n_spec = 1;
    }
    else
    {
        const int ns = std::max(3, config.spectral_samples);
        spectral_cpu.resize(ns);
        if (ns == 3)
        {
            spectral_cpu[0] = { 650.0f, 1.0f, 0.0f, 0.0f };
            spectral_cpu[1] = { 550.0f, 0.0f, 1.0f, 0.0f };
            spectral_cpu[2] = { 450.0f, 0.0f, 0.0f, 1.0f };
        }
        else
        {
            const SensorProfile& prof = get_sensor_profile(config.sensor_model);
            float M_out[3][3];
            resolve_output_matrix(config.output_cs, config.custom_xyz_to_output, M_out);
            spectral_cpu = build_spectral_table(ns, 400.0f, 700.0f, prof, M_out);

            // build_spectral_table normalises the spectral INTEGRAL
            // (sum of cmf_g * dlambda = 1), but the scatter kernel sums cmf
            // WITHOUT the dlambda weight.  Left as-is the PSF
            // brightness would scale with the sample count — doubling
            // spectral_samples would double every output value.  Pre-multiply
            // the table by dlambda so the kernel's plain sum becomes the
            // Riemann sum: a white source integrates to ~1 regardless of ns,
            // matching the 3-sample RGB shortcut above.
            const float d_lambda = (700.0f - 400.0f) / (float)ns;
            for (auto& s : spectral_cpu) {
                s.cmf_r *= d_lambda;
                s.cmf_g *= d_lambda;
                s.cmf_b *= d_lambda;
            }
        }
        n_spec = ns;
    }

    // ---- Device allocations (function-local; freed on early return / end) ----
    struct DevBufs {
        Surface*           d_surfs   = nullptr;
        PSFGpuSource*      d_sources = nullptr;
        ApertureSample*    d_grid    = nullptr;
        GPUSpectralSample* d_spec    = nullptr;
        float*             d_out_r   = nullptr;
        float*             d_out_g   = nullptr;
        float*             d_out_b   = nullptr;
        ~DevBufs() {
            cudaFree(d_surfs);   cudaFree(d_sources);
            cudaFree(d_grid);    cudaFree(d_spec);
            cudaFree(d_out_r);   cudaFree(d_out_g); cudaFree(d_out_b);
        }
    } dev;

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "Ghostlight CUDA error at %s -- %s\n", site, cudaGetErrorString(e));
        if (out_error && out_error->empty())
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "CUDA error at %s -- %s", site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

#define GPU_CHK(call)                                                    \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) { report(_e, #call); return; }            \
    } while (0)

    const size_t n_px = (size_t)composite_w * composite_h;

    GPU_CHK(cudaMalloc(&dev.d_surfs,   n_surfs   * sizeof(Surface)));
    GPU_CHK(cudaMalloc(&dev.d_sources, n_sources * sizeof(PSFGpuSource)));
    GPU_CHK(cudaMalloc(&dev.d_grid,    n_grid    * sizeof(ApertureSample)));
    GPU_CHK(cudaMalloc(&dev.d_spec,    n_spec    * sizeof(GPUSpectralSample)));
    GPU_CHK(cudaMalloc(&dev.d_out_r,   n_px      * sizeof(float)));
    GPU_CHK(cudaMalloc(&dev.d_out_g,   n_px      * sizeof(float)));
    GPU_CHK(cudaMalloc(&dev.d_out_b,   n_px      * sizeof(float)));

    // Table-backed coatings: local device arena (RAII-released) + patched
    // surface copy whose coating pointers reference device memory.
    CoatingDeviceArena   coating_arena;
    std::vector<Surface> coated_surfs;
    if (!upload_coating_tables(lens, coating_arena, coated_surfs, out_error))
        return;
    const Surface* surfs_src = coated_surfs.empty() ? lens.surfaces.data()
                                                    : coated_surfs.data();

    GPU_CHK(cudaMemcpy(dev.d_surfs,   surfs_src,
                       n_surfs * sizeof(Surface), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(dev.d_sources, gpu_sources.data(),
                       n_sources * sizeof(PSFGpuSource), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(dev.d_grid,    grid_samples.data(),
                       n_grid * sizeof(ApertureSample), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(dev.d_spec,    spectral_cpu.data(),
                       n_spec * sizeof(GPUSpectralSample), cudaMemcpyHostToDevice));

    GPU_CHK(cudaMemset(dev.d_out_r, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(dev.d_out_g, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(dev.d_out_b, 0, n_px * sizeof(float)));

    // Upload image-aperture textures (no-op when the lens has no image
    // apertures; the kernel sees nullptr and skips the bitmap test).
    ApertureTexturePack apertures;
    if (!upload_aperture_textures(lens, apertures, out_error))
        return;

    // ---- Launch ----
    const int  grid_blocks = (n_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dim3 block(BLOCK_SIZE, 1, 1);
    const dim3 grid_dim((unsigned)n_sources, (unsigned)grid_blocks, 1);

    if (config.verbose) {
        printf("Ghostlight PSF: %d sources x %d aperture samples x %d spectral "
               "-> grid (%u, %u) block (%d)\n",
               n_sources, n_grid, n_spec, grid_dim.x, grid_dim.y, BLOCK_SIZE);
        fflush(stdout);
    }

    // Splat footprint.  config.splat_sigma_um is a PHYSICAL size (µm on the
    // sensor); convert to pixels via the tile's mm-per-pixel so the rendered
    // spot is resolution-independent (a higher-res tile just samples the same
    // physical Gaussian more finely).  sigma <= 0 falls through to the bilinear
    // single-pixel splat path.
    const float splat_sigma = (mm_per_pixel > 0.0f)
        ? config.splat_sigma_um / (mm_per_pixel * 1000.0f)   // µm -> px
        : 0.0f;
    int splat_radius = 0;
    if (splat_sigma > 0.0f) {
        splat_radius = std::max(1, (int)std::ceil(2.5f * splat_sigma));
        // Sanity cap — a physical splat resolved at high tile resolution can
        // legitimately span many pixels, so the cap is generous; it only
        // clamps pathological zoom-in-plus-huge-splat combinations.
        splat_radius = std::min(splat_radius, 16);
    }

    psf_kernel<<<grid_dim, block>>>(
        dev.d_surfs, n_surfs,
        dev.d_sources, n_sources,
        dev.d_grid, n_grid,
        front_R, start_z,
        mm_per_pixel,
        tile_w, tile_h,
        composite_w, composite_h,
        dev.d_out_r, dev.d_out_g, dev.d_out_b,
        ray_weight,
        monochromatic ? 1 : 0,
        dev.d_spec, n_spec,
        splat_sigma, splat_radius,
        static_cast<cudaTextureObject_t*>(apertures.d_textures));

    GPU_CHK(cudaDeviceSynchronize());

    GPU_CHK(cudaMemcpy(out_r, dev.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_g, dev.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_b, dev.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost));

#undef GPU_CHK
}
