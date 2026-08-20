// ============================================================================
// veil_render.cu — Physical veiling-glare pass.
//
// Pipeline, all on the GPU except a tiny reduction read-back:
//   1. bake_veil_reference — rasterise the analytic glare-spread function
//        g(r) = (a^2 / (r^2 + a^2))^p into a mono grid (a = core radius in
//        texels, p = veil_falloff). No FFT — the kernel is closed-form.
//   2. energy-normalise the mono reference to unit sum (resolution-independent
//        brightness), exactly as the starburst normalises its PSF.
//   3. replicate — write the normalised reference into a neutral RGB sprite.
//        The halo is ACHROMATIC (surface-reflection veiling glare is broadband,
//        unlike aperture diffraction's rainbow); it is tinted by the SOURCE's
//        own output-space colour at splat time, so a white source yields a
//        neutral veil and a red source a red veil.
//   4. downsample — box-sum toward the output pixel pitch.
//   5. splat — area-weighted-resample the sprite onto the output grid at each
//        source's sensor position, scaled by the source colour and veil_gain.
//
// The GSF's core radius on the sensor is veil_spread * sensor-half; the
// downsample and energy-conserving area-resample match render_starburst.
//
// This writes veil_r/g/b as a standalone additive layer.
// ============================================================================

#include "veil_render.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int BLOCK = 256;
constexpr int SPRITE_OVERSAMPLE = 3;      // sprite texels kept per output pixel post-downsample

// Sprite grid + GSF core geometry. The core is 1/CORE_DIV of the grid side, so
// the sprite spans GRID/(2*core) = CORE_DIV/2 core radii (16 here) — enough that
// even a heavy-tailed Lorentzian is baked to a negligible edge. Analytic, so a
// modest grid resolves it.
constexpr int VEIL_GRID     = 1024;
constexpr int VEIL_CORE_DIV = 32;         // core radius (texels) = VEIL_GRID / VEIL_CORE_DIV

struct GPUVeilSource { float px, py, r, g, b; };  // buffer-pixel position + colour

// ---------------------------------------------------------------------------
// Persistent device scratch (leaked singleton + mutex, mirroring the starburst
// cache: never destroyed, so no cudaFree races the static CUDA teardown).
// ---------------------------------------------------------------------------
struct VeilGpu {
    float* d_ref       = nullptr;  size_t ref_floats    = 0;   // [N*N] mono GSF reference
    float* d_sprite    = nullptr;  size_t sprite_floats = 0;   // [3*N*N] neutral RGB sprite
    float* d_sprite_ds = nullptr;  size_t sprite_ds_floats = 0;
    float* d_partials  = nullptr;  size_t partials_floats = 0;
    void*  d_src       = nullptr;  size_t src_bytes     = 0;   // GPUVeilSource[]
    float* d_out_r     = nullptr;
    float* d_out_g     = nullptr;
    float* d_out_b     = nullptr;
    size_t out_floats  = 0;
    float  baked_p     = -1.0f; // falloff p the cached d_ref/d_sprite were baked at (<0 = none)
};

std::mutex g_veil_mutex;
VeilGpu&   veil_gpu() { static VeilGpu* c = new VeilGpu(); return *c; }

bool ensure_bytes(void*& ptr, size_t& cap, size_t need) {
    if (need <= cap) return true;
    cudaFree(ptr); ptr = nullptr; cap = 0;
    if (cudaMalloc(&ptr, need) != cudaSuccess) return false;
    cap = need; return true;
}
bool ensure_floats(float*& ptr, size_t& cap, size_t need) {
    if (need <= cap) return true;
    cudaFree(ptr); ptr = nullptr; cap = 0;
    if (cudaMalloc(&ptr, need * sizeof(float)) != cudaSuccess) return false;
    cap = need; return true;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Rasterise the analytic glare-spread function into the mono reference grid,
// DC-centred: g(r) = (a^2 / (r^2 + a^2))^p, r in texels from the grid centre,
// a = core radius in texels, p = falloff. g(0) = 1; the heavy Lorentzian tail
// (p ~ 1.5 -> ~1/r^3) carries the wide veil wing.
__global__ void bake_veil_reference_kernel(float* d_ref, int N, float a_texels, float p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N, j = idx / N;
    float dx = (float)(i - N / 2);
    float dy = (float)(j - N / 2);
    float a2 = a_texels * a_texels;
    float base = a2 / (dx * dx + dy * dy + a2);   // in (0, 1]
    d_ref[idx] = powf(base, p);
}

// Standard shared-memory sum reduction (two loads per thread) -> one partial per
// block. The host sums the partials.
__global__ void reduce_sum_kernel(const float* in, int n, float* partials)
{
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0.0f;
    if (i < n)              sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void scale_kernel(float* d, int n, float k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] *= k;
}

// Replicate the mono reference into a neutral RGB sprite (halo is achromatic;
// the source colour tints it at splat time).
__global__ void replicate_rgb_kernel(const float* d_ref, float* d_sprite, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = d_ref[idx];
    d_sprite[3 * idx + 0] = v;
    d_sprite[3 * idx + 1] = v;
    d_sprite[3 * idx + 2] = v;
}

// Box-sum the fine N x N sprite into an M x M sprite (M = N / ds). Summing
// (not averaging) makes the subsequent per-texel deposit an exact integral of
// the fine pattern over each coarse texel.
__global__ void downsample_kernel(const float* d_fine, int N, int ds,
                                  float* d_coarse, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * M) return;
    int a = idx % M, b = idx / M;
    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    for (int jj = 0; jj < ds; ++jj)
        for (int ii = 0; ii < ds; ++ii) {
            int fi = a * ds + ii, fj = b * ds + jj;
            const float* p = d_fine + 3 * ((size_t)fj * N + fi);
            sr += p[0]; sg += p[1]; sb += p[2];
        }
    d_coarse[3 * idx + 0] = sr;
    d_coarse[3 * idx + 1] = sg;
    d_coarse[3 * idx + 2] = sb;
}

// Area-weighted resample of the sprite onto the output pixel grid, one thread
// per (source, output pixel) inside the sprite's on-sensor footprint. Each
// output pixel integrates the sprite over its OWN sensor footprint — correct and
// hole/moire-free at every scale. Energy is conserved because the overlap areas
// of one texel across all the pixels it touches sum to the texel's cell area.
// Identical to the starburst's splat_resample_kernel.
__global__ void splat_resample_kernel(const float* d_sprite, int M,
                             float sx_ds_mm, float sy_ds_mm,
                             float px_mm_x, float px_mm_y,
                             int rx, int ry,
                             const GPUVeilSource* d_src, int n_src,
                             float gain,
                             float* d_out_r, float* d_out_g, float* d_out_b,
                             int width, int height)
{
    const long long span_x = 2LL * rx + 1;
    const long long span_y = 2LL * ry + 1;
    const long long per_src = span_x * span_y;
    const long long total   = (long long)n_src * per_src;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int src_i = (int)(tid / per_src);
    long long t = tid % per_src;
    int oy = (int)(t / span_x) - ry;
    int ox = (int)(t % span_x) - rx;

    const GPUVeilSource s = d_src[src_i];
    int ix = (int)floorf(s.px) + ox;
    int iy = (int)floorf(s.py) + oy;
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    float ax0 = ((ix       - s.px) * px_mm_x) / sx_ds_mm + 0.5f * M - 0.5f;
    float ax1 = ((ix + 1.0f - s.px) * px_mm_x) / sx_ds_mm + 0.5f * M - 0.5f;
    float by0 = ((iy       - s.py) * px_mm_y) / sy_ds_mm + 0.5f * M - 0.5f;
    float by1 = ((iy + 1.0f - s.py) * px_mm_y) / sy_ds_mm + 0.5f * M - 0.5f;
    float xlo = ax0 + 0.5f, xhi = ax1 + 0.5f;
    float ylo = by0 + 0.5f, yhi = by1 + 0.5f;
    int i0 = (int)floorf(xlo), i1 = (int)floorf(xhi - 1e-6f);
    int j0 = (int)floorf(ylo), j1 = (int)floorf(yhi - 1e-6f);
    if (i0 < 0) i0 = 0;  if (i1 > M - 1) i1 = M - 1;
    if (j0 < 0) j0 = 0;  if (j1 > M - 1) j1 = M - 1;
    if (i1 < i0 || j1 < j0) return;
    if (i1 - i0 > 64) i1 = i0 + 64;
    if (j1 - j0 > 64) j1 = j0 + 64;

    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
    for (int j = j0; j <= j1; ++j) {
        float yov = fminf(yhi, (float)(j + 1)) - fmaxf(ylo, (float)j);
        if (yov <= 0.0f) continue;
        const float* row = d_sprite + 3 * ((size_t)j * M);
        for (int i = i0; i <= i1; ++i) {
            float xov = fminf(xhi, (float)(i + 1)) - fmaxf(xlo, (float)i);
            if (xov <= 0.0f) continue;
            float wgt = xov * yov;
            const float* p = row + 3 * i;
            acc_r += p[0] * wgt; acc_g += p[1] * wgt; acc_b += p[2] * wgt;
        }
    }

    float vr = acc_r * s.r * gain;
    float vg = acc_g * s.g * gain;
    float vb = acc_b * s.b * gain;
    int pix = iy * width + ix;
    if (fabsf(vr) > 1e-20f) atomicAdd(&d_out_r[pix], vr);
    if (fabsf(vg) > 1e-20f) atomicAdd(&d_out_g[pix], vg);
    if (fabsf(vb) > 1e-20f) atomicAdd(&d_out_b[pix], vb);
}

// Largest power of two <= x, clamped to [1, hi].
int po2_floor(double x, int hi) {
    int p = 1;
    while (p * 2 <= x && p * 2 <= hi) p *= 2;
    return p < 1 ? 1 : p;
}

} // namespace

// ---------------------------------------------------------------------------
bool render_veil(const OpticalSystem&            lens,
                 const LensCalibration&          calib,
                 const std::vector<FlareSource>& sources,
                 const std::vector<float>&       src_px,
                 const std::vector<float>&       src_py,
                 int                             width,
                 int                             height,
                 int                             fmt_w,
                 int                             fmt_h,
                 float                           sensor_half_w,
                 float                           sensor_half_h,
                 const FlareConfig&              cfg,
                 float*                          out_r,
                 float*                          out_g,
                 float*                          out_b,
                 std::string*                    out_error,
                 VeilDebug*                      dbg)
{
    (void)lens;                // analytic GSF veil needs no lens surfaces
    (void)calib;               // sensor geometry arrives via sensor_half_* args
    const DiffractionConfig& dc = cfg.diffraction;
    if (!dc.veil || sources.empty()) return true;

    auto fail = [&](const char* msg) {
        if (out_error && out_error->empty()) *out_error = msg;
        fprintf(stderr, "render_veil: %s\n", msg);
        return false;
    };

    if (width <= 0 || height <= 0 || fmt_w <= 0 || fmt_h <= 0)
        return fail("degenerate output dimensions");

    // ---- GSF physical scale (host; also recorded for the debug capture) ------
    // Core radius on the sensor = veil_spread * sensor-half (height reference,
    // falling back to width). The halo is round in sensor mm (isotropic pitch).
    float half_ref = (sensor_half_h > 0.0f) ? sensor_half_h : sensor_half_w;
    float spread = dc.veil_spread;
    if (!(spread > 0.0f)) spread = 0.12f;
    spread = (spread < 1e-3f) ? 1e-3f : (spread > 4.0f ? 4.0f : spread);
    const float core_mm = spread * half_ref;

    float p = dc.veil_falloff;
    if (!(p >= 1.0f)) p = 1.0f;
    if (p > 3.0f) p = 3.0f;

    const int    N        = VEIL_GRID;
    const float  a_texels = (float)N / (float)VEIL_CORE_DIV;
    const float  dx_fine  = core_mm / a_texels;   // sensor mm per fine texel (isotropic)
    const size_t NN       = (size_t)N * N;
    const float  sx = dx_fine, sy = dx_fine;

    std::lock_guard<std::mutex> lock(g_veil_mutex);
    VeilGpu& g = veil_gpu();

    // ---- Allocate + zero the GSF veil output buffer ----
    const size_t n_px = (size_t)width * height;
    if (n_px > g.out_floats) {
        cudaFree(g.d_out_r); cudaFree(g.d_out_g); cudaFree(g.d_out_b);
        g.d_out_r = g.d_out_g = g.d_out_b = nullptr;
        g.out_floats = 0;
        if (cudaMalloc(&g.d_out_r, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_g, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_b, n_px * sizeof(float)) != cudaSuccess)
            return fail("cudaMalloc failed (veil output)");
        g.out_floats = n_px;
    }
    cudaMemset(g.d_out_r, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_g, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_b, 0, n_px * sizeof(float));

    // ---- GSF halo ------------------------------------------------------------
    {
        if (half_ref <= 0.0f) return fail("degenerate sensor half-size");
        if (!(dx_fine > 0.0f)) return fail("degenerate GSF pitch");

        if (!ensure_floats(g.d_ref,       g.ref_floats,       NN) ||
            !ensure_floats(g.d_sprite,    g.sprite_floats,    NN * 3) ||
            !ensure_floats(g.d_sprite_ds, g.sprite_ds_floats, NN * 3))
            return fail("cudaMalloc failed (ref/sprite)");

        const int blocks = (int)((NN + BLOCK - 1) / BLOCK);

        // 1-3: bake the analytic GSF reference, energy-normalise to unit sum, and
        // replicate into the neutral RGB sprite. All three depend ONLY on the
        // clamped falloff p (N and a_texels are compile-time constants), so cache
        // the result on p: repeated renders at a fixed falloff — progressive
        // source-flare chunks, and unrelated parameter changes — reuse the baked
        // sprite and avoid a blocking energy read-back.
        if (g.baked_p != p) {
            const int reduce_blocks = (int)((NN + BLOCK * 2 - 1) / (BLOCK * 2));
            if (!ensure_floats(g.d_partials, g.partials_floats, reduce_blocks))
                return fail("cudaMalloc failed (partials)");
            bake_veil_reference_kernel<<<blocks, BLOCK>>>(g.d_ref, N, a_texels, p);
            reduce_sum_kernel<<<reduce_blocks, BLOCK>>>(g.d_ref, (int)NN, g.d_partials);
            std::vector<float> partials(reduce_blocks);
            if (cudaMemcpy(partials.data(), g.d_partials, reduce_blocks * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("cudaMemcpy failed (GSF energy reduction)");
            double sum = 0.0;
            for (float pv : partials) sum += pv;
            if (sum <= 0.0) return fail("degenerate GSF (zero energy)");
            scale_kernel<<<blocks, BLOCK>>>(g.d_ref, (int)NN, (float)(1.0 / sum));
            replicate_rgb_kernel<<<blocks, BLOCK>>>(g.d_ref, g.d_sprite, (int)NN);
            g.baked_p = p;
        }

        // 4: downsample toward the output pixel pitch.
        const float px_mm_x = 2.0f * sensor_half_w / (float)fmt_w;
        const float px_mm_y = 2.0f * sensor_half_h / (float)fmt_h;
        double ds_target = std::min((double)px_mm_x / (sx * SPRITE_OVERSAMPLE),
                                    (double)px_mm_y / (sy * SPRITE_OVERSAMPLE));
        int ds = po2_floor(ds_target, N / 8);
        const int   M = N / ds;
        const float sx_ds = sx * ds;
        const float sy_ds = sy * ds;

        const float* splat_src;
        if (ds > 1) {
            int mblocks = (int)(((size_t)M * M + BLOCK - 1) / BLOCK);
            downsample_kernel<<<mblocks, BLOCK>>>(g.d_sprite, N, ds, g.d_sprite_ds, M);
            splat_src = g.d_sprite_ds;
        } else {
            splat_src = g.d_sprite;
        }

        // 5: splat per source into the shared output buffer.
        const int n_src = (int)sources.size();
        std::vector<GPUVeilSource> host_src(n_src);
        for (int i = 0; i < n_src; ++i)
            host_src[i] = { src_px[i], src_py[i], sources[i].r, sources[i].g, sources[i].b };
        if (!ensure_bytes(g.d_src, g.src_bytes, n_src * sizeof(GPUVeilSource)))
            return fail("cudaMalloc d_src");
        cudaMemcpy(g.d_src, host_src.data(), n_src * sizeof(GPUVeilSource),
                   cudaMemcpyHostToDevice);

        // The energy-normalised sprite integrates to unity, so veil_gain is the
        // fraction of the source's own flux the halo carries.
        const float gain = dc.veil_gain;
        int rx = (int)std::ceil(0.5 * M * sx_ds / px_mm_x) + 1;
        int ry = (int)std::ceil(0.5 * M * sy_ds / px_mm_y) + 1;
        if (rx > width)  rx = width;
        if (ry > height) ry = height;
        long long total = (long long)n_src * (2LL * rx + 1) * (2LL * ry + 1);
        int splat_blocks = (int)((total + BLOCK - 1) / BLOCK);
        splat_resample_kernel<<<splat_blocks, BLOCK>>>(splat_src, M, sx_ds, sy_ds,
            px_mm_x, px_mm_y, rx, ry, (const GPUVeilSource*)g.d_src, n_src, gain,
            g.d_out_r, g.d_out_g, g.d_out_b, width, height);
    }

    // ---- Sync + copy the veil layer out (caller-zeroed buffers) --------------
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail(cudaGetErrorString(err));

    cudaMemcpy(out_r, g.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_g, g.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b, g.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Optional GSF validation capture ----
    if (dbg) {
        dbg->grid = N;
        dbg->reference.resize(NN);
        dbg->sprite_rgb.resize(NN * 3);
        cudaMemcpy(dbg->reference.data(), g.d_ref, NN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg->sprite_rgb.data(), g.d_sprite, NN * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        dbg->dx_mm_x = sx; dbg->dx_mm_y = sy;
        dbg->core_mm = core_mm;
        dbg->falloff = p;
        dbg->core_texels = a_texels;
    }
    return true;
}
