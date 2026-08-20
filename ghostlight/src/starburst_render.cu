// ============================================================================
// starburst_render.cu — Aperture-diffraction starburst pass.
//
// Pipeline, all on the GPU except a tiny reduction read-back:
//   1. build_pupil   — rasterise the pupil amplitude A(u,v) into a complex grid
//                      (4x4-supersampled PupilMask silhouette; aspect handled by
//                      the per-axis physical pitch, not by stretching the mask).
//   2. fft2d         — forward FFT via fft_backend.h (cuFFT, isolated there).
//   3. psf_from_fft  — |FFT|^2 with fftshift, DC to the grid centre.
//   4. energy-normalise the mono reference PSF to unit sum (resolution-
//      independent brightness).
//   5. bake_sprite   — integrate over wavelength: PSF_lambda(x) is the reference
//      PSF rescaled by lambda_ref/lambda (Fourier scaling theorem, exact for an
//      achromatic amplitude pupil), weighted by the output-space CMF table.
//   6. downsample    — box-sum the fine sprite toward the output pixel pitch so
//      the splat integrates energy correctly without atomic-contention blow-up.
//   7. splat         — add the sprite, scaled by each source's colour, at each
//      source's sensor position.
//
// The physical scale is set by the calibrated first-order optics: one sprite
// texel is lambda_ref * f_number * pupil_fill millimetres on the sensor, per
// axis.  No arbitrary size knob (scale_trim is a purely artistic multiplier).
// ============================================================================

#include "diffraction.h"
#include "baffle.h"             // GpuBaffleStack, build_gpu_baffles
#include "aperture_sampler.h"   // PupilMask, resolve_pupil_mask
#include "aperture_textures.h"  // ApertureTexturePack (front-glass texture upload)
#include "fft_backend.h"
#include "mdft_render.h"        // mdft_starburst (the exact resample-free engine)
#include "spawn_plane.h"        // SPAWN_OFFSET
#include "spectral.h"
#include "trace_core.h"         // Ray, Vec3f, OpticalSystem (device-safe)
#include "trace_event.h"        // TraceResult, TraceStatus (device-safe)
#include "lens_calibration.h"   // LensCalibration

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The plain CPU forward trace (no ghost bounces) builds the cat's-eye
// survivor envelope on the host. Forward-declared rather than pulled in via
// trace.h, whose diagnostic RayPath overloads are #ifndef __CUDACC__ and so do
// not exist in this .cu translation unit; resolved at link time from trace.cpp.
TraceResult trace_primary_ray(const Ray& ray, const OpticalSystem& lens);

namespace {

constexpr int   BLOCK = 256;
constexpr float LAMBDA_REF_NM = 550.0f;   // reference wavelength for the mono FFT
constexpr int   SPRITE_OVERSAMPLE = 3;    // sprite texels kept per output pixel post-downsample
// Boundary supersampling turns binary cat's-eye rim changes into smooth coverage
// changes as field angle moves. Only boundary cells incur the extra traces.
constexpr int   SURVIVOR_EDGE_SS   = 10;  // envelope rim: smooths the pupil edge vs field
constexpr int   SURVIVOR_CENTER_SS = 8;   // centre solve: smooths the crescent centroid vs field

struct GPUStarSource { float px, py, r, g, b; };  // buffer-pixel position + colour

// ---------------------------------------------------------------------------
// Persistent device scratch (leaked singleton + mutex, mirroring the ghost
// cache: never destroyed, so no cudaFree races the static CUDA teardown).
// ---------------------------------------------------------------------------
struct StarburstGpu {
    void*  d_pupil     = nullptr;  size_t pupil_bytes  = 0;   // float2 [N*N]
    float* d_psf       = nullptr;  size_t psf_floats   = 0;   // [N*N]
    float* d_sprite    = nullptr;  size_t sprite_floats = 0;  // [3*N*N]
    float* d_sprite_ds = nullptr;  size_t sprite_ds_floats = 0;
    float* d_partials  = nullptr;  size_t partials_floats = 0;
    float* d_env       = nullptr;  size_t env_floats   = 0;   // cat's-eye envelope [ME*ME]
    void*  d_spec      = nullptr;  size_t spec_bytes   = 0;   // GPUSpectralSample[]
    void*  d_src       = nullptr;  size_t src_bytes    = 0;   // GPUStarSource[]
    float* d_out_r     = nullptr;
    float* d_out_g     = nullptr;
    float* d_out_b     = nullptr;
    size_t out_floats  = 0;
};

std::mutex     g_star_mutex;
StarburstGpu&  star_gpu() { static StarburstGpu* c = new StarburstGpu(); return *c; }

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

// Extra amplitude modifiers folded into the analytic stop silhouette.
struct PupilExtras {
    // Cat's-eye rim-vignetting envelope, smooth [0,1] over pupil coords [-1,1]^2.
    // nullptr => fully open (no off-axis vignetting).
    const float* env = nullptr;
    int          env_N = 0;

    // Pupil physical scale: normalised coord |u|=1 maps to ep_x mm at the spawn
    // plane. Used by the matte box and texture to reach mm / surface coordinates.
    float ep_x = 1.0f, ep_y = 1.0f;

    // Front-of-lens baffle stack (matte box, hoods, mount), applied as a HARD
    // analytic clip at full FFT resolution so each knife edge produces real
    // diffraction fringes. The pupil ray at (u*ep_x, v*ep_y) on the spawn plane
    // is projected along the source direction to each baffle plane and clipped
    // there. Empty (n==0) is a no-op. (The matte box is merged in as one RECT
    // baffle by build_gpu_baffles, so it is handled by this same clip.)
    GpuBaffleStack baffles;
    float tan_ax = 0.0f, tan_ay = 0.0f;

    // Front-glass APERTURE_IMAGE as a graded amplitude transmission in [0,1].
    // Sampled with the SAME UV convention the ray tracer uses (trace_cuda.h):
    // hit on the surface = (u*ep + tan_a*tex_d0), tex = 0.5 + hit/(2*semi).
    cudaTextureObject_t tex = 0;
    int   has_tex = 0;
    float tex_aspect = 1.0f, tex_semi = 1.0f, tex_d0 = 20.0f;
};

// Bilinear sample of the envelope at pupil coord (u, v) in [-1, 1]^2.
__device__ inline float sample_env(const float* env, int ME, float u, float v)
{
    float fx = (u * 0.5f + 0.5f) * ME - 0.5f;
    float fy = (v * 0.5f + 0.5f) * ME - 0.5f;
    fx = fminf(fmaxf(fx, 0.0f), (float)(ME - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(ME - 1));
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, ME - 1), y1 = min(y0 + 1, ME - 1);
    float tx = fx - x0, ty = fy - y0;
    float a = env[(size_t)y0 * ME + x0] * (1.0f - tx) + env[(size_t)y0 * ME + x1] * tx;
    float b = env[(size_t)y1 * ME + x0] * (1.0f - tx) + env[(size_t)y1 * ME + x1] * tx;
    return a * (1.0f - ty) + b * ty;
}

// Rasterise the effective pupil amplitude A(u,v) = stop x cat's-eye x matte box
// x front texture. Grid coordinate g in [-1,1) maps to entrance-pupil-normalised
// coordinate g / pupil_fill, so the unit pupil disk occupies a pupil_fill
// fraction of the half-grid. 4x4 supersampling antialiases the stop and baffle
// edges; the smooth envelope and texture are sampled directly.
__global__ void build_pupil_kernel(float2* d_pupil, int N, float pupil_fill,
                                    PupilMask mask, PupilExtras ex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N, j = idx / N;

    float re = 0.0f;
    #pragma unroll
    for (int sy = 0; sy < 4; ++sy)
        for (int sx = 0; sx < 4; ++sx) {
            float u = ((i + (sx + 0.5f) * 0.25f) / N * 2.0f - 1.0f) / pupil_fill;
            float v = ((j + (sy + 0.5f) * 0.25f) / N * 2.0f - 1.0f) / pupil_fill;
            if (!mask.contains(u, v)) continue;            // outside stop silhouette

            float a = 1.0f;

            if (ex.env != nullptr)                          // cat's-eye vignetting
                a *= sample_env(ex.env, ex.env_N, u, v);

            if (ex.baffles.n > 0) {                         // hard barn-door / baffle clip
                if (baffle_stack_blocks(ex.baffles, u * ex.ep_x, v * ex.ep_y,
                                        ex.tan_ax, ex.tan_ay)) continue;
            }

            if (ex.has_tex) {                               // front-glass transmission
                float hx = (u * ex.ep_x + ex.tan_ax * ex.tex_d0) / ex.tex_aspect;
                float hy =  v * ex.ep_y + ex.tan_ay * ex.tex_d0;
                float tu = 0.5f + hx / (2.0f * ex.tex_semi);
                float tv = 0.5f + hy / (2.0f * ex.tex_semi);
                a *= tex2D<float>(ex.tex, tu, tv);
            }

            re += a;
        }
    d_pupil[idx] = make_float2(re * (1.0f / 16.0f), 0.0f);
}

// |A(u,v)|^2 per pupil texel, for the throughput reduction (imag is always 0).
__global__ void cmagsq_kernel(const float2* p, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float2 z = p[idx]; out[idx] = z.x * z.x + z.y * z.y; }
}

// |FFT|^2 with fftshift: reads the DC-at-origin transform, writes the power
// spectrum with DC at the grid centre.
__global__ void psf_from_fft_kernel(const float2* d_fft, float* d_psf, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N, j = idx / N;
    int si = (i + N / 2) % N;
    int sj = (j + N / 2) % N;
    float2 z = d_fft[sj * N + si];
    d_psf[idx] = z.x * z.x + z.y * z.y;
}

// Standard shared-memory sum reduction (two loads per thread) → one partial per
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

__device__ inline float bilinear_psf(const float* psf, float fx, float fy, int N)
{
    if (fx < 0.0f || fy < 0.0f || fx >= (float)(N - 1) || fy >= (float)(N - 1))
        return 0.0f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - x0, ty = fy - y0;
    const float* row0 = psf + (size_t)y0 * N;
    const float* row1 = psf + (size_t)(y0 + 1) * N;
    float a = row0[x0] * (1.0f - tx) + row0[x0 + 1] * tx;
    float b = row1[x0] * (1.0f - tx) + row1[x0 + 1] * tx;
    return a * (1.0f - ty) + b * ty;
}

// Integrate the mono PSF over wavelength into an RGB sprite. Each wavelength's
// pattern is the reference PSF rescaled by lambda_ref/lambda (scaling theorem),
// weighted by the output-space CMF (pre-multiplied by the sample spacing on the
// host so brightness is independent of spectral_samples).
__global__ void bake_sprite_kernel(const float* d_psf, int N,
                                    const GPUSpectralSample* d_spec, int n_spec,
                                    float lambda_ref, float* d_sprite)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N, j = idx / N;
    float dxr = (float)(i - N / 2);
    float dyr = (float)(j - N / 2);
    float half = (float)(N / 2);

    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    for (int si = 0; si < n_spec; ++si) {
        const GPUSpectralSample s = d_spec[si];
        float k = lambda_ref / s.lambda;                 // rescale factor
        float val = bilinear_psf(d_psf, half + dxr * k, half + dyr * k, N) * k * k;
        cr += s.cmf_r * val;
        cg += s.cmf_g * val;
        cb += s.cmf_b * val;
    }
    d_sprite[3 * idx + 0] = cr;
    d_sprite[3 * idx + 1] = cg;
    d_sprite[3 * idx + 2] = cb;
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

// Resample the sprite onto the output pixel grid, one thread per (source,
// output pixel) inside the sprite's on-sensor footprint. Each output pixel
// integrates the sprite over its OWN sensor footprint — an exact area-weighted
// box resample: the pixel's extent is projected into sprite-texel space and every
// texel it overlaps contributes its value times the overlap area. Because it is a
// true area integral it is correct and hole-free in every regime:
//   * sprite finer than the pixel (physically scaled / oversampled): it averages
//     the several texels the pixel covers — no scatter moiré (the "faint grid"
//     that appears when 1–3 texels land per pixel).
//   * sprite coarser than the pixel (magnified by scale_trim / a slow lens / a
//     high-res output): the box shrinks below one texel and it interpolates —
//     no LATTICE OF HOLES (the "square Airy full of gaps") that appears
//     once a texel spans more than one pixel.
// Energy is conserved because the overlap areas of one texel across all the pixels
// it touches sum to the texel's own cell area — no separate area-ratio factor.
__global__ void splat_resample_kernel(const float* d_sprite, int M,
                             float sx_ds_mm, float sy_ds_mm,
                             float px_mm_x, float px_mm_y,
                             int rx, int ry,
                             const GPUStarSource* d_src, int n_src,
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

    const GPUStarSource s = d_src[src_i];
    int ix = (int)floorf(s.px) + ox;
    int iy = (int)floorf(s.py) + oy;
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    // The output pixel spans [ix, ix+1) x [iy, iy+1) in pixels; project its
    // extent into sprite-texel coordinates. Texel a sits at sensor offset
    // (a - M/2 + 0.5) * sx_ds from the sprite centre, so a = off/sx_ds + M/2 - 0.5.
    float ax0 = ((ix       - s.px) * px_mm_x) / sx_ds_mm + 0.5f * M - 0.5f;
    float ax1 = ((ix + 1.0f - s.px) * px_mm_x) / sx_ds_mm + 0.5f * M - 0.5f;
    float by0 = ((iy       - s.py) * px_mm_y) / sy_ds_mm + 0.5f * M - 0.5f;
    float by1 = ((iy + 1.0f - s.py) * px_mm_y) / sy_ds_mm + 0.5f * M - 0.5f;
    // Texel-space box edges (sx_ds/sy_ds are positive, so already ordered), padded
    // by half a texel because texel a covers [a-0.5, a+0.5).
    float xlo = ax0 + 0.5f, xhi = ax1 + 0.5f;   // cell of texel a is [a, a+1) here
    float ylo = by0 + 0.5f, yhi = by1 + 0.5f;
    int i0 = (int)floorf(xlo), i1 = (int)floorf(xhi - 1e-6f);
    int j0 = (int)floorf(ylo), j1 = (int)floorf(yhi - 1e-6f);
    if (i0 < 0) i0 = 0;  if (i1 > M - 1) i1 = M - 1;
    if (j0 < 0) j0 = 0;  if (j1 > M - 1) j1 = M - 1;
    if (i1 < i0 || j1 < j0) return;
    // Defensive: the downsample keeps the box to a few texels for realistic
    // sizes; cap it so a pathologically minified sprite (tiny output) can't spin.
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

// Build the cat's-eye rim-vignetting envelope: for each pupil coordinate inside
// the real stop, 1 if a primary ray launched from that pupil height at the source
// field angle survives to the sensor, else 0. The envelope carries ONLY the rim
// vignetting (the crisp analytic stop, applied at full FFT resolution, owns the
// stop edge and spikes). Outside-stop cells are filled by dilation from the
// interior so bilinear upsampling does not introduce a boundary ring. (cx, cy)
// centers sampling on the field's surviving pupil crescent.
void build_survivor_envelope(const OpticalSystem& lens, const PupilMask& real_stop,
                             float ep_x, float ep_y, float cx, float cy,
                             float ax, float ay,
                             float lambda_nm, int ME, std::vector<float>& env)
{
    if (lens.surfaces.empty()) {
        env.assign((size_t)ME * ME, 1.0f);
        return;
    }

    env.assign((size_t)ME * ME, -1.0f);        // -1 = outside stop (fill by dilation below)
    const float z_spawn = lens.surfaces[0].z - SPAWN_OFFSET;
    const float bx = std::tan(ax), by = std::tan(ay);

    for (int j = 0; j < ME; ++j)
        for (int i = 0; i < ME; ++i) {
            const float u = (i + 0.5f) / ME * 2.0f - 1.0f;
            const float v = (j + 0.5f) / ME * 2.0f - 1.0f;
            if (!real_stop.contains(u, v)) continue;    // outside stop -> stays -1

            Ray ray;
            ray.origin = Vec3f(cx + u * ep_x, cy + v * ep_y, z_spawn);
            ray.dir    = Vec3f(bx, by, 1.0f).normalized();
            ray.lambda = lambda_nm;

            TraceResult res = trace_primary_ray(ray, lens);
            bool ok = (res.status == TraceStatus::OK)
                   && std::isfinite(res.position.x) && std::isfinite(res.position.y);
            env[(size_t)j * ME + i] = ok ? 1.0f : 0.0f;
        }

    // Supersample cells that straddle the survival boundary and store fractional
    // coverage. Uniform neighborhoods remain binary.
    {
        const std::vector<float> base = env;      // detect on the pre-refine survival
        const int   K  = SURVIVOR_EDGE_SS;
        const float du = 2.0f / (float)ME;        // one envelope cell in pupil coords
        for (int j = 0; j < ME; ++j)
            for (int i = 0; i < ME; ++i) {
                const size_t idx = (size_t)j * ME + i;
                const float e = base[idx];
                if (e < 0.0f) continue;           // outside stop -> the dilation fills it
                bool boundary = false;
                for (int dj = -1; dj <= 1 && !boundary; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        if (!di && !dj) continue;
                        const int ni = i + di, nj = j + dj;
                        if (ni < 0 || ni >= ME || nj < 0 || nj >= ME) continue;
                        const float ne = base[(size_t)nj * ME + ni];
                        if (ne >= 0.0f && ((ne > 0.5f) != (e > 0.5f))) { boundary = true; break; }
                    }
                if (!boundary) continue;          // interior of a survive/clip region: exact 0/1
                const float u0 = (i + 0.5f) / ME * 2.0f - 1.0f;
                const float v0 = (j + 0.5f) / ME * 2.0f - 1.0f;
                int inside = 0, surv_ss = 0;
                for (int sy = 0; sy < K; ++sy)
                    for (int sx = 0; sx < K; ++sx) {
                        const float uu = u0 + du * ((sx + 0.5f) / K - 0.5f);
                        const float vv = v0 + du * ((sy + 0.5f) / K - 0.5f);
                        if (!real_stop.contains(uu, vv)) continue;   // outside stop -> masked anyway
                        ++inside;
                        Ray ray;
                        ray.origin = Vec3f(cx + uu * ep_x, cy + vv * ep_y, z_spawn);
                        ray.dir    = Vec3f(bx, by, 1.0f).normalized();
                        ray.lambda = lambda_nm;
                        const TraceResult res = trace_primary_ray(ray, lens);
                        if (res.status == TraceStatus::OK &&
                            std::isfinite(res.position.x) && std::isfinite(res.position.y))
                            ++surv_ss;
                    }
                if (inside > 0) env[idx] = (float)surv_ss / (float)inside;
            }
    }

    // Dilate the survival outward into the -1 (outside-stop) band so the boundary
    // matches the adjacent interior instead of jumping to 1. R covers the bilinear
    // upsample reach (1 envelope cell ≈ 1-2 FFT texels); cells with no inside
    // neighbour resolve to 0 (harmless — the analytic stop masks them anyway).
    const int R = 2;
    std::vector<float> filled = env;
    for (int j = 0; j < ME; ++j)
        for (int i = 0; i < ME; ++i) {
            if (env[(size_t)j * ME + i] >= 0.0f) continue;   // inside stop: keep survival
            float best = 0.0f;
            for (int dj = -R; dj <= R; ++dj)
                for (int di = -R; di <= R; ++di) {
                    int ni = i + di, nj = j + dj;
                    if (ni < 0 || ni >= ME || nj < 0 || nj >= ME) continue;
                    float e = env[(size_t)nj * ME + ni];
                    if (e >= 0.0f && e > best) best = e;
                }
            filled[(size_t)j * ME + i] = best;
        }
    env.swap(filled);
}

// Centre of the surviving pupil crescent for an off-axis field: the centroid of
// spawn-plane launch heights whose ray actually reaches the sensor at (ax, ay).
//
// It is the survivor-envelope centre, used instead of the geometric stop-centre
// chief ray.  Past the image circle the stop-centre ray lands in
// the vignetted interior (or a spurious secondary solution), so an envelope
// centred there sees no survivors -> T=0 -> the starburst blanks; and because
// that solve flip-flops between branches at neighbouring field angles, the blank
// switches on and off as the source is dragged (the "jumping in and out of
// black").  The surviving launches' centroid, by contrast, is a smooth function
// of field and sits ON the crescent by construction, so the ±ep window always
// captures it: T fades smoothly to the TRUE vignette edge — far outside the
// chief-ray circle — where the last launch stops surviving (any=false, a clean
// monotonic cut-off) instead of strobing partway out.
static void solve_survivor_center(const OpticalSystem& lens, float ax, float ay,
                                  float lambda_nm, float& out_cx, float& out_cy,
                                  bool& any)
{
    out_cx = 0.0f; out_cy = 0.0f; any = false;
    if (lens.surfaces.empty()) return;
    const float front_R = lens.surfaces[0].semi_aperture;
    if (front_R <= 0.0f) return;

    const float z_spawn = lens.surfaces[0].z - SPAWN_OFFSET;
    const float bx = std::tan(ax), by = std::tan(ay);
    // Launch box: a height h at the spawn plane arrives at the front element
    // offset by ~SPAWN_OFFSET*tan(field), so surviving launches (which must enter
    // the ±front_R element) span front_R + SPAWN_OFFSET*|tan| per axis. Scan a
    // hair wider. This already covers the walk, which is why the starburst pupil
    // takes no spawn_shift(): it centres on the measured survivor centroid below.
    const float span_x = 1.1f * front_R + SPAWN_OFFSET * fabsf(bx);
    const float span_y = 1.1f * front_R + SPAWN_OFFSET * fabsf(by);
    constexpr int S = 81;                         // scan resolution (cheap vs the ME envelope)

    auto survives = [&](float lx, float ly) -> bool {
        Ray ray;
        ray.origin = Vec3f(lx, ly, z_spawn);
        ray.dir    = Vec3f(bx, by, 1.0f).normalized();
        ray.lambda = lambda_nm;
        const TraceResult res = trace_primary_ray(ray, lens);
        return res.status == TraceStatus::OK &&
               std::isfinite(res.position.x) && std::isfinite(res.position.y);
    };

    // Pass 1: binary survival on the S x S launch grid.
    std::vector<uint8_t> surv((size_t)S * S, 0);
    for (int j = 0; j < S; ++j) {
        const float ly = span_y * (2.0f * j / (S - 1) - 1.0f);
        for (int i = 0; i < S; ++i) {
            const float lx = span_x * (2.0f * i / (S - 1) - 1.0f);
            surv[(size_t)j * S + i] = survives(lx, ly) ? 1 : 0;
        }
    }

    // Coverage-weighted centroid. Uniform neighborhoods contribute their cell
    // centers; boundary cells are supersampled at their launch positions so the
    // centroid varies smoothly with field angle.
    const float hx = span_x / (S - 1);            // half the launch-grid cell spacing
    const float hy = span_y / (S - 1);
    const int   K  = SURVIVOR_CENTER_SS;
    const float inv_k2 = 1.0f / (float)(K * K);
    double sxsum = 0.0, sysum = 0.0, wsum = 0.0;
    for (int j = 0; j < S; ++j) {
        const float ly = span_y * (2.0f * j / (S - 1) - 1.0f);
        for (int i = 0; i < S; ++i) {
            const float lx = span_x * (2.0f * i / (S - 1) - 1.0f);
            const uint8_t s = surv[(size_t)j * S + i];
            bool boundary = false;
            for (int dj = -1; dj <= 1 && !boundary; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    if (!di && !dj) continue;
                    const int ni = i + di, nj = j + dj;
                    if (ni < 0 || ni >= S || nj < 0 || nj >= S) continue;
                    if (surv[(size_t)nj * S + ni] != s) { boundary = true; break; }
                }
            if (!boundary) {
                if (s) { sxsum += lx; sysum += ly; wsum += 1.0; }
                continue;
            }
            for (int sy = 0; sy < K; ++sy) {
                const float lly = ly + hy * ((2.0f * (sy + 0.5f) / K) - 1.0f);
                for (int sx = 0; sx < K; ++sx) {
                    const float llx = lx + hx * ((2.0f * (sx + 0.5f) / K) - 1.0f);
                    if (survives(llx, lly)) {
                        sxsum += (double)llx * inv_k2;
                        sysum += (double)lly * inv_k2;
                        wsum  += inv_k2;
                    }
                }
            }
        }
    }
    if (wsum > 0.0) { out_cx = (float)(sxsum / wsum); out_cy = (float)(sysum / wsum); any = true; }
}

// Largest power of two <= x, clamped to [1, hi].
int po2_floor(double x, int hi) {
    int p = 1;
    while (p * 2 <= x && p * 2 <= hi) p *= 2;
    return p < 1 ? 1 : p;
}

} // namespace

// ---------------------------------------------------------------------------
bool render_starburst(const OpticalSystem&            lens,
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
                      StarburstDebug*                 dbg)
{
    const DiffractionConfig& dc = cfg.diffraction;
    if (!dc.starburst || sources.empty()) return true;

    auto fail = [&](const char* msg) {
        if (out_error && out_error->empty()) *out_error = msg;
        fprintf(stderr, "render_starburst: %s\n", msg);
        return false;
    };

    // ---- First-order optics (fallbacks keep a partly-degenerate lens usable) ----
    const float front_R = lens.surfaces.empty() ? 0.0f : lens.surfaces[0].semi_aperture;
    float efl_x = calib.focal_length_x, efl_y = calib.focal_length_y;
    float ep_x  = calib.entrance_pupil_semi_x, ep_y = calib.entrance_pupil_semi_y;
    if (efl_x <= 0.0f && calib.max_half_angle_h > 0.0f)
        efl_x = sensor_half_w / std::tan(calib.max_half_angle_h);
    if (efl_y <= 0.0f && calib.max_half_angle_v > 0.0f)
        efl_y = sensor_half_h / std::tan(calib.max_half_angle_v);
    if (ep_x <= 0.0f) ep_x = front_R;
    if (ep_y <= 0.0f) ep_y = front_R;
    if (efl_x <= 0.0f || efl_y <= 0.0f || ep_x <= 0.0f || ep_y <= 0.0f)
        return fail("degenerate first-order optics (no focal length / pupil)");

    const float fnum_x = efl_x / (2.0f * ep_x);
    const float fnum_y = efl_y / (2.0f * ep_y);

    // ---- Grid + physical pitch ----
    int N = dc.starburst_grid;
    if (N < 256 || (N & (N - 1)) != 0)   // power of two, >= 256
        return fail("starburst_grid must be a power of two >= 256");
    const float fill = (dc.pupil_fill > 0.02f && dc.pupil_fill < 0.9f) ? dc.pupil_fill : 0.30f;

    const float lambda_ref_mm = LAMBDA_REF_NM * 1e-6f;   // nm -> mm
    // Sensor pitch of one fine sprite texel (mm), per axis. dx = lambda * Nw * fill.
    const float dx_fine_x = lambda_ref_mm * fnum_x * fill;
    const float dx_fine_y = lambda_ref_mm * fnum_y * fill;

    // Grow the grid when the periodic FFT extent would leave a visible box in
    // the frame. Grown patterns use the sprite engine because MDFT exposes its
    // per-wavelength sinc nulls at this scale. starburst_grid_cap bounds VRAM.
    StarburstEngine eff_engine = dc.starburst_engine;
    {
        const float trim0   = (dc.scale_trim > 0.0f) ? dc.scale_trim : 1.0f;
        const float dx0_x   = trim0 * dx_fine_x;
        const float dx0_y   = trim0 * dx_fine_y;
        const float frame_w = 2.0f * sensor_half_w;   // == width  * px_mm (buffer==format)
        const float frame_h = 2.0f * sensor_half_h;   // == height * px_mm
        const float frac    = fmaxf(frame_w > 0.0f ? (float)N * dx0_x / frame_w : 0.0f,
                                    frame_h > 0.0f ? (float)N * dx0_y / frame_h : 0.0f);
        constexpr float EXTEND_MIN_FRAC = 0.20f;   // >= this: prominent, fill the frame

        // Resolve the adjustable grid cap: power of two, clamped to [N, HARD_MAX].
        constexpr int STARBURST_GRID_HARD_MAX = 16384;   // VRAM backstop
        int grid_cap = (dc.starburst_grid_cap >= 256) ? dc.starburst_grid_cap : 8192;
        if (grid_cap > STARBURST_GRID_HARD_MAX) grid_cap = STARBURST_GRID_HARD_MAX;
        { int p = 256; while ((p << 1) <= grid_cap) p <<= 1; grid_cap = p; }  // floor to pow2
        if (grid_cap < N) grid_cap = N;

        // Grow the grid when the frame exceeds the illuminated image circle.
        constexpr float OVERSIZE_MARGIN = 1.5f;
        const float circle_w = (calib.image_circle_semi_w > 0.0f)
                                   ? calib.image_circle_semi_w : calib.sensor_half_w;
        const float circle_h = (calib.image_circle_semi_h > 0.0f)
                                   ? calib.image_circle_semi_h : calib.sensor_half_h;
        const bool oversized =
            (circle_w > 0.0f && sensor_half_w > OVERSIZE_MARGIN * circle_w) ||
            (circle_h > 0.0f && sensor_half_h > OVERSIZE_MARGIN * circle_h);
        const bool prominent = (frac >= EXTEND_MIN_FRAC);

        if (prominent) eff_engine = StarburstEngine::SPRITE;   // large-pattern engine

        if (frac < 1.0f && (prominent || oversized) && !sources.empty() &&
            width > 0 && height > 0 && dx0_x > 0.0f && dx0_y > 0.0f && grid_cap > N) {
            // Farthest frame edge from the source cluster, per axis (buffer px).
            float reach_x = 0.0f, reach_y = 0.0f;
            for (size_t i = 0; i < src_px.size() && i < src_py.size(); ++i) {
                reach_x = fmaxf(reach_x, fmaxf(src_px[i], (float)width  - src_px[i]));
                reach_y = fmaxf(reach_y, fmaxf(src_py[i], (float)height - src_py[i]));
            }
            const float px_buf_x = frame_w / (float)width;
            const float px_buf_y = frame_h / (float)height;
            // Need half-extent S/2 >= reach, i.e. N*dx >= 2*reach*px_buf. On an
            // oversized sensor this saturates at grid_cap (can't reach the far
            // frame), which is exactly enough to bury the box edge.
            const double covN = fmax(2.0 * reach_x * px_buf_x / dx0_x,
                                     2.0 * reach_y * px_buf_y / dx0_y);
            int want = (covN > (double)N) ? (int)std::ceil(covN) : N;
            if (want > grid_cap) want = grid_cap;
            int Ne = 256; while (Ne < want) Ne <<= 1;   // smallest power of two >= want
            if (Ne > N) { N = Ne; eff_engine = StarburstEngine::SPRITE; }  // grown -> large -> sprite
        }
    }

    std::lock_guard<std::mutex> lock(g_star_mutex);
    StarburstGpu& g = star_gpu();

    // ---- Spectral table: real CMF, output-space, spacing pre-multiplied so
    //      brightness is independent of sample count (never a plain 3-RGB). ----
    int ns = (dc.spectral_samples > 0) ? dc.spectral_samples : cfg.spectral_samples;
    if (ns < 3) ns = 3;
    const SensorProfile& prof = get_sensor_profile(cfg.sensor_model);
    float M_out[3][3];
    resolve_output_matrix(cfg.output_cs, cfg.custom_xyz_to_output, M_out);
    std::vector<GPUSpectralSample> spec = build_spectral_table(ns, 400.0f, 700.0f, prof, M_out);
    const float d_lambda = (700.0f - 400.0f) / (float)ns;
    for (GPUSpectralSample& s : spec) { s.cmf_r *= d_lambda; s.cmf_g *= d_lambda; s.cmf_b *= d_lambda; }

    // White-balance: normalise each channel's flat-spectrum response to unity so
    // a white source (equal output-space RGB) yields a neutral starburst overall
    // — matching the ghost renderer's white-for-white behaviour. This is a
    // per-channel gain only: it leaves the spatial dispersion untouched (the
    // physical blue-tight core / red-wide wings survive; the core still reads
    // slightly blue-white and clips to white when bright, as a real sunstar does).
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    for (const GPUSpectralSample& s : spec) { sum_r += s.cmf_r; sum_g += s.cmf_g; sum_b += s.cmf_b; }
    for (GPUSpectralSample& s : spec) {
        if (sum_r > 0.0) s.cmf_r /= (float)sum_r;
        if (sum_g > 0.0) s.cmf_g /= (float)sum_g;
        if (sum_b > 0.0) s.cmf_b /= (float)sum_b;
    }

    // ---- Pupil silhouette (aspect handled by the per-axis pitch, so force 1) ----
    ApertureSamplerParams ap;
    ap.blades_override = cfg.aperture_blades;
    ap.rotation_deg    = cfg.aperture_rotation;
    PupilMask mask = resolve_pupil_mask(lens, ap);
    mask.aspect = 1.0f;

    // ---- Effective-pupil modifiers (cat's-eye / matte box / front-glass texture) ----
    // Representative field angle for the single shared sprite: the mean of the
    // source directions. Exact for a point / single source; a sub-degree
    // approximation for an area source (its offsets cluster around this centre).
    double sum_ax = 0.0, sum_ay = 0.0;
    for (const FlareSource& s : sources) { sum_ax += s.angle_x; sum_ay += s.angle_y; }
    const float field_ax = (float)(sum_ax / (double)sources.size());
    const float field_ay = (float)(sum_ay / (double)sources.size());
    const float tan_ax   = std::tan(field_ax);
    const float tan_ay   = std::tan(field_ay);
    const float z_spawn  = lens.surfaces[0].z - SPAWN_OFFSET;

    // ---- Survivor-envelope centre + off-axis rim vignetting -----------------
    // On axis the envelope is one and its center is zero, so it is skipped.
    float chief_cx = 0.0f, chief_cy = 0.0f;
    const float* d_env_ptr = nullptr;
    int          env_N     = 0;
    std::vector<float> env_host;
    const bool off_axis = (std::fabs(field_ax) > 1e-5f || std::fabs(field_ay) > 1e-5f);
    const bool want_env = dc.use_survivor_mask && off_axis;
    if (want_env) {
        // Centre the window on the ACTUAL surviving crescent (the centroid of
        // launches that reach the sensor), NOT the geometric stop-centre chief.
        // The chief solve flip-flops between branches past the image circle and
        // strobes the starburst in/out of black; the survivor centroid moves
        // smoothly and sits on the crescent by construction, so throughput T
        // fades smoothly out to the TRUE vignette edge (far outside the chief-ray
        // image circle).  If NOTHING survives at this field we are past that edge
        // -> emit no starburst, cleanly and monotonically (the real cut-off).
        bool any_survivor = false;
        solve_survivor_center(lens, field_ax, field_ay, LAMBDA_REF_NM,
                              chief_cx, chief_cy, any_survivor);
        if (!any_survivor) {
            if (dbg) {
                const size_t NN0 = (size_t)N * N;
                dbg->grid = N;
                dbg->psf.assign(NN0, 0.0f);
                dbg->sprite_rgb.assign(NN0 * 3, 0.0f);
                dbg->field_angle_x = field_ax; dbg->field_angle_y = field_ay;
                dbg->throughput = 0.0f;
                dbg->chief_offset_x = 0.0f; dbg->chief_offset_y = 0.0f;
                dbg->f_number_x = fnum_x; dbg->f_number_y = fnum_y;
                dbg->focal_length_x = efl_x; dbg->focal_length_y = efl_y;
                dbg->entrance_pupil_semi_x = ep_x; dbg->entrance_pupil_semi_y = ep_y;
                dbg->lambda_ref_nm = LAMBDA_REF_NM;
            }
            return true;   // past the true vignette edge (buffers pre-zeroed)
        }
        int ME = dc.survivor_grid;
        ME = (ME < 16) ? 16 : (ME > 1024 ? 1024 : ME);
        ApertureSamplerParams ap_real;               // real stop (no blade override)
        PupilMask real_stop = resolve_pupil_mask(lens, ap_real);
        real_stop.aspect = 1.0f;                     // anamorphic handled by ep_x/ep_y
        build_survivor_envelope(lens, real_stop, ep_x, ep_y, chief_cx, chief_cy,
                                field_ax, field_ay, LAMBDA_REF_NM, ME, env_host);
        if (!ensure_floats(g.d_env, g.env_floats, (size_t)ME * ME))
            return fail("cudaMalloc failed (survivor envelope)");
        cudaMemcpy(g.d_env, env_host.data(), (size_t)ME * ME * sizeof(float),
                   cudaMemcpyHostToDevice);
        d_env_ptr = g.d_env;
        env_N     = ME;
    }

    // Off-axis cut-off is applied AFTER the pupil throughput T is measured (see
    // below) — gating on the whole survivor envelope's energy, not a single
    // chief ray. A cat's-eye crescent whose exact centre grazes an aperture
    // (chief clipped) but whose bulk survives must still render, morphing and
    // dimming; a single-ray gate hard-cut it and made the starburst "turn off".

    // Front-glass texture: fold the front-most APERTURE_IMAGE surface's bitmap in
    // as a graded amplitude. RAII pack — freed at function exit, well before the
    // static CUDA teardown, so no cudaFree-after-teardown hazard.
    ApertureTexturePack tex_pack;
    cudaTextureObject_t front_tex  = 0;
    int                 front_surf = -1;
    if (dc.use_surface_textures) {
        std::string tex_err;
        if (upload_aperture_textures(lens, tex_pack, &tex_err)) {
            for (size_t s = 0; s < tex_pack.textures.size(); ++s)
                if (tex_pack.textures[s] != 0) {
                    front_surf = (int)s;
                    front_tex  = (cudaTextureObject_t)tex_pack.textures[s];
                    break;
                }
        } else {
            fprintf(stderr, "render_starburst: texture upload failed (%s); continuing without\n",
                    tex_err.c_str());
        }
    }

    // Assemble the modifier bundle (a default PupilExtras reproduces the plain stop).
    PupilExtras ex_eff;
    ex_eff.env    = d_env_ptr;  ex_eff.env_N = env_N;
    ex_eff.ep_x   = ep_x;       ex_eff.ep_y  = ep_y;
    ex_eff.tan_ax = tan_ax;     ex_eff.tan_ay = tan_ay;
    // Front-of-lens baffle stack: matte box (as one RECT baffle) + config baffles.
    ex_eff.baffles = build_gpu_baffles(dc);
    if (front_tex != 0 && front_surf < (int)lens.surfaces.size()) {
        // Match the tracer's UV scale exactly (trace_cuda.h uses aperture_semi_
        // diameter); fall back to the parallel ApertureImage radius if unset.
        float semi = lens.surfaces[front_surf].aperture_semi_diameter;
        if (semi <= 0.0f && front_surf < (int)lens.aperture_images.size())
            semi = lens.aperture_images[front_surf].semi_diameter;
        if (semi > 0.0f) {
            ex_eff.tex        = front_tex;  ex_eff.has_tex = 1;
            ex_eff.tex_aspect = lens.surfaces[front_surf].aperture_aspect;
            ex_eff.tex_semi   = semi;
            ex_eff.tex_d0     = lens.surfaces[front_surf].z - z_spawn;   // 20 mm for surface 0
            if (ex_eff.tex_aspect <= 0.0f) ex_eff.tex_aspect = 1.0f;
        }
    }
    const bool any_mod = (ex_eff.env != nullptr) || (ex_eff.baffles.n > 0) || ex_eff.has_tex;

    // ---- Allocate device scratch ----
    const size_t NN = (size_t)N * N;
    if (!ensure_bytes (g.d_pupil,  g.pupil_bytes,   NN * sizeof(float2)) ||
        !ensure_floats(g.d_psf,    g.psf_floats,    NN) ||
        !ensure_floats(g.d_sprite, g.sprite_floats, NN * 3) ||
        !ensure_floats(g.d_sprite_ds, g.sprite_ds_floats, NN * 3) ||
        !ensure_bytes (g.d_spec,   g.spec_bytes,    spec.size() * sizeof(GPUSpectralSample)) )
        return fail("cudaMalloc failed (pupil/psf/sprite/spec)");

    const int reduce_blocks = (int)((NN + BLOCK * 2 - 1) / (BLOCK * 2));
    if (!ensure_floats(g.d_partials, g.partials_floats, reduce_blocks))
        return fail("cudaMalloc failed (partials)");

    cudaMemcpy(g.d_spec, spec.data(), spec.size() * sizeof(GPUSpectralSample),
               cudaMemcpyHostToDevice);

    // ---- 1: build the effective pupil, and (if modified) its throughput T ----
    int blocks = (int)((NN + BLOCK - 1) / BLOCK);

    // Pupil throughput is surviving energy divided by on-axis reference energy.
    double T = 1.0;
    auto reduce_energy = [&]() -> double {
        cmagsq_kernel<<<blocks, BLOCK>>>((const float2*)g.d_pupil, g.d_psf, (int)NN);
        reduce_sum_kernel<<<reduce_blocks, BLOCK>>>(g.d_psf, (int)NN, g.d_partials);
        std::vector<float> pr(reduce_blocks);
        cudaMemcpy(pr.data(), g.d_partials, reduce_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        double s = 0.0; for (float p : pr) s += p; return s;
    };
    if (any_mod) {
        build_pupil_kernel<<<blocks, BLOCK>>>((float2*)g.d_pupil, N, fill, mask, PupilExtras{});
        const double S_ref = reduce_energy();
        build_pupil_kernel<<<blocks, BLOCK>>>((float2*)g.d_pupil, N, fill, mask, ex_eff);
        const double S_eff = reduce_energy();
        T = (S_ref > 0.0) ? (S_eff / S_ref) : 1.0;
        T = (T < 0.0) ? 0.0 : (T > 1.0 ? 1.0 : T);
    } else {
        build_pupil_kernel<<<blocks, BLOCK>>>((float2*)g.d_pupil, N, fill, mask, ex_eff);
    }

    // Effective pupil now sits in g.d_pupil (pre-FFT). Capture it for validation.
    if (dbg) {
        std::vector<float2> hp(NN);
        cudaMemcpy(hp.data(), g.d_pupil, NN * sizeof(float2), cudaMemcpyDeviceToHost);
        dbg->pupil.resize(NN);
        for (size_t k = 0; k < NN; ++k) dbg->pupil[k] = hp[k].x;
    }

    // Off-axis cut-off: emit no starburst once the surviving pupil energy has
    // effectively vanished (the field is past the lens's imaged circle). Gating
    // on T — the integrated survivor envelope — rather than a single chief ray
    // means a cat's-eye crescent still renders (morphed, dimmed by T) right up
    // to where the last of the pupil clips. On-axis T==1, so this never fires.
    if (want_env && T < 1e-3) {
        if (dbg) {
            dbg->grid = N;
            dbg->psf.assign(NN, 0.0f);            // consistent (empty) debug arrays
            dbg->sprite_rgb.assign(NN * 3, 0.0f); // (pupil captured just above)
            dbg->field_angle_x = field_ax; dbg->field_angle_y = field_ay;
            dbg->throughput = (float)T;
            dbg->chief_offset_x = chief_cx; dbg->chief_offset_y = chief_cy;
            dbg->f_number_x = fnum_x; dbg->f_number_y = fnum_y;
            dbg->focal_length_x = efl_x; dbg->focal_length_y = efl_y;
            dbg->entrance_pupil_semi_x = ep_x; dbg->entrance_pupil_semi_y = ep_y;
            dbg->lambda_ref_nm = LAMBDA_REF_NM;
        }
        return true;   // no starburst; buffers left as the caller zeroed them
    }

    // ---- Engine fork ------------------------------------------------------
    // Everything above (first-order optics, spectral table, effective pupil in
    // g.d_pupil, throughput T and its cut-off, dbg pupil capture) is engine-
    // agnostic. SPRITE falls through to the FFT-sprite path below. MDFT
    // evaluates the diffraction integral directly at
    // the sensor pixels around each source — no sprite, no resample.
    if (eff_engine == StarburstEngine::MDFT) {
        // Physical scale + gain, mirroring the sprite path's own definitions.
        const float px_mm_x = 2.0f * sensor_half_w / (float)fmt_w;
        const float px_mm_y = 2.0f * sensor_half_h / (float)fmt_h;
        const float trim = (dc.scale_trim > 0.0f) ? dc.scale_trim : 1.0f;
        const float dx_x = trim * dx_fine_x;    // trimmed pattern texel pitch (mm)
        const float dx_y = trim * dx_fine_y;
        constexpr float APERTURE_REF_FNUMBER = 2.8f;
        const float aperture_scale = (APERTURE_REF_FNUMBER * APERTURE_REF_FNUMBER)
                                   * calib.pupil_area_frac
                                   / fmaxf(fnum_x * fnum_y, 1e-6f);
        const float gain = dc.starburst_gain * (float)T * aperture_scale;

        std::vector<MdftSource> msrc(sources.size());
        for (size_t i = 0; i < sources.size(); ++i)
            msrc[i] = { src_px[i], src_py[i], sources[i].r, sources[i].g, sources[i].b };

        std::string mdft_err;
        bool ok = mdft_starburst(g.d_pupil, N, fill, spec, LAMBDA_REF_NM,
                                 dx_x, dx_y, px_mm_x, px_mm_y, msrc, gain,
                                 out_r, out_g, out_b, width, height,
                                 &mdft_err);
        if (!ok) return fail(mdft_err.empty() ? "mdft engine failed" : mdft_err.c_str());

        if (dbg) {
            // The MDFT engine has no intermediate sprite/PSF; keep the debug
            // arrays consistently shaped (zeroed) so the binding reshape holds,
            // and report the same scalars the sprite path does.
            dbg->grid = N;
            dbg->psf.assign(NN, 0.0f);
            dbg->sprite_rgb.assign(NN * 3, 0.0f);
            dbg->dx_mm_x = dx_x; dbg->dx_mm_y = dx_y;
            dbg->f_number_x = fnum_x; dbg->f_number_y = fnum_y;
            dbg->focal_length_x = efl_x; dbg->focal_length_y = efl_y;
            dbg->entrance_pupil_semi_x = ep_x; dbg->entrance_pupil_semi_y = ep_y;
            dbg->lambda_ref_nm = LAMBDA_REF_NM;
            dbg->field_angle_x = field_ax; dbg->field_angle_y = field_ay;
            dbg->throughput = (float)T;
            dbg->chief_offset_x = chief_cx; dbg->chief_offset_y = chief_cy;
        }
        return true;
    }

    // ---- 2-5: FFT the effective pupil into the RGB sprite. Two paths converge on
    //      g.d_sprite; steps 6-7 (downsample + splat) below are shared. ----
    std::vector<float> partials(reduce_blocks);

    // Achromatic (W=0): ONE FFT, then the Fourier scaling-theorem bake spreads it
    // across wavelength.
    const char* fft_err = nullptr;
    if (!fft2d_c2c_forward_inplace(g.d_pupil, N, &fft_err))
        return fail(fft_err ? fft_err : "fft failed");

    psf_from_fft_kernel<<<blocks, BLOCK>>>((const float2*)g.d_pupil, g.d_psf, N);

    // energy-normalise the mono PSF to unit sum
    reduce_sum_kernel<<<reduce_blocks, BLOCK>>>(g.d_psf, (int)NN, g.d_partials);
    if (cudaMemcpy(partials.data(), g.d_partials, reduce_blocks * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail("cudaMemcpy failed (pupil energy reduction)");
    double sum = 0.0;
    for (float p : partials) sum += p;
    // A modifier (matte box fully closed, an opaque texture, an all-vignetted
    // crescent that survived the chief test only marginally) can still leave the
    // pupil with no energy. That is a legitimate "no starburst", not an error —
    // return cleanly rather than logging a degenerate-pupil failure per source.
    if (sum <= 0.0) {
        if (any_mod) return true;
        return fail("degenerate pupil (zero energy)");
    }
    scale_kernel<<<blocks, BLOCK>>>(g.d_psf, (int)NN, (float)(1.0 / sum));

    bake_sprite_kernel<<<blocks, BLOCK>>>(g.d_psf, N,
        (const GPUSpectralSample*)g.d_spec, (int)spec.size(), LAMBDA_REF_NM, g.d_sprite);

    // ---- 6: downsample toward the output pixel pitch ----
    const float px_mm_x = 2.0f * sensor_half_w / (float)fmt_w;
    const float px_mm_y = 2.0f * sensor_half_h / (float)fmt_h;
    const float trim = (dc.scale_trim > 0.0f) ? dc.scale_trim : 1.0f;
    const float sx = trim * dx_fine_x;    // effective fine pitch on the sensor (mm)
    const float sy = trim * dx_fine_y;
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

    // ---- 7: splat per source ----
    // All three output channels share one capacity and grow together.
    const size_t n_px = (size_t)width * height;
    if (n_px > g.out_floats) {
        cudaFree(g.d_out_r); cudaFree(g.d_out_g); cudaFree(g.d_out_b);
        g.d_out_r = g.d_out_g = g.d_out_b = nullptr;
        g.out_floats = 0;
        if (cudaMalloc(&g.d_out_r, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_g, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_b, n_px * sizeof(float)) != cudaSuccess)
            return fail("cudaMalloc failed (starburst output)");
        g.out_floats = n_px;
    }
    cudaMemset(g.d_out_r, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_g, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_b, 0, n_px * sizeof(float));

    const int n_src = (int)sources.size();
    std::vector<GPUStarSource> host_src(n_src);
    for (int i = 0; i < n_src; ++i)
        host_src[i] = { src_px[i], src_py[i], sources[i].r, sources[i].g, sources[i].b };
    if (!ensure_bytes(g.d_src, g.src_bytes, n_src * sizeof(GPUStarSource)))
        return fail("cudaMalloc d_src");
    cudaMemcpy(g.d_src, host_src.data(), n_src * sizeof(GPUStarSource),
               cudaMemcpyHostToDevice);

    // The normalized sprite integrates to source flux and is independent of
    // flare_gain. T applies pupil throughput; aperture collection scales with
    // entrance-pupil area, 1/(f#_x*f#_y), relative to the reference f-number.
    constexpr float APERTURE_REF_FNUMBER = 2.8f;
    // Correct the f-number's bounding-disk area for non-circular pupils.
    const float aperture_scale = (APERTURE_REF_FNUMBER * APERTURE_REF_FNUMBER)
                               * calib.pupil_area_frac
                               / fmaxf(fnum_x * fnum_y, 1e-6f);
    const float gain = dc.starburst_gain * (float)T * aperture_scale;

    // Area-weighted resample of the sprite onto the output grid — one thread per
    // (source, output pixel) in the sprite's on-sensor footprint. Correct and
    // hole/moiré-free at every scale (see splat_resample_kernel). rx/ry bound the
    // footprint to the frame so an extreme scale_trim doesn't launch a sea of
    // out-of-bounds threads. (The prior downsample keeps the per-pixel texel box
    // small — a few texels — in the common finer-than-pixel case.)
    int rx = (int)std::ceil(0.5 * M * sx_ds / px_mm_x) + 1;
    int ry = (int)std::ceil(0.5 * M * sy_ds / px_mm_y) + 1;
    if (rx > width)  rx = width;
    if (ry > height) ry = height;
    long long total = (long long)n_src * (2LL * rx + 1) * (2LL * ry + 1);
    int splat_blocks = (int)((total + BLOCK - 1) / BLOCK);
    splat_resample_kernel<<<splat_blocks, BLOCK>>>(splat_src, M, sx_ds, sy_ds,
        px_mm_x, px_mm_y, rx, ry, (const GPUStarSource*)g.d_src, n_src, gain,
        g.d_out_r, g.d_out_g, g.d_out_b, width, height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail(cudaGetErrorString(err));

    // ---- Copy the starburst layer out (caller-zeroed buffers) ----
    cudaMemcpy(out_r, g.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_g, g.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b, g.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Optional validation capture ----
    if (dbg) {
        dbg->grid = N;
        dbg->psf.resize(NN);
        dbg->sprite_rgb.resize(NN * 3);
        cudaMemcpy(dbg->psf.data(), g.d_psf, NN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg->sprite_rgb.data(), g.d_sprite, NN * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        dbg->dx_mm_x = sx; dbg->dx_mm_y = sy;
        dbg->f_number_x = fnum_x; dbg->f_number_y = fnum_y;
        dbg->focal_length_x = efl_x; dbg->focal_length_y = efl_y;
        dbg->entrance_pupil_semi_x = ep_x; dbg->entrance_pupil_semi_y = ep_y;
        dbg->lambda_ref_nm = LAMBDA_REF_NM;
        dbg->field_angle_x = field_ax; dbg->field_angle_y = field_ay;
        dbg->throughput = (float)T;
        dbg->chief_offset_x = chief_cx; dbg->chief_offset_y = chief_cy;
    }
    return true;
}
