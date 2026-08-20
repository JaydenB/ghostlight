// ============================================================================
// mdft_render.cu — Matrix-DFT starburst engine (see mdft_render.h).
//
// Pipeline per render:
//   1. autocorrelation  C = ifft2(|fft2(A)|^2), compact support Mc x Mc,
//      normalised so the pattern integrates to one texel-area over a period
//      to match the sprite engine's unit-sum PSF.
//   2. per source, per wavelength: two small separable matrix products evaluate
//      the exact pixel-box-integrated intensity at the window pixels:
//         tmp[a][kx] = sum_b C[a][b] Px[b][kx]                 (Mc x Kx)
//         I[ky][kx]  = Re sum_a Py[ky][a] tmp[a][kx]           (Ky x Kx)
//      Px/Py carry the sinc box factor and the per-wavelength scaled phase, so
//      the result is an exact area integral — alias-free as the source moves.
//   3. splat: window * K_norm(px^2/dx^2) * source colour * gain into the output.
//
// C is real for an amplitude pupil (W=0), built ONCE then reused across all
// colours (its autocorrelation is achromatic).  Convention: C[a][b] has a = y-lag
// (row), b = x-lag (col); Px pairs with x (b), Py with y (a).
// ============================================================================

#include "mdft_render.h"
#include "fft_backend.h"
#include "spectral.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {

constexpr int   BLOCK = 256;
constexpr float PI_F  = 3.14159265358979323846f;

// Persistent device scratch (leaked singleton + mutex, mirroring the sprite /
// ghost caches — never destroyed, so no cudaFree races the CUDA teardown).
struct MdftGpu {
    void*   d_work   = nullptr; size_t work_bytes = 0;  // float2 [N*N] autocorr FFT work
    void*   d_C      = nullptr; size_t C_bytes    = 0;  // float2 [Mc*Mc] autocorrelation
    void*   d_Px     = nullptr; size_t Px_bytes   = 0;  // float2 [Mc*Kx]
    void*   d_Py     = nullptr; size_t Py_bytes   = 0;  // float2 [Ky*Mc]
    void*   d_tmp    = nullptr; size_t tmp_bytes  = 0;  // float2 [Mc*Kx]
    float*  d_win    = nullptr; size_t win_floats = 0;  // [3*Kx*Ky] RGB window
    float*  d_out_r  = nullptr;
    float*  d_out_g  = nullptr;
    float*  d_out_b  = nullptr;
    size_t  out_floats = 0;
};

std::mutex g_mdft_mutex;
MdftGpu&   mdft_gpu() { static MdftGpu* c = new MdftGpu(); return *c; }

bool ensure_bytes(void*& p, size_t& cap, size_t need) {
    if (need <= cap) return true;
    cudaFree(p); p = nullptr; cap = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cap = need; return true;
}
bool ensure_floats(float*& p, size_t& cap, size_t need) {
    if (need <= cap) return true;
    cudaFree(p); p = nullptr; cap = 0;
    if (cudaMalloc(&p, need * sizeof(float)) != cudaSuccess) return false;
    cap = need; return true;
}

// Normalised sinc, sin(pi x)/(pi x), with the removable singularity handled.
__device__ inline float sincf_(float x) {
    if (fabsf(x) < 1e-7f) return 1.0f;
    float a = PI_F * x;
    return sinf(a) / a;
}

// |F|^2 into a complex buffer (imag 0), in place: dst[i] = (|F_i|^2, 0).
__global__ void magsq_kernel(float2* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 z = d[i];
    d[i] = make_float2(z.x * z.x + z.y * z.y, 0.0f);
}

// Extract the compact autocorrelation support from G = fft2(|F|^2).
//   C[a][b] = conj( G[ gy*N + gx ] ),  gy = (a-mc+N)%N, gx = (b-mc+N)%N
// (a = y-lag, b = x-lag).  The 1/N^2 of the inverse transform and the unit-sum
// normalisation 1/(G00*N^2) are folded into a single scale applied afterwards.
__global__ void extract_C_kernel(float2* d_C, const float2* d_G, int N, int mc) {
    int Mc  = 2 * mc + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Mc * Mc) return;
    int b = idx % Mc;             // x-lag index
    int a = idx / Mc;             // y-lag index
    int gx = ((b - mc) % N + N) % N;
    int gy = ((a - mc) % N + N) % N;
    float2 g = d_G[(size_t)gy * N + gx];
    d_C[idx] = make_float2(g.x, -g.y);   // conj
}

__global__ void scale_complex_kernel(float2* d, int n, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    d[i].x *= k; d[i].y *= k;
}

// Build the x-axis factor Px[b][kx] (Mc x Kx), row-major b*Kx+kx.
//   m = b - mc;  xrel = ((kx0 + kx + 0.5) - src_px) * px_mm_x   (mm)
//   Px = sinc(m*px_mm_x/Sx_l) * exp(-i 2pi m xrel / Sx_l),  Sx_l = Sx * lam_scale
__global__ void build_px_kernel(float2* d_Px, int Mc, int Kx, int mc,
                                float Sx_l, float px_mm_x, int kx0, float src_px) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Mc * Kx) return;
    int kx = idx % Kx;
    int b  = idx / Kx;
    float m    = (float)(b - mc);
    float xrel = ((float)(kx0 + kx) + 0.5f - src_px) * px_mm_x;
    float w    = sincf_(m * px_mm_x / Sx_l);
    float arg  = 2.0f * PI_F * m * xrel / Sx_l;
    d_Px[idx]  = make_float2(w * cosf(arg), -w * sinf(arg));
}

// Build the y-axis factor Py[ky][a] (Ky x Mc), row-major ky*Mc+a.
__global__ void build_py_kernel(float2* d_Py, int Ky, int Mc, int mc,
                                float Sy_l, float px_mm_y, int ky0, float src_py) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ky * Mc) return;
    int a  = idx % Mc;
    int ky = idx / Mc;
    float m    = (float)(a - mc);
    float yrel = ((float)(ky0 + ky) + 0.5f - src_py) * px_mm_y;
    float w    = sincf_(m * px_mm_y / Sy_l);
    float arg  = 2.0f * PI_F * m * yrel / Sy_l;
    d_Py[idx]  = make_float2(w * cosf(arg), -w * sinf(arg));
}

// Contract along x: tmp[a][kx] = sum_b C[a][b] * Px[b][kx].
__global__ void contract_x_kernel(float2* d_tmp, const float2* d_C, const float2* d_Px,
                              int Mc, int Kx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Mc * Kx) return;
    int kx = idx % Kx;
    int a  = idx / Kx;
    const float2* Crow = d_C + (size_t)a * Mc;
    float sr = 0.0f, si = 0.0f;
    for (int b = 0; b < Mc; ++b) {
        float2 c = Crow[b];
        float2 p = d_Px[(size_t)b * Kx + kx];
        sr += c.x * p.x - c.y * p.y;
        si += c.x * p.y + c.y * p.x;
    }
    d_tmp[idx] = make_float2(sr, si);
}

// Contract along y and accumulate into the RGB window, weighting this
// wavelength by its CMF value and the k^2 unit-sum Jacobian.
__global__ void accumulate_y_kernel(float* d_win, const float2* d_Py, const float2* d_tmp,
                                    int Ky, int Kx, int Mc,
                                    float cw_r, float cw_g, float cw_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Kx * Ky) return;
    int kx = idx % Kx;
    int ky = idx / Kx;
    const float2* Pyrow = d_Py + (size_t)ky * Mc;
    float acc = 0.0f;
    for (int a = 0; a < Mc; ++a) {
        float2 py = Pyrow[a];
        float2 t  = d_tmp[(size_t)a * Kx + kx];
        acc += py.x * t.x - py.y * t.y;         // Re(py * tmp)
    }
    d_win[3 * idx + 0] += cw_r * acc;
    d_win[3 * idx + 1] += cw_g * acc;
    d_win[3 * idx + 2] += cw_b * acc;
}

// Deposit the finished window into the output buffers (atomicAdd — windows of
// different sources may overlap).  value = K_norm * win * source colour * gain.
__global__ void splat_window_kernel(const float* d_win, int Kx, int Ky,
                                    int kx0, int ky0, int width, int height,
                                    float knorm_gain, float sr, float sg, float sb,
                                    float* d_out_r, float* d_out_g, float* d_out_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Kx * Ky) return;
    int kx = idx % Kx;
    int ky = idx / Kx;
    int ix = kx0 + kx, iy = ky0 + ky;
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;
    int pix = iy * width + ix;
    float vr = d_win[3 * idx + 0] * knorm_gain * sr;
    float vg = d_win[3 * idx + 1] * knorm_gain * sg;
    float vb = d_win[3 * idx + 2] * knorm_gain * sb;
    if (fabsf(vr) > 1e-20f) atomicAdd(&d_out_r[pix], vr);
    if (fabsf(vg) > 1e-20f) atomicAdd(&d_out_g[pix], vg);
    if (fabsf(vb) > 1e-20f) atomicAdd(&d_out_b[pix], vb);
}

} // namespace

// ---------------------------------------------------------------------------
bool mdft_starburst(const void*                          d_pupil_amp,
                    int                                  N,
                    float                                pupil_fill,
                    const std::vector<GPUSpectralSample>& spec,
                    float                                lambda_ref_nm,
                    float                                dx_x,
                    float                                dx_y,
                    float                                px_mm_x,
                    float                                px_mm_y,
                    const std::vector<MdftSource>&       sources,
                    float                                gain,
                    float*                               out_r,
                    float*                               out_g,
                    float*                               out_b,
                    int                                  width,
                    int                                  height,
                    std::string*                         err)
{
    auto fail = [&](const char* msg) {
        if (err && err->empty()) *err = msg;
        fprintf(stderr, "mdft_starburst: %s\n", msg);
        return false;
    };
    if (sources.empty() || N <= 0) return true;

    std::lock_guard<std::mutex> lock(g_mdft_mutex);
    MdftGpu& g = mdft_gpu();

    const size_t NN = (size_t)N * N;
    const float  Sx = (float)N * dx_x;      // pattern period (mm), per axis
    const float  Sy = (float)N * dx_y;
    const float  fill = (pupil_fill > 0.02f && pupil_fill < 0.9f) ? pupil_fill : 0.30f;
    // Compact autocorrelation support: |lag| <= fill*N (+ margin).  2*fill < 1 in
    // the sane regime, so the N-grid circular autocorrelation is wrap-free.
    int mc = (int)std::ceil(fill * (double)N) + 4;
    if (mc > N / 2 - 1) mc = N / 2 - 1;
    const int Mc = 2 * mc + 1;

    // ---- 1: autocorrelation C, built ONCE and reused across all colours (its
    //      autocorrelation is achromatic for an amplitude pupil, W=0). --------------
    const size_t McMc = (size_t)Mc * Mc;
    if (!ensure_bytes(g.d_work, g.work_bytes, NN * sizeof(float2)) ||
        !ensure_bytes(g.d_C,    g.C_bytes,    McMc * sizeof(float2)))
        return fail("cudaMalloc failed (autocorr work / C)");

    const char* fe = nullptr;
    int nblk = (int)((NN + BLOCK - 1) / BLOCK);
    int cblk = (int)((McMc + BLOCK - 1) / BLOCK);

    // A -> work, F = fft2(A), |F|^2, G = fft2(|F|^2).
    cudaMemcpy(g.d_work, d_pupil_amp, NN * sizeof(float2), cudaMemcpyDeviceToDevice);
    if (!fft2d_c2c_forward_inplace(g.d_work, N, &fe)) return fail(fe ? fe : "fft(A) failed");
    magsq_kernel<<<nblk, BLOCK>>>((float2*)g.d_work, (int)NN);
    if (!fft2d_c2c_forward_inplace(g.d_work, N, &fe)) return fail(fe ? fe : "fft(|F|^2) failed");
    extract_C_kernel<<<cblk, BLOCK>>>((float2*)g.d_C, (const float2*)g.d_work, N, mc);

    // Normalise C by 1/(G00 * N^2). G00 = C[centre].x = sum|A|^2.
    float2 c00;
    if (cudaMemcpy(&c00, (const float2*)g.d_C + (size_t)mc * Mc + mc, sizeof(float2),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail("cudaMemcpy failed (C00)");
    if (!(c00.x > 0.0f)) return true;   // degenerate/empty pupil — no starburst
    float cscale = 1.0f / (c00.x * (float)N * (float)N);
    scale_complex_kernel<<<cblk, BLOCK>>>((float2*)g.d_C, (int)McMc, cscale);

    // ---- output buffers (own device output, overwrite host out) ----
    const size_t n_px = (size_t)width * height;
    if (n_px > g.out_floats) {
        cudaFree(g.d_out_r); cudaFree(g.d_out_g); cudaFree(g.d_out_b);
        g.d_out_r = g.d_out_g = g.d_out_b = nullptr; g.out_floats = 0;
        if (cudaMalloc(&g.d_out_r, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_g, n_px * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&g.d_out_b, n_px * sizeof(float)) != cudaSuccess)
            return fail("cudaMalloc failed (mdft output)");
        g.out_floats = n_px;
    }
    cudaMemset(g.d_out_r, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_g, 0, n_px * sizeof(float));
    cudaMemset(g.d_out_b, 0, n_px * sizeof(float));

    // Half-window in pixels is half the pattern period, the maximum reach
    // without wrap. kx0/kx1 clamp it to the frame below.
    int half_x = (int)std::floor(0.5 * (double)Sx / (double)px_mm_x);
    int half_y = (int)std::floor(0.5 * (double)Sy / (double)px_mm_y);
    if (half_x < 1) half_x = 1;
    if (half_y < 1) half_y = 1;

    // K_norm converts the box-average intensity (unit-total pattern) into the
    // sprite's per-pixel texel-density units; gain folds throughput / aperture.
    const float knorm_gain = (px_mm_x * px_mm_y) / (dx_x * dx_y) * gain;

    // ---- 2-3: per source, per wavelength ----------------------------------
    for (const MdftSource& s : sources) {
        int kx0 = std::max(0, (int)std::floor(s.px) - half_x);
        int kx1 = std::min(width,  (int)std::floor(s.px) + half_x + 1);
        int ky0 = std::max(0, (int)std::floor(s.py) - half_y);
        int ky1 = std::min(height, (int)std::floor(s.py) + half_y + 1);
        int Kx = kx1 - kx0, Ky = ky1 - ky0;
        if (Kx <= 0 || Ky <= 0) continue;    // source too far off-frame

        if (!ensure_bytes (g.d_Px,  g.Px_bytes,  (size_t)Mc * Kx * sizeof(float2)) ||
            !ensure_bytes (g.d_Py,  g.Py_bytes,  (size_t)Ky * Mc * sizeof(float2)) ||
            !ensure_bytes (g.d_tmp, g.tmp_bytes, (size_t)Mc * Kx * sizeof(float2)) ||
            !ensure_floats(g.d_win, g.win_floats, (size_t)3 * Kx * Ky))
            return fail("cudaMalloc failed (window scratch)");
        cudaMemset(g.d_win, 0, (size_t)3 * Kx * Ky * sizeof(float));

        int px_blk = (int)(((size_t)Mc * Kx + BLOCK - 1) / BLOCK);
        int py_blk = (int)(((size_t)Ky * Mc + BLOCK - 1) / BLOCK);
        int s1_blk = px_blk;
        int s2_blk = (int)(((size_t)Kx * Ky + BLOCK - 1) / BLOCK);

        for (int li = 0; li < (int)spec.size(); ++li) {
            const GPUSpectralSample& sm = spec[li];
            float lam_scale = sm.lambda / lambda_ref_nm;      // S_l = S * lam/lam_ref
            float k2 = (lambda_ref_nm / sm.lambda) * (lambda_ref_nm / sm.lambda);
            float Sx_l = Sx * lam_scale, Sy_l = Sy * lam_scale;
            build_px_kernel<<<px_blk, BLOCK>>>((float2*)g.d_Px, Mc, Kx, mc,
                                               Sx_l, px_mm_x, kx0, s.px);
            build_py_kernel<<<py_blk, BLOCK>>>((float2*)g.d_Py, Ky, Mc, mc,
                                               Sy_l, px_mm_y, ky0, s.py);
            const float2* C_l = (const float2*)g.d_C;
            contract_x_kernel<<<s1_blk, BLOCK>>>((float2*)g.d_tmp, C_l,
                                             (const float2*)g.d_Px, Mc, Kx);
            accumulate_y_kernel<<<s2_blk, BLOCK>>>(g.d_win, (const float2*)g.d_Py,
                                                   (const float2*)g.d_tmp, Ky, Kx, Mc,
                                                   sm.cmf_r * k2, sm.cmf_g * k2, sm.cmf_b * k2);
        }

        splat_window_kernel<<<s2_blk, BLOCK>>>(g.d_win, Kx, Ky, kx0, ky0, width, height,
                                               knorm_gain, s.r, s.g, s.b,
                                               g.d_out_r, g.d_out_g, g.d_out_b);
    }

    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) return fail(cudaGetErrorString(ce));

    cudaMemcpy(out_r, g.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_g, g.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b, g.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost);
    return true;
}
