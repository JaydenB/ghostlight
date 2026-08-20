// ============================================================================
// ghost_render.cu — GPU ghost scatter kernel and CPU launcher
// ============================================================================

#include "ghost_render.h"
#include "aperture_sampler.h"
#include "aperture_sdf.h"        // find_sdf_target_surface (SdfKey build)
#include "aperture_textures.h"
#include "baffle.h"              // GpuBaffleStack, build_gpu_baffles
#include "hurb.h"                // GpuHurb, build_gpu_hurb
#include "spawn_plane.h"         // SPAWN_OFFSET, spawn_shift
#include "spectral.h"
#include "trace_cuda.h"
#include "optical_system.h"
#include "ghost.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <mutex>
#include <vector>
#include <algorithm>

// ===========================================================================
// GpuBufferCache — persistent device-memory pool (internal to this TU)
// ===========================================================================

// ONE process-wide instance, a mutex-guarded leaked singleton (never destroyed,
// so we don't cudaFree after the static CUDA runtime tears down at exit).
// Callers do not create or pass one. Each buffer grows only when the new data is
// larger than the current allocation, avoiding a cudaMalloc + cudaFree pair every
// frame. Internal GPU POD types (GPUPair, GPUSource, etc.) stay hidden behind void*.
struct GpuBufferCache
{
    // Input parameter buffers (void* hides internal GPU POD types).
    void*  d_surfs        = nullptr;  std::size_t surfs_bytes        = 0;
    void*  d_pairs        = nullptr;  std::size_t pairs_bytes        = 0;
    void*  d_src          = nullptr;  std::size_t src_bytes          = 0;
    void*  d_grid         = nullptr;  std::size_t grid_bytes         = 0;
    void*  d_spec         = nullptr;  std::size_t spec_bytes         = 0;
    // Per-(λ, surface) IOR tables — n_spec × n_surfs floats each,
    // laid out [λ][surface] so each λ slice is contiguous. ior_at = surface's
    // post-medium IOR; ior_before = medium on the "before" side (last active
    // surface), precomputed so the trace skips the per-hit Sellmeier eval and
    // the O(N²) backward walk.
    void*  d_ior_at       = nullptr;  std::size_t ior_at_bytes       = 0;
    void*  d_ior_before   = nullptr;  std::size_t ior_before_bytes   = 0;
    // Off-sensor cull: coarse probe grid + per-pair active mask
    // (1 = trace, 0 = culled). See launch_ghost_render().
    void*  d_probe_grid   = nullptr;  std::size_t probe_grid_bytes   = 0;
    void*  d_pair_active  = nullptr;  std::size_t pair_active_bytes  = 0;
    // Concentration: per-(pair, source) pupil-space survivor bounds.
    // d_ps_bbox: probe-side atomic accumulator (biased-float-as-int mins /
    // maxs + hit counts; 5 ints per slot, mins block | maxs block | hits).
    // d_ps_rect: host-finalized sample rects (float4 {u0,v0,u1,v1} per slot;
    // u0 > u1 marks invalid -> that (pair, source) uses the full-pupil path).
    void*  d_ps_bbox      = nullptr;  std::size_t ps_bbox_bytes      = 0;
    void*  d_ps_rect      = nullptr;  std::size_t ps_rect_bytes      = 0;
    // Per-(pair, source) adaptive sample budget {n_r_ps, nr_lat}
    // (int2). Only populated when FlareConfig::adaptive_sample_budgets is set.
    void*  d_ps_budget    = nullptr;  std::size_t ps_budget_bytes    = 0;
    // Image-aperture texture lifecycle (shared with the PSF renderer).
    // See aperture_textures.h.
    ApertureTexturePack apertures;
    // Signed-distance field for the image-aperture STOP (float4 [d,nx,ny,0]),
    // used by the HURB edge kick. Baked+uploaded only when HURB is on; keyed by
    // g_sdf_key so an unchanged mask skips the re-bake. See aperture_sdf.h.
    ApertureTexturePack aperture_sdf;
    // Device blob for table-backed coating models (see coating_upload.h).
    CoatingDeviceArena coatings;

    // Output accumulation buffers — float* so their size is self-documenting.
    // All three channels are always grown together (same pixel count).
    float* d_out_r    = nullptr;
    float* d_out_g    = nullptr;
    float* d_out_b    = nullptr;
    std::size_t out_floats = 0;   // capacity of each channel in floats

    // Free all device allocations and reset capacities to zero.
    // Safe to call multiple times and on a partially-constructed cache.
    void release();

    ~GpuBufferCache() { release(); }

    GpuBufferCache()  = default;
    GpuBufferCache(const GpuBufferCache&)            = delete;
    GpuBufferCache& operator=(const GpuBufferCache&) = delete;
};

void GpuBufferCache::release()
{
    // ApertureTexturePack's destructor handles its own teardown ordering.
    apertures.release();
    aperture_sdf.release();
    coatings.release();

    cudaFree(d_surfs);        d_surfs        = nullptr;  surfs_bytes        = 0;
    cudaFree(d_pairs);        d_pairs        = nullptr;  pairs_bytes        = 0;
    cudaFree(d_src);          d_src          = nullptr;  src_bytes          = 0;
    cudaFree(d_grid);         d_grid         = nullptr;  grid_bytes         = 0;
    cudaFree(d_spec);         d_spec         = nullptr;  spec_bytes         = 0;
    cudaFree(d_ior_at);       d_ior_at       = nullptr;  ior_at_bytes       = 0;
    cudaFree(d_ior_before);   d_ior_before   = nullptr;  ior_before_bytes   = 0;
    cudaFree(d_probe_grid);   d_probe_grid   = nullptr;  probe_grid_bytes   = 0;
    cudaFree(d_pair_active);  d_pair_active  = nullptr;  pair_active_bytes  = 0;
    cudaFree(d_ps_bbox);      d_ps_bbox      = nullptr;  ps_bbox_bytes      = 0;
    cudaFree(d_ps_rect);      d_ps_rect      = nullptr;  ps_rect_bytes      = 0;
    cudaFree(d_ps_budget);    d_ps_budget    = nullptr;  ps_budget_bytes    = 0;
    cudaFree(d_out_r);        d_out_r        = nullptr;
    cudaFree(d_out_g);        d_out_g        = nullptr;
    cudaFree(d_out_b);        d_out_b        = nullptr;
    out_floats = 0;
}

// ===========================================================================
// GPU-side POD types
// ===========================================================================

struct GPUPair      { int surf_a, surf_b; float area_boost; };
struct GPUSource    { float angle_x, angle_y, r, g, b; };
// ApertureSample (from aperture_sampler.h) is the entrance-pupil grid type.

// ===========================================================================
// Scatter kernel
// ===========================================================================

static constexpr int BLOCK_SIZE = 256;

// Off-sensor cull: side length of the coarse pupil probe grid
// (40x40 -> ~1250 masked samples per pair, per source, per wavelength), a
// sensor-box margin for the on-sensor test, and the number of independently
// jittered probe passes. The margin keeps pairs whose on-sensor footprint sits
// just past the frame edge (a finite-footprint ghost centred just off-frame still
// lands on-sensor pixels). A SINGLE coarse probe can
// still miss a thin on-sensor sliver in the pupil interior (e.g. an
// aperture_blades-clipped anamorphic ghost) and false-cull a live pair, so we
// run CULL_PROBE_PASSES wang-hash-jittered probes and keep a pair if ANY pass
// sees it on-sensor (cull only when every pass agrees it is dead). Together
// these reduce false culls of thin survivor regions. See launch_ghost_render().
static constexpr int   PROBE_GRID        = 40;
static constexpr float CULL_MARGIN       = 1.25f;
static constexpr int   CULL_PROBE_PASSES = 3;

// Concentration: the probe additionally records, per (pair, source),
// the pupil-space bounding box of samples that reached the (margin-inflated)
// sensor. The main kernel then lays that (pair, source)'s whole sample budget
// uniformly over the box instead of the full pupil. The unbiased per-ray weight
// is (|R| / A_ref) / n_R with
// A_ref = 4 * n_grid / N^2 (the mask area as measured by the SAME grid), and
// every sample re-tested against the shared PupilMask in-kernel — samples the
// mask rejects contribute zero but still count in n_R. In the full-pupil limit
// this reproduces sum(T_i) / n_grid identically.
//   CONC_DILATE_CELLS: box dilation in coarse-probe cells (edge falls at most
//     one cell from a hit sample; 1.5 covers it + jitter slack).
//   CONC_MIN_HITS: minimum distinct surviving pupil positions before the box
//     is trusted; below it that (pair, source) falls back to the full-pupil
//     path (concentration never hard-culls on thin evidence).
static constexpr float CONC_DILATE_CELLS = 1.5f;
static constexpr int   CONC_MIN_HITS     = 3;

// Densely re-probe under-resolved entries so small survivor regions do not
// alternate between concentration, fallback, and culling. Min/mid/max
// wavelengths bracket chromatic spread; REFINE_MAX_ENTRIES bounds the work.
static constexpr int REFINE_GRID        = 128;   // dense re-probe side (~10x a coarse pass)
static constexpr int REFINE_HITS_MAX    = 16;    // refine entries with hits < this
static constexpr int REFINE_SPEC        = 3;     // min/mid/max wavelengths for the re-probe
static constexpr int REFINE_MAX_ENTRIES = 4096;  // cost guard: skip refinement above this

// Per-launch concentration inputs, passed to the kernel by value.
// rects == nullptr disables the concentrated path entirely (probe passes, the
// stats diagnostic, AOV launches, and concentrate_samples == false).
struct ConcParams
{
    const float4* rects   = nullptr; // [n_pairs*n_src] {u0,v0,u1,v1}; u0>u1 = invalid
    PupilMask     mask;              // shared silhouette predicate (aperture_sampler.h)
    int           jitter  = 2;       // pupil_jitter mode (0 regular / 1 wang / 2 Halton)
    unsigned      seed_offset = 0;   // wang-hash stream offset (seed * 1000003)
    int           nr_lattice  = 1;   // lattice side for jitter modes 0/1
    int           n_r         = 1;   // samples laid over the rect (estimator denominator)
    float         area_norm   = 0.f; // 1 / (A_ref * n_r); weight = |R| * area_norm
    // budgets == nullptr gives every (pair, source) the uniform budget above.
    // Non-null -> per-ps {n_r_ps, nr_lat_ps}; the estimator weight becomes
    // |R| * inv_aref / n_r_ps, unbiased for any per-ps count.
    const int2*   budgets     = nullptr;
    float         inv_aref    = 0.f; // 1 / A_ref (used only when budgets != nullptr)
};

// Orderable encoding for pupil coords in [-1, 1]: bias to a positive float so
// the raw IEEE bit pattern is monotonic as a plain int -> atomicMin/atomicMax
// work directly. (No sign-flip trick needed; 0x7F7F7F7F memset > any encode.)
__device__ static inline int   enc_pupil(float x) { return __float_as_int(x + 2.0f); }
static inline float dec_pupil_host(int i)
{
    float f;
    std::memcpy(&f, &i, sizeof(f));
    return f - 2.0f;
}

// STATS = false: the real render (splats into d_out_*; the counter pointers are
//                unused and passed nullptr).
// STATS = true:  a diagnostic pass that repeats the trace but NEVER splats —
//                it only tallies per-pair traces / survivors / on-sensor hits
//                into the d_pair_* counters. Run separately (and un-timed) so
//                the STATS=false timing stays clean.
template<bool STATS, bool HURB>
__global__ __launch_bounds__(BLOCK_SIZE) void ghost_kernel(
    const Surface*       d_surfs,
    int                  n_surfs,
    const GPUPair*       d_pairs,
    int                  n_sources,
    const GPUSource*     d_sources,
    const ApertureSample*     d_grid,
    int                  n_grid,
    float                front_R,
    float                start_z,
    GpuBaffleStack       baffles,
    GpuHurb              hurb,
    unsigned int         hurb_seed_base, // per-ray seed mixed with source-angle bits
    float                sensor_half_w,
    float                sensor_half_h,
    int                  width,
    int                  height,
    float*               d_out_r,
    float*               d_out_g,
    float*               d_out_b,
    float                gain,
    float                ray_weight,
    const GPUSpectralSample* d_spec,
    int                  n_spec,
    const float*         d_ior_at,       // [n_spec][n_surfs] IOR tables
    const float*         d_ior_before,
    float                fmt_w,
    float                fmt_h,
    float                fmt_x0_in_buf,
    float                fmt_y0_in_buf,
    const cudaTextureObject_t* d_aperture_textures,
    const cudaTextureObject_t* d_aperture_sdf,   // HURB image-stop SDF; null = no kick
    const unsigned char* d_pair_active,   // cull mask; null = trace all
    const unsigned char* d_ps_refine,     // per-(pair,src) refine mask; null = all
    ConcParams           conc,            // concentration; rects null = off
    int*                 d_ps_bbox,       // STATS: per-(pair,src) bbox+hits; null = off
    unsigned long long*  d_pair_traces,
    unsigned long long*  d_pair_survived,
    unsigned long long*  d_pair_onsensor)
{
    int ps_idx   = blockIdx.x;
    int pair_idx = ps_idx / n_sources;
    int src_idx  = ps_idx % n_sources;

    int grid_idx = (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;
    if (grid_idx >= n_grid) return;

    // A pair the probe found off-sensor for every source/λ exits before
    // tracing. Null mask (probe pass, cull disabled) traces everything.
    if (d_pair_active != nullptr && d_pair_active[pair_idx] == 0)
        return;

    // Refinement probe: trace only the marginal (pair, source) entries
    // it targets (null mask on every other launch traces everything).
    if (d_ps_refine != nullptr && d_ps_refine[ps_idx] == 0)
        return;

    const GPUPair& pair = d_pairs[pair_idx];

    if (pair.surf_a < 0 || pair.surf_b <= pair.surf_a || pair.surf_b >= n_surfs)
        return;

    const GPUSource&    src = d_sources[src_idx];

    // ---- Sample position + estimator weight -------------------------------
    // Default (full-pupil) path: the pre-masked host grid, weight 1/n_grid.
    // Concentrated path (valid rect): regenerate this thread's sample inside
    // the (pair, source)'s survivor rect with the SAME jitter family, re-test
    // the shared pupil mask in-kernel (a rejected sample contributes zero but
    // stays in the n_R denominator), and scale the weight by |R|/A_ref so the
    // estimator matches the full-pupil one in expectation.
    float u, v;
    float w_ray = ray_weight;
    bool  concentrated = false;
    if (conc.rects != nullptr)
    {
        const float4 rc = conc.rects[ps_idx];
        if (rc.x <= rc.z)                       // valid rect for this (pair, src)
        {
            concentrated = true;
            // Adaptive per-entry sample count and lattice side; null uses the
            // uniform budget.
            int n_r_ps = conc.n_r;
            int nr_lat = conc.nr_lattice;
            if (conc.budgets != nullptr)
            {
                const int2 b = conc.budgets[ps_idx];
                n_r_ps = b.x;
                nr_lat = b.y;
            }
            if (grid_idx >= n_r_ps) return;     // lattice modes use Nr^2 <= n_grid
            namespace asd = aperture_sampler_detail;
            float fu, fv;                       // in [0,1)^2 over the rect
            if (conc.jitter == 2)
            {
                fu = asd::halton2((uint32_t)grid_idx);
                fv = asd::halton3((uint32_t)grid_idx);
            }
            else
            {
                const int   gx = grid_idx % nr_lat;
                const int   gy = grid_idx / nr_lat;
                const int   nn = nr_lat * nr_lat;
                const float ju = (conc.jitter == 1)
                    ? asd::wang_hash((uint32_t)grid_idx + conc.seed_offset) / 4294967296.0f : 0.5f;
                const float jv = (conc.jitter == 1)
                    ? asd::wang_hash((uint32_t)grid_idx + (uint32_t)nn + conc.seed_offset) / 4294967296.0f : 0.5f;
                fu = (gx + ju) / nr_lat;
                fv = (gy + jv) / nr_lat;
            }
            u = rc.x + fu * (rc.z - rc.x);
            v = rc.y + fv * (rc.w - rc.y);
            if (!conc.mask.contains(u, v))      // zero contribution, counted in n_R
                return;
            // Per-ray weight: |R| / (A_ref * n_r_ps). With budgets this is
            // |R| * inv_aref / n_r_ps; the uniform path folds both into area_norm.
            w_ray = (rc.z - rc.x) * (rc.w - rc.y)
                  * (conc.budgets != nullptr ? conc.inv_aref / (float)n_r_ps
                                             : conc.area_norm);
        }
    }
    if (!concentrated)
    {
        const ApertureSample& gs = d_grid[grid_idx];
        u = gs.u;
        v = gs.v;
    }

    float bx = tanf(src.angle_x);
    float by = tanf(src.angle_y);
    Vec3f beam_dir = Vec3f(bx, by, 1.0f).normalized();

    // Follow the beam: the disc is the front aperture's back-projection onto the
    // spawn plane, not an axis-centred copy of it (spawn_plane.h). Zero on axis.
    float sdx, sdy;
    spawn_shift(bx, by, sdx, sdy);

    Ray base_ray;
    base_ray.origin = Vec3f(u * front_R + sdx, v * front_R + sdy, start_z);
    base_ray.dir    = beam_dir;
    base_ray.lambda = 0.0f;

    // Front-of-lens baffles (matte box, hoods, mount): project the entrance ray
    // to each baffle plane and drop it if it lands outside. Achromatic, so tested
    // once before the spectral loop; shared with the diffraction pupil (starburst).
    // n == 0 (the default) is a free zero-trip loop that changes nothing.
    if (baffle_stack_blocks(baffles, base_ray.origin.x, base_ray.origin.y, bx, by))
        return;

    // Per-ray HURB seed base. Source-ANGLE bits (NOT src_idx, which is chunk-local
    // in sourceflare — seeding from the angle makes chunked == unchunked) mixed with
    // the pupil sample (grid_idx), the ghost pair, and the config jitter seed. The
    // spectral loop folds in si so each wavelength draws an independent kick stream.
    [[maybe_unused]] unsigned int hurb_ray_seed = 0u;
    if constexpr (HURB)
        hurb_ray_seed = (unsigned int)__float_as_int(src.angle_x) * 0x9E3779B9u
                      + (unsigned int)__float_as_int(src.angle_y) * 0x85EBCA6Bu
                      + (unsigned int)grid_idx * 0xC2B2AE35u
                      + (unsigned int)pair_idx * 0x27D4EB2Fu
                      + hurb_seed_base;

    [[maybe_unused]] unsigned int loc_traces = 0, loc_surv = 0, loc_onsensor = 0;

    for (int si = 0; si < n_spec; ++si)
    {
        if constexpr (STATS) ++loc_traces;

        const GPUSpectralSample& spec = d_spec[si];

        Ray r    = base_ray;
        r.lambda = spec.lambda;

        unsigned int hurb_seed_l = 0u;
        if constexpr (HURB) {
            hurb_seed_l = hurb_wang_hash(hurb_ray_seed + (unsigned int)si * 0x9E3779B9u);
            // Front-of-lens baffle edges (matte box / hoods): kick the ray before
            // it enters the lens, on a stream independent of the surface-rim kicks.
            // n == 0 (no baffles) or a wide clearance -> no kick (hurb_sigma ~ 0).
            if (baffles.n > 0) {
                unsigned int bstate = hurb_wang_hash(hurb_seed_l ^ 0xBAFF1E5Du);
                Vec3f bn;
                float bd = baffle_stack_edge_distance(baffles, r.origin.x, r.origin.y,
                                                      bx, by, bn);
                hurb_apply_kick(hurb, r.dir, bn, bd, r.lambda, bstate);
            }
        }

        TraceResult res = d_trace_ghost_ray<HURB>(r, d_surfs, n_surfs,
                                                  pair.surf_a, pair.surf_b,
                                                  d_ior_at     + (size_t)si * n_surfs,
                                                  d_ior_before + (size_t)si * n_surfs,
                                                  d_aperture_textures,
                                                  d_aperture_sdf,
                                                  hurb, hurb_seed_l);

        if (res.status != TraceStatus::OK) continue;
        if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;

        if constexpr (STATS)
        {
            // Survived the lens. On-sensor test uses the same frame bounds as
            // the splat below, but independent of the brightness cull so the
            // useful-fraction reflects geometry, not weight.
            ++loc_surv;
            // Format-window map: sensor mm to buffer pixel.
            float spx = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
            float spy = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;
            if (isfinite(spx) && isfinite(spy) &&
                spx >= 0.0f && spx < (float)width &&
                spy >= 0.0f && spy < (float)height)
                ++loc_onsensor;
            continue;  // diagnostic pass never splats
        }

        float base = res.weight * w_ray * gain * pair.area_boost;
        if (base < 1e-14f) continue;

        // Format-window map: sensor mm to buffer pixel.
        float px = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
        float py = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;

        if (!isfinite(px) || !isfinite(py)) continue;

        px = fmaxf(-2.0e9f, fminf(2.0e9f, px));
        py = fmaxf(-2.0e9f, fminf(2.0e9f, py));

        int   x0 = (int)floorf(px - 0.5f);
        int   y0 = (int)floorf(py - 0.5f);
        float fx = (px - 0.5f) - (float)x0;
        float fy = (py - 0.5f) - (float)y0;

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx           * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx           * fy;

        float P_lambda = src.r * spec.cmf_r + src.g * spec.cmf_g + src.b * spec.cmf_b;
        float power    = P_lambda * base;
        float cr       = power * spec.cmf_r;
        float cg       = power * spec.cmf_g;
        float cb       = power * spec.cmf_b;

#define SPLAT_RGB(xi, yi, wt)                                                             \
        if ((xi) >= 0 && (xi) < width && (yi) >= 0 && (yi) < height) {                   \
            int _pix = (yi) * width + (xi);                                               \
            if (fabsf(cr) > 1e-14f) atomicAdd(&d_out_r[_pix], cr * (wt));                \
            if (fabsf(cg) > 1e-14f) atomicAdd(&d_out_g[_pix], cg * (wt));                \
            if (fabsf(cb) > 1e-14f) atomicAdd(&d_out_b[_pix], cb * (wt));                \
        }

        SPLAT_RGB(x0,   y0,   w00)
        SPLAT_RGB(x0+1, y0,   w10)
        SPLAT_RGB(x0,   y0+1, w01)
        SPLAT_RGB(x0+1, y0+1, w11)
#undef SPLAT_RGB
    }

    if constexpr (STATS)
    {
        // One 64-bit atomic per counter per thread (per-pair address → spread
        // across n_pairs slots, so contention stays modest). Threads that took
        // an early return above contributed no traces and skip this cleanly.
        atomicAdd(d_pair_traces   + pair_idx, (unsigned long long)loc_traces);
        atomicAdd(d_pair_survived + pair_idx, (unsigned long long)loc_surv);
        atomicAdd(d_pair_onsensor + pair_idx, (unsigned long long)loc_onsensor);

        // Concentration: fold this sample's pupil position into the
        // (pair, source) survivor bbox iff any wavelength reached the sensor.
        // Layout: [0, 2*n_ps) mins (u,v), [2*n_ps, 4*n_ps) maxs, [4*n_ps, 5*n_ps)
        // hit counts (distinct pupil positions), with n_ps = gridDim.x.
        if (d_ps_bbox != nullptr && loc_onsensor > 0)
        {
            const int n_ps = (int)gridDim.x;
            const int eu = enc_pupil(u);
            const int ev = enc_pupil(v);
            atomicMin(d_ps_bbox + 2 * ps_idx,                  eu);
            atomicMin(d_ps_bbox + 2 * ps_idx + 1,              ev);
            atomicMax(d_ps_bbox + 2 * n_ps + 2 * ps_idx,       eu);
            atomicMax(d_ps_bbox + 2 * n_ps + 2 * ps_idx + 1,   ev);
            atomicAdd(d_ps_bbox + 4 * n_ps + ps_idx,           1);
        }
    }
}

// ===========================================================================
// Persistent GPU scratch + CPU-invariant caches
// ===========================================================================
//
// One process-wide GpuBufferCache, plus the entrance-pupil grid and spectral
// table (+ correction matrix) memoised by their inputs. All guarded by
// g_ghost_mutex, which every launch_ghost_render() call holds for its whole
// body — so the GIL-released bindings can't race the device, and the caches are
// reused across frames and AOV per-pair launches.
namespace {

std::mutex g_ghost_mutex;

// Leaked singleton: never destroyed, so we don't cudaFree after the static CUDA
// runtime has torn down at process exit (a documented Windows hazard).
GpuBufferCache& ghost_cache() {
    static GpuBufferCache* c = new GpuBufferCache();
    return *c;
}

// entrance-pupil grid cache — key captures every input build_aperture_samples()
// reads (sampler params + the stop surface's shape/blades/rotation/aspect).
struct GridKey {
    int   ray_grid = -1, pupil_jitter = 0, jitter_seed = 0, blades_override = 0;
    float rotation_deg = 0.0f;
    int   stop_shape = -1, stop_blades = 0;
    float stop_aspect = 1.0f, stop_rotation_rad = 0.0f;
    // Authored blade shape participates in the grid cache key.
    ApertureShapeParams stop_shape_params;
    bool operator==(const GridKey& o) const {
        return ray_grid == o.ray_grid && pupil_jitter == o.pupil_jitter
            && jitter_seed == o.jitter_seed && blades_override == o.blades_override
            && rotation_deg == o.rotation_deg && stop_shape == o.stop_shape
            && stop_blades == o.stop_blades && stop_aspect == o.stop_aspect
            && stop_rotation_rad == o.stop_rotation_rad
            && stop_shape_params.curvature   == o.stop_shape_params.curvature
            && stop_shape_params.twist       == o.stop_shape_params.twist
            && stop_shape_params.notch       == o.stop_shape_params.notch
            && stop_shape_params.notch_angle == o.stop_shape_params.notch_angle;
    }
};
std::vector<ApertureSample> g_grid;
GridKey g_grid_key;
bool    g_grid_valid = false;
bool    g_grid_dev_uploaded = false;  // cache.d_grid currently mirrors g_grid (skip re-upload)

// spectral table + correction matrix cache — key = the config inputs.
struct SpecKey {
    int   ns = -1, sensor_model = -1, output_cs = -1;
    bool  correct = false;
    float custom[9] = {};
    bool operator==(const SpecKey& o) const {
        if (ns != o.ns || sensor_model != o.sensor_model
            || output_cs != o.output_cs || correct != o.correct) return false;
        for (int i = 0; i < 9; ++i) if (custom[i] != o.custom[i]) return false;
        return true;
    }
};
std::vector<GPUSpectralSample> g_spec;
SpecKey g_spec_key;
float   g_spec_C[3][3];
bool    g_spec_valid = false;
bool    g_spec_do_correct = false;
bool    g_spec_dev_uploaded = false;  // cache.d_spec currently mirrors g_spec (skip re-upload)

// Aperture-SDF bake cache — key captures every input bake_aperture_sdf() reads
// for the image-aperture stop. pixels_ptr/size catch reallocation (Python
// reassigns img.pixels); sample_hash catches in-place edits of the same buffer.
struct SdfKey {
    int    stop_index = -1, width = 0, height = 0;
    float  semi_diameter = 0.0f, aspect = 1.0f;
    const float* pixels_ptr = nullptr;
    std::size_t  pixels_size = 0;
    std::uint64_t sample_hash = 0;
    bool operator==(const SdfKey& o) const {
        return stop_index == o.stop_index && width == o.width && height == o.height
            && semi_diameter == o.semi_diameter && aspect == o.aspect
            && pixels_ptr == o.pixels_ptr && pixels_size == o.pixels_size
            && sample_hash == o.sample_hash;
    }
};
SdfKey g_sdf_key;

// Build the SDF cache key from the lens (stop-only; empty key when no image stop).
static SdfKey build_sdf_key(const OpticalSystem& lens)
{
    SdfKey k;
    const int target = find_sdf_target_surface(lens);
    if (target < 0) return k;
    const Surface&       s   = lens.surfaces[target];
    const ApertureImage& img = lens.aperture_images[target];
    k.stop_index    = target;
    k.width         = img.width;
    k.height        = img.height;
    k.semi_diameter = (s.aperture_semi_diameter > 0.0f) ? s.aperture_semi_diameter
                                                        : img.semi_diameter;
    k.aspect        = s.aperture_aspect;
    k.pixels_ptr    = img.pixels.data();
    k.pixels_size   = img.pixels.size();
    // FNV-1a over <=4096 strided pixels — cheap content fingerprint.
    std::uint64_t h = 1469598103934665603ull;
    const std::size_t n = img.pixels.size();
    const std::size_t stride = n > 4096 ? n / 4096 : 1;
    for (std::size_t p = 0; p < n; p += stride) {
        std::uint32_t bits;
        std::memcpy(&bits, &img.pixels[p], sizeof(bits));
        h = (h ^ bits) * 1099511628211ull;
    }
    k.sample_hash = h;
    return k;
}

// Output-space colour/brightness correction (the spectral reference Gram C),
// applied on-device in place just before the D2H copy. Per launch, so each AOV
// per-pair layer is corrected and the layers still sum consistently (C is
// linear). The __*_rn intrinsics preserve the host's left-to-right arithmetic
// order with separate IEEE round-to-nearest operations.
__global__ void spectral_correct_kernel(
    float* r, float* g, float* b, size_t n,
    float c00, float c01, float c02,
    float c10, float c11, float c12,
    float c20, float c21, float c22)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float rr = r[i], gg = g[i], bb = b[i];
    r[i] = __fadd_rn(__fadd_rn(__fmul_rn(c00, rr), __fmul_rn(c01, gg)), __fmul_rn(c02, bb));
    g[i] = __fadd_rn(__fadd_rn(__fmul_rn(c10, rr), __fmul_rn(c11, gg)), __fmul_rn(c12, bb));
    b[i] = __fadd_rn(__fadd_rn(__fmul_rn(c20, rr), __fmul_rn(c21, gg)), __fmul_rn(c22, bb));
}

} // namespace

// ===========================================================================
// CPU launcher
// ===========================================================================

void launch_ghost_render(
    const OpticalSystem&               lens,
    const std::vector<GhostPair>&   active_pairs,
    const std::vector<float>&       pair_area_boosts,
    const std::vector<FlareSource>& sources,
    float                           sensor_half_w,
    float                           sensor_half_h,
    float*                          out_r,
    float*                          out_g,
    float*                          out_b,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    const FlareConfig&              config,
    std::string*                    out_error,
    GhostRenderStats*               out_stats)
{
    if (active_pairs.empty() || sources.empty()) return;

    const int n_surfs = lens.num_surfaces();
    if (n_surfs <= 0) return;

    // Serialise all GPU access + cache use for the whole body.
    std::lock_guard<std::mutex> gpu_lock(g_ghost_mutex);
    GpuBufferCache& cache = ghost_cache();

    // Instrumentation. All markers below are only assigned when stats is
    // true, so a normal render pays nothing (and produces identical output).
    const bool stats = (out_stats != nullptr);
    using clock = std::chrono::steady_clock;
    clock::time_point t_fn0, t_prep0, t_prep1, t_up0, t_up1, t_k1, t_d1;
    if (stats) t_fn0 = clock::now();

    // GPU availability check.
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

    {
        cudaError_t prev_err = cudaGetLastError();
        if (prev_err != cudaSuccess) {
            fprintf(stderr, "Ghostlight: clearing CUDA sticky error: %s\n",
                    cudaGetErrorString(prev_err));
        }
    }

    const int n_pairs = (int)active_pairs.size();
    const int n_src   = (int)sources.size();

    // ---- Build entrance-pupil grid ----
    // Mask + jitter logic lives in aperture_sampler.h so the PSF renderer
    // sees the same sampling distribution.
    if (stats) t_prep0 = clock::now();
    ApertureSamplerParams sampler_p;
    sampler_p.ray_grid         = config.ray_grid;
    sampler_p.pupil_jitter     = config.pupil_jitter;
    sampler_p.jitter_seed      = config.jitter_seed;
    sampler_p.blades_override  = config.aperture_blades;
    sampler_p.rotation_deg     = config.aperture_rotation;

    // Cache by sampler parameters and every stop field read by
    // build_aperture_samples().
    GridKey gk;
    gk.ray_grid = config.ray_grid; gk.pupil_jitter = config.pupil_jitter;
    gk.jitter_seed = config.jitter_seed; gk.blades_override = config.aperture_blades;
    gk.rotation_deg = config.aperture_rotation;
    for (const auto& s : lens.surfaces)
        if (s.is_stop) {
            gk.stop_shape = (int)s.aperture_shape; gk.stop_blades = s.aperture_blades;
            gk.stop_aspect = s.aperture_aspect;    gk.stop_rotation_rad = s.aperture_rotation_rad;
            gk.stop_shape_params = ApertureShapeParams{
                s.aperture_curvature, s.aperture_twist,
                s.aperture_notch_rad, s.aperture_notch_angle_rad};
            break;
        }
    if (!g_grid_valid || !(gk == g_grid_key)) {
        g_grid = build_aperture_samples(lens, sampler_p);
        g_grid_key = gk;
        g_grid_valid = true;
        g_grid_dev_uploaded = false;   // host grid changed -> device copy is stale
    }
    const std::vector<ApertureSample>& grid_samples = g_grid;

    const int   n_grid     = (int)grid_samples.size();
    if (n_grid == 0) return;
    const float ray_weight = 1.0f / n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - SPAWN_OFFSET;

    // Front-of-lens baffle stack (matte box + config baffles) shared with the
    // diffraction pupil. Empty by default -> every launch below is unaffected
    // (a zero-trip clip).
    GpuBaffleStack baffles = build_gpu_baffles(config.diffraction);

    // hurb.on selects the template specialization with edge diffraction. The
    // seed base folds in the config jitter seed; the kernel mixes it with per-ray
    // source-angle bits (chunk-invariant — see the kernel's hurb_ray_seed).
    const GpuHurb hurb = build_gpu_hurb(config.diffraction);
    const unsigned int hurb_seed_base = (unsigned int)config.jitter_seed * 2654435761u + 1u;

    // ---- Pack GPU pair / source arrays ----
    std::vector<GPUPair> gpu_pairs(n_pairs);
    for (int i = 0; i < n_pairs; ++i)
        gpu_pairs[i] = { active_pairs[i].surf_a,
                         active_pairs[i].surf_b,
                         pair_area_boosts[i] };

    std::vector<GPUSource> gpu_src(n_src);
    for (int i = 0; i < n_src; ++i)
        gpu_src[i] = { sources[i].angle_x, sources[i].angle_y,
                       sources[i].r, sources[i].g, sources[i].b };

    // ---- Build spectral table (cached by its config inputs — the
    //      correction rebuilds a dense reference Gram otherwise) ----
    const int ns = std::max(3, config.spectral_samples);

    SpecKey sk;
    sk.ns = ns; sk.sensor_model = (int)config.sensor_model;
    sk.output_cs = (int)config.output_cs; sk.correct = config.spectral_correction;
    std::copy_n(&config.custom_xyz_to_output[0][0], 9, sk.custom);

    if (!g_spec_valid || !(sk == g_spec_key))
    {
        if (ns == 3)
        {
            // Fixed RGB wavelengths with identity weights.
            g_spec.assign(3, GPUSpectralSample{});
            g_spec[0] = { 650.0f, 1.0f, 0.0f, 0.0f };
            g_spec[1] = { 550.0f, 0.0f, 1.0f, 0.0f };
            g_spec[2] = { 450.0f, 0.0f, 0.0f, 1.0f };
            g_spec_do_correct = false;
        }
        else
        {
            const SensorProfile& prof = get_sensor_profile(config.sensor_model);
            float M_out[3][3];
            resolve_output_matrix(config.output_cs, config.custom_xyz_to_output, M_out);
            g_spec = build_spectral_table(ns, 400.0f, 700.0f, prof, M_out);
            g_spec_do_correct = config.spectral_correction;
            if (g_spec_do_correct)
                compute_spectral_correction(g_spec, 400.0f, 700.0f, prof, M_out, g_spec_C);
        }
        g_spec_key = sk;
        g_spec_valid = true;
        g_spec_dev_uploaded = false;    // host spectral table changed -> device copy is stale
    }
    const std::vector<GPUSpectralSample>& spectral_cpu = g_spec;
    const bool do_spectral_correct = g_spec_do_correct;
    const int  n_spec = ns;

    // ---- per-(λ, surface) IOR tables (host precompute) ----
    // Kills the trace's per-hit Sellmeier eval and the O(N²) backward walk.
    // Layout [λ][surface]; ior_before[s] = last active surface's IOR before s
    // (or air), exactly matching a backward walk. Cheap (~n_surfs·n_spec
    // evals), so recomputed per launch rather than cached.
    std::vector<float> ior_at_host((size_t)n_spec * n_surfs);
    std::vector<float> ior_before_host((size_t)n_spec * n_surfs);
    for (int si = 0; si < n_spec; ++si)
    {
        const float lam = spectral_cpu[si].lambda;
        float* at = &ior_at_host[(size_t)si * n_surfs];
        float* bf = &ior_before_host[(size_t)si * n_surfs];
        float last = 1.0f;
        for (int s = 0; s < n_surfs; ++s)
        {
            at[s] = lens.surfaces[s].ior_at(lam);
            bf[s] = last;
            if (lens.surfaces[s].is_active) last = at[s];
        }
    }
    if (stats) t_prep1 = clock::now();

    const size_t n_px = (size_t)width * height;

    int  grid_blocks = (n_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_dim((unsigned)(n_pairs * n_src), (unsigned)grid_blocks, 1);

    // Compile-time HURB dispatch for every ghost_kernel launch: hurb.on picks the
    // <STATS,true> or the zero-cost <STATS,false> instantiation. All
    // four passes (coarse probe / dense refine / main render / stats) route through
    // this so their cull + concentration decisions match the actual render.
    // GRIDCFG is the per-launch grid; `block`, `hurb` are captured from scope.
#define GHOST_LAUNCH(STATS_V, GRIDCFG, ...)                                      \
    do {                                                                         \
        if (hurb.on) ghost_kernel<STATS_V, true ><<<GRIDCFG, block>>>(__VA_ARGS__); \
        else         ghost_kernel<STATS_V, false><<<GRIDCFG, block>>>(__VA_ARGS__); \
    } while (0)

    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "Ghostlight CUDA error at %s -- %s\n", site, cudaGetErrorString(e));
        if (out_error && out_error->empty())
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "CUDA error at %s -- %s", site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    auto ensure = [&](void*& ptr, size_t& cap, size_t need, const char* tag) -> bool {
        if (need <= cap) return true;
        cudaFree(ptr);  ptr = nullptr;  cap = 0;
        cudaError_t e = cudaMalloc(&ptr, need);
        if (e != cudaSuccess) { report(e, tag); return false; }
        cap = need;
        return true;
    };

    if (stats) t_up0 = clock::now();

    if (!ensure(cache.d_surfs,        cache.surfs_bytes,        n_surfs * sizeof(Surface),                        "d_surfs"        )) return;
    if (!ensure(cache.d_pairs,        cache.pairs_bytes,        n_pairs * sizeof(GPUPair),                        "d_pairs"        )) return;
    if (!ensure(cache.d_src,          cache.src_bytes,          n_src   * sizeof(GPUSource),                      "d_src"          )) return;
    { void* prev = cache.d_grid;
      if (!ensure(cache.d_grid,       cache.grid_bytes,         n_grid  * sizeof(ApertureSample),                      "d_grid"         )) return;
      if (cache.d_grid != prev) g_grid_dev_uploaded = false; }   // realloc -> device copy is stale
    { void* prev = cache.d_spec;
      if (!ensure(cache.d_spec,       cache.spec_bytes,         n_spec  * sizeof(GPUSpectralSample),              "d_spec"         )) return;
      if (cache.d_spec != prev) g_spec_dev_uploaded = false; }   // realloc -> device copy is stale
    if (!ensure(cache.d_ior_at,       cache.ior_at_bytes,       ior_at_host.size()     * sizeof(float),          "d_ior_at"       )) return;
    if (!ensure(cache.d_ior_before,   cache.ior_before_bytes,   ior_before_host.size() * sizeof(float),          "d_ior_before"   )) return;

    if (n_px > cache.out_floats)
    {
        cudaFree(cache.d_out_r);  cache.d_out_r = nullptr;
        cudaFree(cache.d_out_g);  cache.d_out_g = nullptr;
        cudaFree(cache.d_out_b);  cache.d_out_b = nullptr;
        cache.out_floats = 0;

        cudaError_t er = cudaMalloc(&cache.d_out_r, n_px * sizeof(float));
        if (er != cudaSuccess) { report(er, "d_out_r"); return; }
        cudaError_t eg = cudaMalloc(&cache.d_out_g, n_px * sizeof(float));
        if (eg != cudaSuccess) { report(eg, "d_out_g"); return; }
        cudaError_t eb = cudaMalloc(&cache.d_out_b, n_px * sizeof(float));
        if (eb != cudaSuccess) { report(eb, "d_out_b"); return; }
        cache.out_floats = n_px;
    }

#define GPU_CHK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { report(_e, #call); return; } \
    } while(0)

    // Upload any image-aperture textures (no-op when the lens has no image
    // apertures or no pixels have been loaded; kernel sees a nullptr handle
    // table and treats every surface as passthrough).
    if (!upload_aperture_textures(lens, cache.apertures, out_error))
        return;

    // HURB image-stop SDF: bake+upload only when HURB is on (the kernel path is
    // compiled out otherwise, so an SDF would never be sampled). Skip the re-bake
    // when the stop mask is unchanged. release() nulls d_textures,
    // which auto-invalidates the cache.
    if (hurb.on)
    {
        SdfKey k = build_sdf_key(lens);
        if (!(cache.aperture_sdf.d_textures != nullptr && k == g_sdf_key))
        {
            if (!upload_aperture_sdf_textures(lens, cache.aperture_sdf, out_error))
                return;
            g_sdf_key = k;
        }
    }

    // Table-backed coatings: pack tables into the device arena and get a
    // surface copy whose coating pointers reference device memory.  Empty
    // `coated_surfs` = no tables anywhere → upload the originals verbatim.
    std::vector<Surface> coated_surfs;
    if (!upload_coating_tables(lens, cache.coatings, coated_surfs, out_error))
        return;
    const Surface* surfs_src = coated_surfs.empty() ? lens.surfaces.data()
                                                    : coated_surfs.data();

    GPU_CHK(cudaMemcpy(cache.d_surfs,        surfs_src,
                       n_surfs * sizeof(Surface), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_pairs,        gpu_pairs.data(),
                       n_pairs * sizeof(GPUPair), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_src,          gpu_src.data(),
                       n_src   * sizeof(GPUSource), cudaMemcpyHostToDevice));
    // The grid and spectral table change only when
    // their host caches rebuild (g_grid_valid / g_spec_valid); skip the H2D
    // re-upload when the device already mirrors them. AOV per-pair layers,
    // progressive source-flare chunks and non-grid parameter changes reuse
    // the same grid, so this elides the bulk of the per-launch upload. (surfaces /
    // pairs / src / IOR stay unconditional: small, per-call, and d_surfs is tied to
    // the per-call coating arena.)
    if (!g_grid_dev_uploaded) {
        GPU_CHK(cudaMemcpy(cache.d_grid,         grid_samples.data(),
                           n_grid  * sizeof(ApertureSample), cudaMemcpyHostToDevice));
        g_grid_dev_uploaded = true;
    }
    if (!g_spec_dev_uploaded) {
        GPU_CHK(cudaMemcpy(cache.d_spec,         spectral_cpu.data(),
                           n_spec  * sizeof(GPUSpectralSample), cudaMemcpyHostToDevice));
        g_spec_dev_uploaded = true;
    }
    GPU_CHK(cudaMemcpy(cache.d_ior_at,     ior_at_host.data(),
                       ior_at_host.size()     * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_ior_before, ior_before_host.data(),
                       ior_before_host.size() * sizeof(float), cudaMemcpyHostToDevice));

    GPU_CHK(cudaMemset(cache.d_out_r, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_g, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_b, 0, n_px * sizeof(float)));

    if (stats) { cudaDeviceSynchronize(); t_up1 = clock::now(); }

    // ---- coarse GPU probe -> per-pair cull mask + survivor rects ----
    // A pair the probe finds off-sensor for every source + wavelength contributes
    // nothing, so the main kernel skips it (cull_dead_pairs). The same probe also
    // records each (pair, source)'s pupil-space survivor bbox, which the main
    // kernel concentrates its sample budget into (concentrate_samples). Same
    // d_trace_ghost_ray as the render, so decisions are consistent with what
    // would be drawn. Disabled for AOV and the three-sample spectral path.
    bool cull_ready = false;
    bool rect_ready = false;
    const bool want_conc = config.concentrate_samples && n_pairs > 1 && ns != 3;
    const int  n_ps      = n_pairs * n_src;
    std::vector<float4> ps_rects;   // host copy of finalized rects (adaptive budgets)

    // Scoped probe, culling, refinement, and concentration stage.
    auto probe_cull_and_concentrate = [&]()
    {
    if ((config.cull_dead_pairs || want_conc) && n_pairs > 1 && ns != 3)
    {
        // Multi-probe robustness: one coarse probe can miss a thin on-sensor sliver
        // (e.g. an aperture_blades-clipped anamorphic ghost) and false-cull a live
        // pair. Pass 0 probes every pair; a pair it finds off-sensor for all sources
        // + wavelengths is a CULL CANDIDATE. Passes 1..N-1 then re-probe ONLY those
        // candidates (the rest early-exit on the mask) with fresh wang-hash jitter;
        // any candidate that lands on-sensor in a later pass is kept. On-sensor counts
        // accumulate into d_po across passes, so d_po[p] > 0 <=> some pass saw pair p
        // reach the (margin-inflated) sensor, and a pair is culled only when EVERY
        // pass agrees it is dead. Keep-if-any is safe (extra passes only keep MORE
        // alive), and re-probing only candidates keeps the overhead proportional to
        // the potential saving, so culling never goes net-negative on low-cull frames.
        const size_t cnt_b    = (size_t)n_pairs * sizeof(unsigned long long);
        const size_t grid_cap = (size_t)PROBE_GRID * PROBE_GRID * sizeof(ApertureSample);
        const size_t bbox_b   = (size_t)n_ps * 5 * sizeof(int);
        unsigned long long *d_pt = nullptr, *d_ps = nullptr, *d_po = nullptr;
        unsigned char*      d_refine = nullptr;   // per-ps refine mask (transient)
        GPUSpectralSample*  d_spec3  = nullptr;   // reduced-lambda probe tables
        float*              d_iat3   = nullptr;
        float*              d_ibf3   = nullptr;
        if (ensure(cache.d_probe_grid,  cache.probe_grid_bytes, grid_cap, "d_probe_grid")
            && ensure(cache.d_pair_active, cache.pair_active_bytes, (size_t)n_pairs * sizeof(unsigned char), "d_pair_active")
            && (!want_conc || ensure(cache.d_ps_bbox, cache.ps_bbox_bytes, bbox_b, "d_ps_bbox"))
            && cudaMalloc(&d_pt, cnt_b) == cudaSuccess
            && cudaMalloc(&d_ps, cnt_b) == cudaSuccess
            && cudaMalloc(&d_po, cnt_b) == cudaSuccess)
        {
            cudaMemset(d_pt, 0, cnt_b); cudaMemset(d_ps, 0, cnt_b); cudaMemset(d_po, 0, cnt_b);
            int* d_bbox = nullptr;
            if (want_conc)
            {
                // mins span init 0x7F7F7F7F (> any biased-float encode); maxs +
                // hit-count spans init 0. See the kernel's STATS bbox layout.
                d_bbox = static_cast<int*>(cache.d_ps_bbox);
                cudaMemset(d_bbox,              0x7F, (size_t)n_ps * 2 * sizeof(int));
                cudaMemset(d_bbox + 2 * n_ps,   0x00, (size_t)n_ps * 3 * sizeof(int));
            }
            std::vector<unsigned char> active(n_pairs, 1);  // default keep-all on failure
            std::vector<unsigned char> cand(n_pairs, 0);
            bool pass0_ok = false;
            for (int pass = 0; pass < CULL_PROBE_PASSES; ++pass)
            {
                ApertureSamplerParams pp = sampler_p;
                pp.ray_grid     = PROBE_GRID;
                pp.pupil_jitter = 1;                                          // wang-hash: varies with seed
                pp.jitter_seed  = (int)(0x9E3779B9u * (unsigned)(pass + 1));  // fixed & independent per pass
                std::vector<ApertureSample> probe = build_aperture_samples(lens, pp);
                const int n_probe = (int)probe.size();
                if (n_probe <= 0) { if (pass == 0) break; else continue; }
                // Default-stream ordering: this H2D waits for the previous pass's
                // kernel to finish reading d_probe_grid before overwriting it.
                cudaMemcpy(cache.d_probe_grid, probe.data(),
                           (size_t)n_probe * sizeof(ApertureSample), cudaMemcpyHostToDevice);
                // Pass 0 traces every pair; later passes re-probe only the candidate
                // (dead-so-far) pairs held in d_pair_active — the rest early-exit.
                // EXCEPT when concentrating: alive pairs then need every pass's
                // samples folded into their survivor bbox (an underestimated bbox
                // clips a ghost edge), so all passes run
                // unmasked and the small extra probe cost buys bbox completeness.
                const unsigned char* d_mask =
                    (pass == 0 || want_conc)
                        ? nullptr
                        : static_cast<const unsigned char*>(cache.d_pair_active);
                int  pg_blocks = (n_probe + BLOCK_SIZE - 1) / BLOCK_SIZE;
                dim3 pgrid((unsigned)(n_pairs * n_src), (unsigned)pg_blocks, 1);
                GHOST_LAUNCH(true, pgrid,
                    static_cast<Surface*>(cache.d_surfs), n_surfs,
                    static_cast<GPUPair*>(cache.d_pairs), n_src,
                    static_cast<GPUSource*>(cache.d_src),
                    static_cast<ApertureSample*>(cache.d_probe_grid), n_probe,
                    front_R, start_z, baffles, hurb, hurb_seed_base,
                    sensor_half_w * CULL_MARGIN, sensor_half_h * CULL_MARGIN,  // margin
                    width, height,
                    cache.d_out_r, cache.d_out_g, cache.d_out_b,   // untouched by STATS
                    config.flare_gain, ray_weight,
                    static_cast<GPUSpectralSample*>(cache.d_spec), n_spec,
                    static_cast<const float*>(cache.d_ior_at),
                    static_cast<const float*>(cache.d_ior_before),
                    (float)fmt_w, (float)fmt_h, (float)fmt_x0_in_buf, (float)fmt_y0_in_buf,
                    static_cast<cudaTextureObject_t*>(cache.apertures.d_textures),
                    static_cast<const cudaTextureObject_t*>(cache.aperture_sdf.d_textures),
                    d_mask,
                    nullptr,                       // no per-ps refine mask (coarse pass)
                    ConcParams{},                  // probe samples the full pupil
                    d_bbox,                        // bbox recording (null unless conc)
                    d_pt, d_ps, d_po);

                if (pass == 0)
                {
                    // Establish the cull candidates (pairs off-sensor so far). If none,
                    // nothing can be culled, so skip the extra passes — culling then
                    // pays only one probe on frames it cannot speed up (busy images).
                    // (Concentration still needs the remaining passes for bbox
                    // completeness, so it never takes this shortcut.)
                    if (cudaDeviceSynchronize() != cudaSuccess) break;
                    pass0_ok = true;
                    if (!want_conc)
                    {
                        std::vector<unsigned long long> po0(n_pairs), pt0(n_pairs);
                        cudaMemcpy(po0.data(), d_po, cnt_b, cudaMemcpyDeviceToHost);
                        cudaMemcpy(pt0.data(), d_pt, cnt_b, cudaMemcpyDeviceToHost);
                        int n_cand = 0;
                        for (int p = 0; p < n_pairs; ++p)
                            if (pt0[p] > 0 && po0[p] == 0) { cand[p] = 1; ++n_cand; }
                        if (n_cand == 0) break;
                        cudaMemcpy(cache.d_pair_active, cand.data(),
                                   (size_t)n_pairs * sizeof(unsigned char), cudaMemcpyHostToDevice);
                    }
                }
            }

            // ---- Dense refinement probe ------------------------------
            // Re-probe the (pair, source) entries the coarse grid under-resolved
            // (hits < REFINE_HITS_MAX — the marginal band AND the hits==0 dropouts)
            // with a much denser grid at REFINE_SPEC wavelengths, folding the extra
            // survivors into the SAME d_bbox / d_po before the rects and cull mask
            // are finalized below. The blocking hits D2H also serves as the barrier
            // for the (unsynced) later coarse passes, so the dense grid can safely
            // overwrite d_probe_grid.
            if (want_conc && d_bbox != nullptr && pass0_ok)
            {
                std::vector<int> hits_now(n_ps);
                cudaMemcpy(hits_now.data(), d_bbox + (size_t)4 * n_ps,
                           (size_t)n_ps * sizeof(int), cudaMemcpyDeviceToHost);
                std::vector<unsigned char> refine(n_ps, 0);
                int n_refine = 0;
                for (int i = 0; i < n_ps; ++i)
                    if (hits_now[i] < REFINE_HITS_MAX)     // marginal + hits==0 dropouts
                        { refine[i] = 1; ++n_refine; }

                const int rns   = std::min(REFINE_SPEC, n_spec);
                const int li[3] = { 0, n_spec / 2, n_spec - 1 };   // min / mid / max lambda
                if (n_refine > 0 && n_refine <= REFINE_MAX_ENTRIES)
                {
                    ApertureSamplerParams rp = sampler_p;
                    rp.ray_grid     = REFINE_GRID;
                    rp.pupil_jitter = 1;
                    rp.jitter_seed  = (int)0xB5297A4Du;   // distinct from the coarse passes
                    std::vector<ApertureSample> rprobe = build_aperture_samples(lens, rp);
                    const int    n_rprobe  = (int)rprobe.size();
                    const size_t rgrid_cap = (size_t)REFINE_GRID * REFINE_GRID * sizeof(ApertureSample);
                    if (n_rprobe > 0
                        && ensure(cache.d_probe_grid, cache.probe_grid_bytes, rgrid_cap, "d_probe_grid")
                        && cudaMalloc(&d_refine, (size_t)n_ps * sizeof(unsigned char)) == cudaSuccess
                        && cudaMalloc(&d_spec3,  (size_t)rns * sizeof(GPUSpectralSample)) == cudaSuccess
                        && cudaMalloc(&d_iat3,   (size_t)rns * n_surfs * sizeof(float)) == cudaSuccess
                        && cudaMalloc(&d_ibf3,   (size_t)rns * n_surfs * sizeof(float)) == cudaSuccess)
                    {
                        // Reduced-lambda probe tables: min / mid / max of the
                        // spectral set (consistent lambda + IOR triples).
                        std::vector<GPUSpectralSample> spec3(rns);
                        std::vector<float> iat3((size_t)rns * n_surfs), ibf3((size_t)rns * n_surfs);
                        for (int k = 0; k < rns; ++k) {
                            spec3[k] = spectral_cpu[li[k]];
                            std::copy_n(&ior_at_host[(size_t)li[k] * n_surfs],     n_surfs, &iat3[(size_t)k * n_surfs]);
                            std::copy_n(&ior_before_host[(size_t)li[k] * n_surfs], n_surfs, &ibf3[(size_t)k * n_surfs]);
                        }
                        cudaMemcpy(d_refine, refine.data(),
                                   (size_t)n_ps * sizeof(unsigned char), cudaMemcpyHostToDevice);
                        cudaMemcpy(cache.d_probe_grid, rprobe.data(),
                                   (size_t)n_rprobe * sizeof(ApertureSample), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_spec3, spec3.data(), (size_t)rns * sizeof(GPUSpectralSample), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_iat3,  iat3.data(),  (size_t)rns * n_surfs * sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_ibf3,  ibf3.data(),  (size_t)rns * n_surfs * sizeof(float), cudaMemcpyHostToDevice);
                        int  rg_blocks = (n_rprobe + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        dim3 rgrid((unsigned)(n_pairs * n_src), (unsigned)rg_blocks, 1);
                        GHOST_LAUNCH(true, rgrid,
                            static_cast<Surface*>(cache.d_surfs), n_surfs,
                            static_cast<GPUPair*>(cache.d_pairs), n_src,
                            static_cast<GPUSource*>(cache.d_src),
                            static_cast<ApertureSample*>(cache.d_probe_grid), n_rprobe,
                            front_R, start_z, baffles, hurb, hurb_seed_base,
                            sensor_half_w * CULL_MARGIN, sensor_half_h * CULL_MARGIN,
                            width, height,
                            cache.d_out_r, cache.d_out_g, cache.d_out_b,
                            config.flare_gain, ray_weight,
                            d_spec3, rns,
                            d_iat3, d_ibf3,
                            (float)fmt_w, (float)fmt_h, (float)fmt_x0_in_buf, (float)fmt_y0_in_buf,
                            static_cast<cudaTextureObject_t*>(cache.apertures.d_textures),
                            static_cast<const cudaTextureObject_t*>(cache.aperture_sdf.d_textures),
                            nullptr,           // trace all pairs; the per-ps mask gates
                            d_refine,          // ... to the under-resolved entries only
                            ConcParams{},
                            d_bbox,
                            d_pt, d_ps, d_po);
                    }
                }
            }

            if (pass0_ok && cudaDeviceSynchronize() == cudaSuccess)
            {
                std::vector<unsigned long long> onsens(n_pairs);
                cudaMemcpy(onsens.data(), d_po, cnt_b, cudaMemcpyDeviceToHost);
                for (int p = 0; p < n_pairs; ++p)
                    active[p] = (onsens[p] > 0) ? 1 : 0;

                // Finalize the survivor rects: decode the atomic bbox, require
                // CONC_MIN_HITS distinct surviving pupil positions (thin evidence
                // -> full-pupil fallback, never a hard cull), dilate by probe
                // cells to cover the edge-of-region uncertainty, clip to the
                // sampled square. u0 > u1 marks "no rect" to the kernel.
                if (want_conc && d_bbox != nullptr)
                {
                    std::vector<int> bb((size_t)n_ps * 5);
                    cudaMemcpy(bb.data(), d_bbox, bbox_b, cudaMemcpyDeviceToHost);
                    std::vector<float4> rects(n_ps, make_float4(2.f, 2.f, -2.f, -2.f));
                    const float dil = CONC_DILATE_CELLS * (2.0f / (float)PROBE_GRID);
                    for (int i = 0; i < n_ps; ++i)
                    {
                        const int hits = bb[(size_t)4 * n_ps + i];
                        if (hits < CONC_MIN_HITS) continue;
                        float u0 = dec_pupil_host(bb[(size_t)2 * i]);
                        float v0 = dec_pupil_host(bb[(size_t)2 * i + 1]);
                        float u1 = dec_pupil_host(bb[(size_t)2 * n_ps + 2 * i]);
                        float v1 = dec_pupil_host(bb[(size_t)2 * n_ps + 2 * i + 1]);
                        if (!(u0 <= u1 && v0 <= v1)) continue;
                        u0 = std::max(-1.0f, u0 - dil); v0 = std::max(-1.0f, v0 - dil);
                        u1 = std::min( 1.0f, u1 + dil); v1 = std::min( 1.0f, v1 + dil);
                        rects[i] = make_float4(u0, v0, u1, v1);
                    }
                    ps_rects = rects;   // host copy for the adaptive-budget formula

                    // Telemetry: surface the per-(pair,
                    // source) probe hit counts and finalized rects so a diagnostic
                    // can correlate the smooth<->speckle flip with hits crossing
                    // CONC_MIN_HITS. Pure host-side copy of data already computed;
                    // only runs when stats are collected and never touches the
                    // render path.
                    if (out_stats != nullptr)
                    {
                        out_stats->ps_hits.resize(n_ps);
                        out_stats->ps_rect.assign((size_t)n_ps * 4, 0.0f);
                        for (int i = 0; i < n_ps; ++i)
                        {
                            out_stats->ps_hits[i]         = bb[(size_t)4 * n_ps + i];
                            out_stats->ps_rect[4 * i + 0] = rects[i].x;
                            out_stats->ps_rect[4 * i + 1] = rects[i].y;
                            out_stats->ps_rect[4 * i + 2] = rects[i].z;
                            out_stats->ps_rect[4 * i + 3] = rects[i].w;
                        }
                    }

                    if (ensure(cache.d_ps_rect, cache.ps_rect_bytes,
                               (size_t)n_ps * sizeof(float4), "d_ps_rect")
                        && cudaMemcpy(cache.d_ps_rect, rects.data(),
                                      (size_t)n_ps * sizeof(float4),
                                      cudaMemcpyHostToDevice) == cudaSuccess)
                        rect_ready = true;
                }
            }
            cudaMemcpy(cache.d_pair_active, active.data(),
                       (size_t)n_pairs * sizeof(unsigned char), cudaMemcpyHostToDevice);
            cull_ready = config.cull_dead_pairs;
        }
        cudaFree(d_pt); cudaFree(d_ps); cudaFree(d_po); cudaFree(d_refine);
        cudaFree(d_spec3); cudaFree(d_iat3); cudaFree(d_ibf3);
    }
    };
    probe_cull_and_concentrate();

    // Concentration params for the real launch (rects == nullptr when off).
    ConcParams conc;
    if (rect_ready)
    {
        conc.rects       = static_cast<const float4*>(cache.d_ps_rect);
        conc.mask        = resolve_pupil_mask(lens, sampler_p);
        conc.jitter      = sampler_p.pupil_jitter;
        conc.seed_offset = (unsigned)sampler_p.jitter_seed * 1000003u;
        const int nr     = std::max(1, (int)std::floor(std::sqrt((double)n_grid)));
        conc.nr_lattice  = nr;
        conc.n_r         = (sampler_p.pupil_jitter == 2) ? n_grid : nr * nr;
        const float N    = (float)config.ray_grid;
        const float a_ref = 4.0f * (float)n_grid / (N * N);   // mask area, grid-measured
        conc.area_norm   = 1.0f / (a_ref * (float)conc.n_r);

        // Per-(pair, source) adaptive sample budgets.
        // Each concentrated ps draws a sample COUNT scaled to its survivor-rect
        // area: raw = boost * n_grid * |R_i| / A_ref = boost * n_grid * (rect's
        // fraction of the pupil mask). Budgets are floored for visibility,
        // capped at n_grid, and rounded to a 256-thread block. Reweighting by
        // |R| * inv_aref / n_r_ps keeps the estimator unbiased. Invalid rects
        // retain the uniform full-pupil budget.
        if (config.adaptive_sample_budgets && ps_rects.size() == (size_t)n_ps)
        {
            std::vector<int2> budgets(n_ps, make_int2(conc.n_r, conc.nr_lattice));
            const float boost   = std::max(1.0f, config.adaptive_density_boost);
            int   floor_s = std::max(256, config.adaptive_min_samples);
            floor_s = ((floor_s + 255) / 256) * 256;                 // 256-block multiple
            const int floor_nr = std::max(1, (int)std::lround(std::sqrt((double)floor_s)));
            for (int i = 0; i < n_ps; ++i)
            {
                const float4 r = ps_rects[i];
                if (!(r.x <= r.z)) continue;                         // invalid rect -> uniform (unused)
                const float area = (r.z - r.x) * (r.w - r.y);        // |R_i|, grid-consistent
                const float raw  = boost * (float)n_grid * area / a_ref;
                if (sampler_p.pupil_jitter == 2)                     // Halton
                {
                    int b = (int)std::ceil((double)raw);
                    b = ((b + 255) / 256) * 256;                     // whole 256-blocks
                    b = std::max(floor_s, std::min(b, n_grid));
                    budgets[i] = make_int2(b, conc.nr_lattice);
                }
                else                                                 // lattice (jitter 0/1)
                {
                    int nr_ps = (int)std::lround(std::sqrt((double)std::max(0.0f, raw)));
                    nr_ps = std::max(floor_nr, std::min(nr_ps, conc.nr_lattice));
                    budgets[i] = make_int2(nr_ps * nr_ps, nr_ps);
                }
            }
            if (out_stats != nullptr)
            {
                out_stats->ps_budget.resize(n_ps);
                for (int i = 0; i < n_ps; ++i) out_stats->ps_budget[i] = budgets[i].x;
            }
            if (ensure(cache.d_ps_budget, cache.ps_budget_bytes,
                       (size_t)n_ps * sizeof(int2), "d_ps_budget")
                && cudaMemcpy(cache.d_ps_budget, budgets.data(),
                              (size_t)n_ps * sizeof(int2),
                              cudaMemcpyHostToDevice) == cudaSuccess)
            {
                conc.budgets  = static_cast<const int2*>(cache.d_ps_budget);
                conc.inv_aref = 1.0f / a_ref;
            }
        }
    }

    if (config.verbose) {
        printf("Ghostlight CUDA: %d pairs x %d sources x %d samples "
               "-> grid (%u, %u, 1)  block (%d)\n",
               n_pairs, n_src, n_grid,
               grid_dim.x, grid_dim.y, BLOCK_SIZE);

        fflush(stdout);
    }

    GHOST_LAUNCH(false, grid_dim,
        static_cast<Surface*>(cache.d_surfs), n_surfs,
        static_cast<GPUPair*>(cache.d_pairs), n_src,
        static_cast<GPUSource*>(cache.d_src),
        static_cast<ApertureSample*>(cache.d_grid), n_grid,
        front_R, start_z, baffles, hurb, hurb_seed_base,
        sensor_half_w, sensor_half_h,
        width, height,
        cache.d_out_r, cache.d_out_g, cache.d_out_b,
        config.flare_gain, ray_weight,
        static_cast<GPUSpectralSample*>(cache.d_spec), n_spec,
        static_cast<const float*>(cache.d_ior_at),
        static_cast<const float*>(cache.d_ior_before),
        (float)fmt_w, (float)fmt_h,
        (float)fmt_x0_in_buf, (float)fmt_y0_in_buf,
        static_cast<cudaTextureObject_t*>(cache.apertures.d_textures),
        static_cast<const cudaTextureObject_t*>(cache.aperture_sdf.d_textures),
        cull_ready ? static_cast<const unsigned char*>(cache.d_pair_active) : nullptr,
        nullptr,                       // refine mask: main render traces all
        conc, nullptr,
        nullptr, nullptr, nullptr);

    // Apply output-space colour correction before download.
    if (do_spectral_correct) {
        const int CORR_BLK    = 256;
        const int corr_blocks = (int)((n_px + CORR_BLK - 1) / CORR_BLK);
        spectral_correct_kernel<<<corr_blocks, CORR_BLK>>>(
            cache.d_out_r, cache.d_out_g, cache.d_out_b, n_px,
            g_spec_C[0][0], g_spec_C[0][1], g_spec_C[0][2],
            g_spec_C[1][0], g_spec_C[1][1], g_spec_C[1][2],
            g_spec_C[2][0], g_spec_C[2][1], g_spec_C[2][2]);
    }

    GPU_CHK(cudaDeviceSynchronize());
    if (stats) t_k1 = clock::now();

    GPU_CHK(cudaMemcpy(out_r, cache.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_g, cache.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_b, cache.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    if (stats) t_d1 = clock::now();

    // ---- Diagnostic pass: per-pair survivor counters ----
    // Second, un-timed launch of the STATS instantiation. It repeats the trace
    // but never splats, so d_out_* (already copied to the host above) is
    // untouched. Counters live in a small per-pair scratch buffer. Kept in a
    // named closure to match probe_cull_and_concentrate above.
    auto collect_survivor_stats = [&]()
    {
    if (stats)
    {
        auto ms = [](clock::time_point a, clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        out_stats->ms_grid_build = ms(t_prep0, t_prep1);
        out_stats->ms_upload     = ms(t_up0,   t_up1);
        out_stats->ms_kernel     = ms(t_up1,   t_k1);
        out_stats->ms_download   = ms(t_k1,    t_d1);
        out_stats->n_pairs   = n_pairs;
        out_stats->n_sources = n_src;
        out_stats->n_grid    = n_grid;
        out_stats->n_spec    = n_spec;

        const size_t cnt_bytes = (size_t)n_pairs * sizeof(unsigned long long);
        unsigned long long *d_pt = nullptr, *d_ps = nullptr, *d_po = nullptr;
        bool ok = (cudaMalloc(&d_pt, cnt_bytes) == cudaSuccess)
               && (cudaMalloc(&d_ps, cnt_bytes) == cudaSuccess)
               && (cudaMalloc(&d_po, cnt_bytes) == cudaSuccess);
        if (ok)
        {
            cudaMemset(d_pt, 0, cnt_bytes);
            cudaMemset(d_ps, 0, cnt_bytes);
            cudaMemset(d_po, 0, cnt_bytes);

            GHOST_LAUNCH(true, grid_dim,
                static_cast<Surface*>(cache.d_surfs), n_surfs,
                static_cast<GPUPair*>(cache.d_pairs), n_src,
                static_cast<GPUSource*>(cache.d_src),
                static_cast<ApertureSample*>(cache.d_grid), n_grid,
                front_R, start_z, baffles, hurb, hurb_seed_base,
                sensor_half_w, sensor_half_h,
                width, height,
                cache.d_out_r, cache.d_out_g, cache.d_out_b,
                config.flare_gain, ray_weight,
                static_cast<GPUSpectralSample*>(cache.d_spec), n_spec,
                static_cast<const float*>(cache.d_ior_at),
                static_cast<const float*>(cache.d_ior_before),
                (float)fmt_w, (float)fmt_h,
                (float)fmt_x0_in_buf, (float)fmt_y0_in_buf,
                static_cast<cudaTextureObject_t*>(cache.apertures.d_textures),
                static_cast<const cudaTextureObject_t*>(cache.aperture_sdf.d_textures),
                nullptr,               // stats measures ALL pairs (pre-cull picture)
                nullptr,               // ... with no refinement mask
                ConcParams{}, nullptr, // ... on the full pupil (pre-concentration)
                d_pt, d_ps, d_po);

            if (cudaDeviceSynchronize() == cudaSuccess)
            {
                std::vector<unsigned long long> h_pt(n_pairs), h_ps(n_pairs), h_po(n_pairs);
                cudaMemcpy(h_pt.data(), d_pt, cnt_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ps.data(), d_ps, cnt_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_po.data(), d_po, cnt_bytes, cudaMemcpyDeviceToHost);

                out_stats->pair_surf_a.resize(n_pairs);
                out_stats->pair_surf_b.resize(n_pairs);
                for (int i = 0; i < n_pairs; ++i) {
                    out_stats->pair_surf_a[i] = active_pairs[i].surf_a;
                    out_stats->pair_surf_b[i] = active_pairs[i].surf_b;
                }
                unsigned long long tt = 0, ts = 0, to = 0;
                for (int i = 0; i < n_pairs; ++i) { tt += h_pt[i]; ts += h_ps[i]; to += h_po[i]; }
                out_stats->traces_total     = tt;
                out_stats->traces_survived  = ts;
                out_stats->traces_on_sensor = to;
                out_stats->pair_traces    = std::move(h_pt);
                out_stats->pair_survived  = std::move(h_ps);
                out_stats->pair_on_sensor = std::move(h_po);
            }
        }
        cudaFree(d_pt); cudaFree(d_ps); cudaFree(d_po);

        out_stats->ms_total = ms(t_fn0, clock::now());
    }
    };
    collect_survivor_stats();

#undef GPU_CHK
#undef GHOST_LAUNCH
}

// ===========================================================================
// render_ghost_pipeline — shared CPU pipeline used by both flare renderers
// ===========================================================================

bool render_ghost_pipeline(
    const OpticalSystem&               lens,
    const LensCalibration&          calib,
    const std::vector<FlareSource>& sources,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    float*                          ghost_r,
    float*                          ghost_g,
    float*                          ghost_b,
    const FlareConfig&              cfg,
    std::string*                    out_error,
    std::vector<FlareAovLayer>*     aov_out,
    GhostRenderStats*               out_stats)
{
    const int    w   = width;
    const int    h   = height;
    const size_t npx = (size_t)w * h;

    if (sources.empty()) return true;

    // Filter ghost pairs.
    // eff_half_w/h is the effective sensor half-extent: cfg's override when present,
    // otherwise the calibrated value. Used for area-boost (brightness normalisation)
    // *and* as the kernel's position denominator below, so that overriding
    // cfg.sensor_half_w/h crops/zooms the ghost output (smaller sensor → ghosts
    // pushed outward in screen space, larger → pulled inward). Source-to-angle
    // mapping in PointFlareRenderer/SourceFlareRenderer continues to use the
    // calibrated covered field so the ghost pattern doesn't morph under the slider.
    const float eff_half_w = (cfg.sensor_half_w > 0.0f) ? cfg.sensor_half_w : calib.sensor_half_w;
    const float eff_half_h = (cfg.sensor_half_h > 0.0f) ? cfg.sensor_half_h : calib.sensor_half_h;

    std::vector<GhostPair> active_pairs;
    std::vector<float>     area_boosts;
    filter_ghost_pairs(lens, eff_half_w, eff_half_h,
                       cfg, active_pairs, area_boosts);

    // Pair selection runs in lock-step with
    // active_pairs and area_boosts so each surviving pair keeps the area-boost
    // computed for it. Empty filters leave every pre-filtered pair active.
    if (cfg.ghost_filter.mode != GhostFilter::Mode::ALL
        && !cfg.ghost_filter.pairs.empty())
    {
        const auto& want = cfg.ghost_filter.pairs;
        auto in_filter = [&](const GhostPair& p) {
            for (const auto& fp : want)
                if (fp.first == p.surf_a && fp.second == p.surf_b)
                    return true;
            return false;
        };
        const bool include_mode = (cfg.ghost_filter.mode == GhostFilter::Mode::INCLUDE);
        std::vector<GhostPair> kept_pairs;
        std::vector<float>     kept_boosts;
        kept_pairs.reserve(active_pairs.size());
        kept_boosts.reserve(active_pairs.size());
        for (size_t i = 0; i < active_pairs.size(); ++i)
        {
            const bool hit  = in_filter(active_pairs[i]);
            const bool keep = include_mode ? hit : !hit;
            if (keep)
            {
                kept_pairs.push_back(active_pairs[i]);
                kept_boosts.push_back(area_boosts[i]);
            }
        }
        active_pairs.swap(kept_pairs);
        area_boosts.swap(kept_boosts);
    }

    if (cfg.verbose)
        printf("Ghostlight ghosts: %zu active ghost pair(s)\n", active_pairs.size());

    if (active_pairs.empty()) return true;

    // Render active pairs on the GPU.
    if (cfg.aov_mode == GhostAovMode::PER_PAIR && aov_out != nullptr)
    {
        const int n_pairs_active = (int)active_pairs.size();
        const int n_layers = (cfg.aov_max_pairs < 0)
                             ? n_pairs_active
                             : std::min(n_pairs_active, cfg.aov_max_pairs);

        aov_out->resize(n_layers);
        for (int i = 0; i < n_layers; ++i)
        {
            (*aov_out)[i].surf_a = active_pairs[i].surf_a;
            (*aov_out)[i].surf_b = active_pairs[i].surf_b;
            (*aov_out)[i].r.assign(npx, 0.0f);
            (*aov_out)[i].g.assign(npx, 0.0f);
            (*aov_out)[i].b.assign(npx, 0.0f);
        }

        // Render each pair individually into its AOV layer.
        for (int i = 0; i < n_layers; ++i)
        {
            std::vector<GhostPair> single_pair  = { active_pairs[i] };
            std::vector<float>     single_boost = { area_boosts[i] };
            launch_ghost_render(lens, single_pair, single_boost, sources,
                                eff_half_w, eff_half_h,
                                (*aov_out)[i].r.data(),
                                (*aov_out)[i].g.data(),
                                (*aov_out)[i].b.data(),
                                w, h,
                                fmt_w, fmt_h, fmt_x0_in_buf, fmt_y0_in_buf,
                                cfg, out_error);
            if (out_error && !out_error->empty()) return false;
        }

        // Build combined channels from AOV layers.
        for (int i = 0; i < n_layers; ++i)
            for (size_t px = 0; px < npx; ++px)
            {
                ghost_r[px] += (*aov_out)[i].r[px];
                ghost_g[px] += (*aov_out)[i].g[px];
                ghost_b[px] += (*aov_out)[i].b[px];
            }
    }
    else
    {
        launch_ghost_render(lens, active_pairs, area_boosts, sources,
                            eff_half_w, eff_half_h,
                            ghost_r, ghost_g, ghost_b,
                            w, h,
                            fmt_w, fmt_h, fmt_x0_in_buf, fmt_y0_in_buf,
                            cfg, out_error, out_stats);

        if (out_error && !out_error->empty()) return false;
    }

    return true;
}
