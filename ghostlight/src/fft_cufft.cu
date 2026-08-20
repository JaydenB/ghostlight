// ============================================================================
// fft_cufft.cu — cuFFT implementation of the fft_backend.h interface.
//
// This is the sole cuFFT-dependent translation unit (see fft_backend.h).  It
// keeps one cached plan per transform size, guarded by a mutex so the GIL-
// released Python bindings cannot race plan creation.
// ============================================================================

#include "fft_backend.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <mutex>
#include <vector>

namespace {

std::mutex g_fft_mutex;

struct CachedPlan {
    int          n    = 0;
    cufftHandle  plan = 0;
    bool         valid = false;
};

// A handful of distinct FFT sizes at most (grid presets), so a linear vector
// cache is simpler and faster than a map. Leaked at process exit like the
// renderer's device caches — cuFFT plan destruction after CUDA teardown is the
// same Windows hazard the ghost cache documents.
std::vector<CachedPlan>& plan_cache()
{
    static std::vector<CachedPlan>* c = new std::vector<CachedPlan>();
    return *c;
}

// Fetch (or lazily create) the cached 2D C2C plan for an N x N transform.
// Returns 0 on failure.
cufftHandle get_plan(int n, const char** err)
{
    for (const CachedPlan& p : plan_cache())
        if (p.valid && p.n == n)
            return p.plan;

    cufftHandle plan = 0;
    cufftResult r = cufftPlan2d(&plan, n, n, CUFFT_C2C);
    if (r != CUFFT_SUCCESS) {
        if (err) *err = "cufftPlan2d failed";
        return 0;
    }
    plan_cache().push_back({n, plan, true});
    return plan;
}

} // namespace

bool fft2d_c2c_forward_inplace(void* d_data, int N, const char** err)
{
    if (err) *err = nullptr;
    if (d_data == nullptr || N <= 0) {
        if (err) *err = "fft2d: invalid arguments";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_fft_mutex);

    cufftHandle plan = get_plan(N, err);
    if (plan == 0) return false;

    cufftResult r = cufftExecC2C(plan,
                                 reinterpret_cast<cufftComplex*>(d_data),
                                 reinterpret_cast<cufftComplex*>(d_data),
                                 CUFFT_FORWARD);
    if (r != CUFFT_SUCCESS) {
        if (err) *err = "cufftExecC2C failed";
        return false;
    }

    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
        if (err) *err = cudaGetErrorString(ce);
        return false;
    }
    return true;
}

void fft_backend_release()
{
    std::lock_guard<std::mutex> lock(g_fft_mutex);
    for (CachedPlan& p : plan_cache())
        if (p.valid) { cufftDestroy(p.plan); p.valid = false; }
    plan_cache().clear();
}
