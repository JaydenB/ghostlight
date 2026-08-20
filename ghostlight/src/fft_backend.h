// ============================================================================
// fft_backend.h — Minimal 2D complex FFT abstraction.
//
// The ONLY translation unit that includes cuFFT is fft_cufft.cu.  Everything
// else (the starburst renderer, its callers) talks to the FFT through this
// header, which exposes no cuFFT types.  To drop the cuFFT dependency — e.g.
// for a build that must not link a proprietary library — replace fft_cufft.cu
// with a self-contained radix-2 kernel implementing this same interface; no
// other file changes.
// ============================================================================
#pragma once

#include <cstddef>

// Forward, in-place, un-normalised 2D complex-to-complex FFT of an N x N array.
//
// d_data points to N*N interleaved single-precision complex values (re, im) in
// DEVICE memory, row-major.  N must be a power of two.  The transform follows
// the usual numpy/cuFFT convention (no normalisation; the caller divides by
// N*N when an inverse round-trip needs it — the starburst never does, it only
// takes |FFT|^2).
//
// Plans are cached internally by N, so repeated same-size calls reuse one plan.
// Returns false and sets *err (a static string, never freed) on failure.
bool fft2d_c2c_forward_inplace(void* d_data, int N, const char** err);

// Release any cached FFT plans and backend state.  Safe to call multiple times;
// safe to never call (the process teardown reclaims device memory).
void fft_backend_release();
