cmake_minimum_required(VERSION 3.18)
project(ghostlight_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Link CUDART explicitly instead of inheriting CMake's static default.
set(CMAKE_CUDA_RUNTIME_LIBRARY None)

# Default arch list covers Maxwell through Blackwell.
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86;89;90" ..
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 120)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ---------------------------------------------------------------------------
# ghostlight_core static library
# ---------------------------------------------------------------------------
add_library(ghostlight_core STATIC
    src/PointFlareRenderer.cpp
    src/SourceFlareRenderer.cpp
    src/PSFRenderer.cpp
    src/lens_calibration.cpp
    src/spawn_probe.cpp
    src/source_map.cpp
    src/spectral.cpp
    src/aperture_textures.cu
    src/coating_upload.cu
    src/ghost_render.cu
    src/psf_render.cu
    src/starburst_render.cu
    src/mdft_render.cu
    src/veil_render.cu
    src/gate_render.cu
    src/fft_cufft.cu
    src/optical_system.cpp
    src/trace.cpp
    src/ghost.cpp
    src/aperture_sdf.cpp
)

# cuFFT backs the diffraction starburst (fft_cufft.cu). It is the only external
# CUDA library dependency; the abstraction in fft_backend.h keeps it swappable.
# Linked dynamically (cufft64_*.dll) — the toolkit's bin dir must be reachable
# at runtime, which it is wherever the CUDA toolkit that built this is installed.
#
# Pin the toolkit to the one the CUDA compiler comes from, so a machine with
# several CUDA installs links cuFFT from the SAME version as the cudart the
# CUDA language already selected (mixing versions breaks the link).
if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED CMAKE_CUDA_COMPILER)
    get_filename_component(_ghostlight_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDAToolkit_ROOT   "${_ghostlight_cuda_bin}" DIRECTORY)
endif()
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ghostlight_core PUBLIC CUDA::cudart CUDA::cufft)

target_include_directories(ghostlight_core PRIVATE
    src
    third_party    # nlohmann/json.hpp (vendored, MIT)
)

if(WIN32)
    target_compile_definitions(ghostlight_core PRIVATE
        NOMINMAX
        _USE_MATH_DEFINES
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(ghostlight_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )
else()
    target_compile_options(ghostlight_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-deprecated-declarations -Wno-unused-variable>
    )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
        target_compile_options(ghostlight_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-class-memaccess>
        )
    endif()
endif()

# Allow constexpr functions on device without __device__ tags.
target_compile_options(ghostlight_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Fast-math for the GPU translation units, decomposed deliberately:
# relax IEEE division + sqrt (the sqrt/div-bound trace's dominant per-hit cost)
# and flush denormals, but do NOT enable --use_fast_math's transcendental
# intrinsic substitution — the dispersion/coating sinf/cosf/expf stay full
# precision. --prec-div/--prec-sqrt keep IEEE *comparison* semantics, so the
# NaN-rejecting guards (!(h2<=r2), isfinite, t>1e-6f) are unaffected, and every
# guard threshold sits ~32 orders of magnitude above the denormal range so FTZ
# is benign.
target_compile_options(ghostlight_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
)
