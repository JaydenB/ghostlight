// ============================================================================
// aperture_textures.cu — APERTURE_IMAGE bitmap → cudaTextureObject upload.
//
// See aperture_textures.h for the API and lifetime contract.
// ============================================================================

#include "aperture_textures.h"
#include "aperture_sdf.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

void ApertureTexturePack::release()
{
    // Destroy texture objects first; their backing cudaArrays may otherwise
    // be referenced by the object on free.
    for (size_t i = 0; i < textures.size(); ++i)
    {
        cudaTextureObject_t tex = (cudaTextureObject_t)textures[i];
        if (tex != 0) cudaDestroyTextureObject(tex);
    }
    for (size_t i = 0; i < arrays.size(); ++i)
    {
        cudaArray_t arr = reinterpret_cast<cudaArray_t>(arrays[i]);
        if (arr != nullptr) cudaFreeArray(arr);
    }
    textures.clear();
    arrays.clear();
    cudaFree(d_textures);
    d_textures       = nullptr;
}

bool upload_aperture_textures(const OpticalSystem& lens,
                              ApertureTexturePack& pack,
                              std::string*         out_error)
{
    // Re-upload from scratch: pixel data may have changed and ApertureImage
    // carries no version key the caller could check cheaply.
    pack.release();

    const int n_surfs = (int)lens.surfaces.size();
    if (n_surfs <= 0) return true;

    // Skip the whole upload when no surface has usable bitmap data — kernel
    // sees nullptr handle table and treats every surface as pass-through.
    bool any = false;
    for (int i = 0; i < n_surfs; ++i)
    {
        if (lens.surfaces[i].aperture_shape == APERTURE_IMAGE
            && i < (int)lens.aperture_images.size()
            && !lens.aperture_images[i].pixels.empty()
            && lens.aperture_images[i].width  > 0
            && lens.aperture_images[i].height > 0
            && lens.aperture_images[i].pixels.size()
               == (size_t)lens.aperture_images[i].width
                * (size_t)lens.aperture_images[i].height)
        {
            any = true;
            break;
        }
    }
    if (!any) return true;

    pack.textures.assign((size_t)n_surfs, 0);
    pack.arrays.assign((size_t)n_surfs, 0);

    cudaChannelFormatDesc chan = cudaCreateChannelDesc<float>();

    for (int i = 0; i < n_surfs; ++i)
    {
        const Surface& s = lens.surfaces[i];
        if (s.aperture_shape != APERTURE_IMAGE) continue;
        if (i >= (int)lens.aperture_images.size()) continue;
        const ApertureImage& img = lens.aperture_images[i];
        if (img.pixels.empty() || img.width <= 0 || img.height <= 0) continue;
        if (img.pixels.size() != (size_t)img.width * (size_t)img.height)
        {
            if (out_error)
                *out_error = "upload_aperture_textures: image dimensions do not match pixel data";
            pack.release();
            return false;
        }

        cudaArray_t arr = nullptr;
        cudaError_t e = cudaMallocArray(&arr, &chan, img.width, img.height);
        if (e != cudaSuccess)
        {
            if (out_error)
                *out_error = std::string("upload_aperture_textures: cudaMallocArray failed -- ")
                             + cudaGetErrorString(e);
            pack.release();
            return false;
        }

        e = cudaMemcpy2DToArray(
            arr, 0, 0,
            img.pixels.data(),
            img.width * sizeof(float),
            img.width * sizeof(float),
            (size_t)img.height,
            cudaMemcpyHostToDevice);
        if (e != cudaSuccess)
        {
            if (out_error)
                *out_error = std::string("upload_aperture_textures: cudaMemcpy2DToArray failed -- ")
                             + cudaGetErrorString(e);
            cudaFreeArray(arr);
            pack.release();
            return false;
        }

        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = arr;

        cudaTextureDesc tdesc{};
        tdesc.addressMode[0]   = cudaAddressModeClamp;
        tdesc.addressMode[1]   = cudaAddressModeClamp;
        tdesc.filterMode       = cudaFilterModeLinear;
        tdesc.readMode         = cudaReadModeElementType;
        tdesc.normalizedCoords = 1;

        cudaTextureObject_t tex = 0;
        e = cudaCreateTextureObject(&tex, &res, &tdesc, nullptr);
        if (e != cudaSuccess)
        {
            if (out_error)
                *out_error = std::string("upload_aperture_textures: cudaCreateTextureObject failed -- ")
                             + cudaGetErrorString(e);
            cudaFreeArray(arr);
            pack.release();
            return false;
        }

        pack.textures[i] = (std::uint64_t)tex;
        pack.arrays[i]   = reinterpret_cast<std::uint64_t>(arr);
    }

    // Push the per-surface handle table to the device.
    const size_t bytes = (size_t)n_surfs * sizeof(cudaTextureObject_t);
    cudaError_t e = cudaMalloc(&pack.d_textures, bytes);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_textures: cudaMalloc d_textures failed -- ")
                         + cudaGetErrorString(e);
        pack.release();
        return false;
    }

    // Repack the host vector as packed cudaTextureObject_t[] for the upload.
    // cudaTextureObject_t is a 64-bit handle, so the bit-pattern is preserved.
    std::vector<cudaTextureObject_t> packed((size_t)n_surfs);
    for (int i = 0; i < n_surfs; ++i)
        packed[i] = (cudaTextureObject_t)pack.textures[i];

    e = cudaMemcpy(pack.d_textures, packed.data(), bytes,
                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_textures: cudaMemcpy d_textures failed -- ")
                         + cudaGetErrorString(e);
        pack.release();
        return false;
    }

    return true;
}

bool upload_aperture_sdf_textures(const OpticalSystem& lens,
                                  ApertureTexturePack& pack,
                                  std::string*         out_error)
{
    pack.release();

    const int n_surfs = (int)lens.surfaces.size();
    if (n_surfs <= 0) return true;

    const int target = find_sdf_target_surface(lens);
    if (target < 0) return true;   // no image-aperture stop -> nothing to upload

    const Surface&       s   = lens.surfaces[target];
    const ApertureImage& img = lens.aperture_images[target];

    // Match the trace UV scale: aperture_semi_diameter with an ApertureImage
    // fallback (mirrors the starburst dirt fold), aspect from the surface.
    float sd = s.aperture_semi_diameter;
    if (sd <= 0.0f) sd = img.semi_diameter;

    ApertureSdfBake bake;
    if (!bake_aperture_sdf(img, sd, s.aperture_aspect, bake))
        return true;               // degenerate image -> treat as no SDF

    pack.textures.assign((size_t)n_surfs, 0);
    pack.arrays.assign((size_t)n_surfs, 0);

    cudaChannelFormatDesc chan = cudaCreateChannelDesc<float4>();

    cudaArray_t arr = nullptr;
    cudaError_t e = cudaMallocArray(&arr, &chan, bake.width, bake.height);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_sdf_textures: cudaMallocArray failed -- ")
                         + cudaGetErrorString(e);
        pack.release();
        return false;
    }

    e = cudaMemcpy2DToArray(
        arr, 0, 0,
        bake.texels.data(),
        (size_t)bake.width * sizeof(float4),
        (size_t)bake.width * sizeof(float4),
        (size_t)bake.height,
        cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_sdf_textures: cudaMemcpy2DToArray failed -- ")
                         + cudaGetErrorString(e);
        cudaFreeArray(arr);
        pack.release();
        return false;
    }

    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;

    cudaTextureDesc tdesc{};
    tdesc.addressMode[0]   = cudaAddressModeClamp;
    tdesc.addressMode[1]   = cudaAddressModeClamp;
    tdesc.filterMode       = cudaFilterModeLinear;
    tdesc.readMode         = cudaReadModeElementType;
    tdesc.normalizedCoords = 1;

    cudaTextureObject_t tex = 0;
    e = cudaCreateTextureObject(&tex, &res, &tdesc, nullptr);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_sdf_textures: cudaCreateTextureObject failed -- ")
                         + cudaGetErrorString(e);
        cudaFreeArray(arr);
        pack.release();
        return false;
    }

    pack.textures[target] = (std::uint64_t)tex;
    pack.arrays[target]   = reinterpret_cast<std::uint64_t>(arr);

    const size_t bytes = (size_t)n_surfs * sizeof(cudaTextureObject_t);
    e = cudaMalloc(&pack.d_textures, bytes);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_sdf_textures: cudaMalloc d_textures failed -- ")
                         + cudaGetErrorString(e);
        pack.release();
        return false;
    }

    std::vector<cudaTextureObject_t> packed((size_t)n_surfs, 0);
    packed[target] = tex;

    e = cudaMemcpy(pack.d_textures, packed.data(), bytes,
                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        if (out_error)
            *out_error = std::string("upload_aperture_sdf_textures: cudaMemcpy d_textures failed -- ")
                         + cudaGetErrorString(e);
        pack.release();
        return false;
    }

    return true;
}
