// ============================================================================
// coating_upload.cu — pack coating tables into a device blob + patch surfaces
// ============================================================================
#include "coating_upload.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace
{

// Round up to a 16-byte boundary so every table segment is aligned for the
// widest loads the kernel performs.
inline std::size_t align16(std::size_t n)
{
    return (n + 15u) & ~(std::size_t)15u;
}

void report(cudaError_t e, const char* site, std::string* out_error)
{
    fprintf(stderr, "Ghostlight CUDA error at %s -- %s\n",
            site, cudaGetErrorString(e));
    if (out_error && out_error->empty())
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "CUDA error at %s -- %s",
                 site, cudaGetErrorString(e));
        *out_error = buf;
    }
}

} // namespace

void CoatingDeviceArena::release()
{
    if (d_blob)
    {
        cudaFree(d_blob);
        d_blob = nullptr;
    }
    blob_bytes = 0;
}

bool upload_coating_tables(const OpticalSystem& lens,
                           CoatingDeviceArena&  arena,
                           std::vector<Surface>& patched,
                           std::string*          out_error)
{
    patched.clear();

    const std::size_t n_surfs = lens.surfaces.size();
    if (lens.coating_tables.size() < n_surfs)
    {
        // Programmatically built system that never called finalize(); no
        // table data can exist, so the verbatim upload path is correct.
        return true;
    }

    // ---- Pass 1: measure ----------------------------------------------------
    std::size_t need = 0;
    bool any = false;
    for (std::size_t i = 0; i < n_surfs; ++i)
    {
        const CoatingTables& t = lens.coating_tables[i];
        if (!t.has_data()) continue;
        any = true;
        need += align16(t.table.size() * sizeof(CoatingTable1D));
        need += align16(t.sa_wavelengths.size() * sizeof(float));
        need += align16(t.sa_angles.size() * sizeof(float));
        need += align16(t.sa_r.size() * sizeof(float));
    }
    if (!any)
        return true; // patched stays empty; caller uploads surfaces verbatim

    // ---- Grow the device blob (monotonic, like GpuBufferCache) --------------
    if (need > arena.blob_bytes)
    {
        arena.release();
        cudaError_t e = cudaMalloc(&arena.d_blob, need);
        if (e != cudaSuccess)
        {
            report(e, "coating arena cudaMalloc", out_error);
            return false;
        }
        arena.blob_bytes = need;
    }

    // ---- Pass 2: pack host blob + patch a surface copy ----------------------
    std::vector<unsigned char> host_blob(need);
    patched = lens.surfaces;

    char* d_base = (char*)arena.d_blob;
    std::size_t off = 0;

    auto place = [&](const void* src, std::size_t bytes) -> char*
    {
        char* d_ptr = d_base + off;
        std::memcpy(host_blob.data() + off, src, bytes);
        off += align16(bytes);
        return d_ptr;
    };

    for (std::size_t i = 0; i < n_surfs; ++i)
    {
        const CoatingTables& t = lens.coating_tables[i];
        if (!t.has_data()) continue;

        Coating& c = patched[i].coating;

        if (!t.table.empty())
        {
            c.table = (const CoatingTable1D*)place(
                t.table.data(), t.table.size() * sizeof(CoatingTable1D));
            c.table_count = (int)t.table.size();
        }

        const bool sa_ok = !t.sa_r.empty()
                        && !t.sa_wavelengths.empty()
                        && !t.sa_angles.empty()
                        && t.sa_r.size() == t.sa_wavelengths.size()
                                          * t.sa_angles.size();
        if (sa_ok)
        {
            c.sa_wavelengths = (const float*)place(
                t.sa_wavelengths.data(),
                t.sa_wavelengths.size() * sizeof(float));
            c.sa_angles = (const float*)place(
                t.sa_angles.data(), t.sa_angles.size() * sizeof(float));
            c.sa_r = (const float*)place(
                t.sa_r.data(), t.sa_r.size() * sizeof(float));
            c.sa_n_wavelengths = (int)t.sa_wavelengths.size();
            c.sa_n_angles      = (int)t.sa_angles.size();
        }
    }

    cudaError_t e = cudaMemcpy(arena.d_blob, host_blob.data(), need,
                               cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        report(e, "coating arena cudaMemcpy", out_error);
        patched.clear();
        return false;
    }

    return true;
}
