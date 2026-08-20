#pragma once
// Zero-copy helpers: move FlareBuffers / PSFOutput std::vectors into
// Python numpy arrays via a py::capsule that owns the heap allocation.

#include <cstdio>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector <-> Python list (stats per-pair arrays)

#include "flare_buffers.h"
#include "PSFRenderer.h"

namespace py = pybind11;

// Move a std::vector<float> into a 2-D numpy array (H, W).
// The capsule owns the heap memory for the lifetime of the returned array.
inline py::array_t<float>
view_vector_moved(std::vector<float>&& v, int h, int w)
{
    auto* holder = new std::vector<float>(std::move(v));
    py::capsule free_holder(holder,
        [](void* p) { delete static_cast<std::vector<float>*>(p); });
    return py::array_t<float>(
        {(py::ssize_t)h, (py::ssize_t)w},
        {(py::ssize_t)(sizeof(float) * w), (py::ssize_t)sizeof(float)},
        holder->data(), free_holder);
}

// Move a std::vector<float> into a 3-D numpy array (D, H, W).
inline py::array_t<float>
view_vector_moved_3d(std::vector<float>&& v, int d, int h, int w)
{
    auto* holder = new std::vector<float>(std::move(v));
    py::capsule free_holder(holder,
        [](void* p) { delete static_cast<std::vector<float>*>(p); });
    return py::array_t<float>(
        {(py::ssize_t)d, (py::ssize_t)h, (py::ssize_t)w},
        {(py::ssize_t)(sizeof(float) * h * w),
         (py::ssize_t)(sizeof(float) * w),
         (py::ssize_t)sizeof(float)},
        holder->data(), free_holder);
}

// Convert FlareBuffers to a dict of numpy arrays (zero-copy for all channels).
inline py::dict flare_buffers_to_dict(FlareBuffers&& b)
{
    py::dict d;
    d["width"]   = b.width;
    d["height"]  = b.height;
    d["ghost_r"] = view_vector_moved(std::move(b.ghost_r), b.height, b.width);
    d["ghost_g"] = view_vector_moved(std::move(b.ghost_g), b.height, b.width);
    d["ghost_b"] = view_vector_moved(std::move(b.ghost_b), b.height, b.width);
    if (!b.starburst_r.empty()) {
        d["starburst_r"] = view_vector_moved(std::move(b.starburst_r), b.height, b.width);
        d["starburst_g"] = view_vector_moved(std::move(b.starburst_g), b.height, b.width);
        d["starburst_b"] = view_vector_moved(std::move(b.starburst_b), b.height, b.width);
    }
    if (!b.veil_r.empty()) {
        d["veil_r"] = view_vector_moved(std::move(b.veil_r), b.height, b.width);
        d["veil_g"] = view_vector_moved(std::move(b.veil_g), b.height, b.width);
        d["veil_b"] = view_vector_moved(std::move(b.veil_b), b.height, b.width);
    }
    if (!b.gate_r.empty()) {
        d["gate_r"] = view_vector_moved(std::move(b.gate_r), b.height, b.width);
        d["gate_g"] = view_vector_moved(std::move(b.gate_g), b.height, b.width);
        d["gate_b"] = view_vector_moved(std::move(b.gate_b), b.height, b.width);
    }
    for (auto& layer : b.aov_layers) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "ghost_s%d_s%d", layer.surf_a, layer.surf_b);
        std::string r_key = std::string(prefix) + "_r";
        std::string g_key = std::string(prefix) + "_g";
        std::string b_key = std::string(prefix) + "_b";
        d[r_key.c_str()] = view_vector_moved(std::move(layer.r), b.height, b.width);
        d[g_key.c_str()] = view_vector_moved(std::move(layer.g), b.height, b.width);
        d[b_key.c_str()] = view_vector_moved(std::move(layer.b), b.height, b.width);
    }
    if (b.has_stats) {
        const GhostRenderStats& st = b.stats;
        py::dict s;
        s["ms_grid_build"]    = st.ms_grid_build;
        s["ms_upload"]        = st.ms_upload;
        s["ms_kernel"]        = st.ms_kernel;
        s["ms_download"]      = st.ms_download;
        s["ms_total"]         = st.ms_total;
        s["n_pairs"]          = st.n_pairs;
        s["n_sources"]        = st.n_sources;
        s["n_grid"]           = st.n_grid;
        s["n_spec"]           = st.n_spec;
        s["traces_total"]     = st.traces_total;
        s["traces_survived"]  = st.traces_survived;
        s["traces_on_sensor"] = st.traces_on_sensor;
        s["pair_surf_a"]      = st.pair_surf_a;
        s["pair_surf_b"]      = st.pair_surf_b;
        s["pair_traces"]      = st.pair_traces;
        s["pair_survived"]    = st.pair_survived;
        s["pair_on_sensor"]   = st.pair_on_sensor;
        s["ps_hits"]          = st.ps_hits;   // concentration telemetry
        s["ps_rect"]          = st.ps_rect;   // concentration survivor rects
        s["ps_budget"]        = st.ps_budget; // adaptive sample budgets
        d["stats"] = s;
    }
    return d;
}

// Move a std::vector<float> into a 1-D numpy array of length n.
inline py::array_t<float>
view_vector_moved_1d(std::vector<float>&& v, int n)
{
    auto* holder = new std::vector<float>(std::move(v));
    py::capsule free_holder(holder,
        [](void* p) { delete static_cast<std::vector<float>*>(p); });
    return py::array_t<float>(
        {(py::ssize_t)n},
        {(py::ssize_t)sizeof(float)},
        holder->data(), free_holder);
}

// Move a std::vector<uint8_t> into a 1-D numpy array of length n.
inline py::array_t<uint8_t>
view_vector_moved_1d_u8(std::vector<uint8_t>&& v, int n)
{
    auto* holder = new std::vector<uint8_t>(std::move(v));
    py::capsule free_holder(holder,
        [](void* p) { delete static_cast<std::vector<uint8_t>*>(p); });
    return py::array_t<uint8_t>(
        {(py::ssize_t)n},
        {(py::ssize_t)sizeof(uint8_t)},
        holder->data(), free_holder);
}

// Convert PSFOutput to a dict of numpy arrays.
// r/g/b are composite (composite_h, composite_w) — slice per tile in Python:
//     tile = r[gy*tile_h:(gy+1)*tile_h, gx*tile_w:(gx+1)*tile_w]
inline py::dict psf_output_to_dict(PSFOutput&& o)
{
    const int n_sources = (int)o.chief_x_mm.size();
    py::dict d;
    d["composite_w"]    = o.composite_w;
    d["composite_h"]    = o.composite_h;
    d["tile_w"]         = o.tile_w;
    d["tile_h"]         = o.tile_h;
    d["grid_nx"]        = o.grid_nx;
    d["grid_ny"]        = o.grid_ny;
    d["tile_extent_mm"] = o.tile_extent_mm;
    d["r"] = view_vector_moved(std::move(o.out_r), o.composite_h, o.composite_w);
    d["g"] = view_vector_moved(std::move(o.out_g), o.composite_h, o.composite_w);
    d["b"] = view_vector_moved(std::move(o.out_b), o.composite_h, o.composite_w);
    d["chief_x_mm"] = view_vector_moved_1d(std::move(o.chief_x_mm), n_sources);
    d["chief_y_mm"] = view_vector_moved_1d(std::move(o.chief_y_mm), n_sources);
    // Per-source cell report, populated in both centering modes.
    d["status"]          = view_vector_moved_1d_u8(std::move(o.status), n_sources);
    d["pupil_fraction"]  = view_vector_moved_1d(std::move(o.pupil_fraction), n_sources);
    d["aim_residual_mm"] = view_vector_moved_1d(std::move(o.aim_residual_mm), n_sources);
    return d;
}
