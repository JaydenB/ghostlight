// ============================================================================
// ghost.cpp — Ghost pair enumeration and pre-filtering
// ============================================================================

#include "ghost.h"
#include "spawn_plane.h"   // SPAWN_OFFSET (these probes are axial: no shift)
#include "trace.h"

#include <cmath>
#include <algorithm>
#include <vector>

// Standard RGB probe wavelengths used for intensity estimation.
static constexpr float PROBE_WAVELENGTHS[3] = {650.0f, 550.0f, 450.0f};

// ---------------------------------------------------------------------------
std::vector<GhostPair> enumerate_ghost_pairs(const OpticalSystem &lens)
{
    std::vector<GhostPair> pairs;
    int N = lens.num_surfaces();
    for (int a = 0; a < N; ++a)
    {
        if (!lens.surfaces[a].is_active) continue;
        for (int b = a + 1; b < N; ++b)
        {
            if (!lens.surfaces[b].is_active) continue;
            pairs.push_back({a, b});
        }
    }
    return pairs;
}

// ---------------------------------------------------------------------------
static float estimate_ghost_intensity(const OpticalSystem &lens,
                                      int bounce_a, int bounce_b)
{
    Ray ray;
    ray.origin = Vec3f(0, 0, lens.surfaces[0].z - SPAWN_OFFSET);
    ray.dir    = Vec3f(0, 0, 1);

    float total = 0;
    for (int ch = 0; ch < 3; ++ch)
    {
        ray.lambda = PROBE_WAVELENGTHS[ch];
        TraceResult r = trace_ghost_ray(ray, lens, bounce_a, bounce_b);
        if (r.status == TraceStatus::OK)
            total += r.weight;
    }
    return total / 3.0f;
}

// ---------------------------------------------------------------------------
static float estimate_ghost_spread(const OpticalSystem &lens,
                                   int bounce_a, int bounce_b,
                                   float sensor_half_w, float sensor_half_h,
                                   const FlareConfig& config)
{
    constexpr int G = 8;
    float front_R = lens.surfaces[0].semi_aperture;
    float start_z = lens.surfaces[0].z - SPAWN_OFFSET;

    float min_x = 1e30f, max_x = -1e30f;
    float min_y = 1e30f, max_y = -1e30f;
    int valid_count = 0;

    for (int gy = 0; gy < G; ++gy)
    {
        for (int gx = 0; gx < G; ++gx)
        {
            float u = ((gx + 0.5f) / G) * 2.0f - 1.0f;
            float v = ((gy + 0.5f) / G) * 2.0f - 1.0f;
            if (u * u + v * v > 1.0f)
                continue;

            Ray ray;
            ray.origin = Vec3f(u * front_R, v * front_R, start_z);
            ray.dir    = Vec3f(0, 0, 1);
            ray.lambda = PROBE_WAVELENGTHS[1]; // green

            TraceResult res = trace_ghost_ray(ray, lens, bounce_a, bounce_b);
            if (res.status != TraceStatus::OK)
                continue;

            min_x = std::min(min_x, res.position.x);
            max_x = std::max(max_x, res.position.x);
            min_y = std::min(min_y, res.position.y);
            max_y = std::max(max_y, res.position.y);
            ++valid_count;
        }
    }

    if (valid_count < 2)
        return 1.0f;

    float ghost_w = std::max(max_x - min_x, 0.01f);
    float ghost_h = std::max(max_y - min_y, 0.01f);
    float sensor_w = 2.0f * sensor_half_w;
    float sensor_h = 2.0f * sensor_half_h;

    float area_ratio = (ghost_w * ghost_h) / (sensor_w * sensor_h);
    return std::clamp(area_ratio, 1.0f, config.max_area_boost);
}

// ---------------------------------------------------------------------------
void filter_ghost_pairs(const OpticalSystem&       lens,
                        float                   sensor_half_w,
                        float                   sensor_half_h,
                        const FlareConfig&      config,
                        std::vector<GhostPair>& active_pairs_out,
                        std::vector<float>&     area_boosts_out)
{
    auto pairs = enumerate_ghost_pairs(lens);
    active_pairs_out.clear();
    area_boosts_out.clear();

    for (auto& p : pairs)
    {
        float ior_before_a = lens.ior_before(p.surf_a);
        float ior_after_a  = lens.surfaces[p.surf_a].ior;
        float ior_before_b = lens.ior_before(p.surf_b);
        float ior_after_b  = lens.surfaces[p.surf_b].ior;

        if (std::abs(ior_before_a - ior_after_a) < 0.001f ||
            std::abs(ior_before_b - ior_after_b) < 0.001f)
            continue;

        float est = estimate_ghost_intensity(lens, p.surf_a, p.surf_b);
        if (est < config.min_ghost_intensity)
            continue;

        float boost = 1.0f;
        if (config.ghost_normalize)
            boost = estimate_ghost_spread(lens, p.surf_a, p.surf_b,
                                          sensor_half_w, sensor_half_h, config);
        active_pairs_out.push_back(p);
        area_boosts_out.push_back(boost);
    }
}
