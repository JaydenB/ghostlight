// ============================================================================
// ghost.h — Ghost reflection enumeration and rendering
// ============================================================================
#pragma once

#include "optical_system.h"
#include "render_config.h"

#include <vector>

// A ghost bounce pair: surfaces where light reflects instead of transmitting.
struct GhostPair
{
    int surf_a; // first bounce surface (closer to front)
    int surf_b; // second bounce surface (closer to sensor)
};

// A light source that produces flare, located by its angle off the optical
// axis (angle_x, angle_y) — not a screen position — with an HDR colour.
struct FlareSource
{
    float angle_x; // horizontal angle from optical axis (radians)
    float angle_y; // vertical angle from optical axis (radians)
    float r, g, b; // HDR intensity
};

// Enumerate all valid ghost bounce pairs for the lens system.
std::vector<GhostPair> enumerate_ghost_pairs(const OpticalSystem &lens);

// Pre-filter ghost pairs for a lens system.
//
// Removes pairs whose surfaces have no meaningful IOR contrast, then traces a
// single on-axis probe ray to discard pairs below config.min_ghost_intensity.
// Optionally estimates the per-pair ghost spread and returns an area-boost
// factor (see FlareConfig::ghost_normalize).
//
// sensor_half_w/h: half-dimensions of the sensor in mm.
void filter_ghost_pairs(const OpticalSystem&       lens,
                        float                   sensor_half_w,
                        float                   sensor_half_h,
                        const FlareConfig&      config,
                        std::vector<GhostPair>& active_pairs_out,
                        std::vector<float>&     area_boosts_out);
