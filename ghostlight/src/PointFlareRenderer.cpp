// ============================================================================
// PointFlareRenderer.cpp — Single point source ghost renderer
//
// A point source is the degenerate extended source: one collimated direction
// with full weight.  All the real work lives in SourceFlareRenderer.cpp.
// ============================================================================

#include "PointFlareRenderer.h"
#include "SourceFlareRenderer.h"

bool render_point_flare(int width, int height,
                        const OpticalSystem&       lens,
                        const LensCalibration&  calib,
                        const PointFlareConfig& cfg,
                        FlareBuffers&           out)
{
    static const float kCenterOffset[3] = {0.0f, 0.0f, 1.0f};
    return render_source_flare(width, height, kCenterOffset, 1,
                               lens, calib, cfg, out);
}
