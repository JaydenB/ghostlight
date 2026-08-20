// ============================================================================
// trace.cpp — CPU sequential lens ray tracer
//
// All geometry and optics math comes from trace_core.h (the shared
// __host__ __device__ functions).  No inline math lives here.
//
// The four public trace functions (fast/diagnostic × primary/ghost) share
// trace_surface_step() for intersection, masking, weighting, attenuation, and
// refraction/reflection. Variation between callers is passed in: the far-side index n2
// (ior_at forward vs ior_before backward), whether the surface reflects, the
// post-reflection medium, and a recorder policy (a no-op on the fast paths, an
// event sink on the diagnostic overloads — the branch compiles away when unused).
// ============================================================================

#include "trace.h"
#include "trace_core.h"

#include <cmath>

namespace
{

// Nearest-neighbour bitmap sampling masks APERTURE_IMAGE surfaces on the
// CPU path.  The GPU mirrors this with cudaTextureObject_t + tex2D<float>;
// the two diverge at pixel boundaries (linear vs nearest) but match for any
// ray that lands clearly inside / outside a pixel.
//
// Returns true (passthrough) when no pixels are loaded — image-aperture
// surfaces are sampled by check_image_aperture() below, which only invokes
// this for shape == APERTURE_IMAGE.
bool sample_aperture_image(const ApertureImage& img, const Vec3f& hit,
                           float aspect, float semi_diameter)
{
    if (img.pixels.empty()) return true; // passthrough: image not loaded yet
    if (img.width <= 0 || img.height <= 0 || !(aspect > 0.0f)
        || !(semi_diameter > 0.0f)
        || img.pixels.size() != (size_t)img.width * (size_t)img.height)
        return false;

    float hx = hit.x / aspect;
    float u  = 0.5f + hx     / (2.0f * semi_diameter);
    float v  = 0.5f + hit.y  / (2.0f * semi_diameter);
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return false;

    int px = (int)(u * (img.width  - 1) + 0.5f);
    int py = (int)(v * (img.height - 1) + 0.5f);
    return img.pixels[(size_t)py * img.width + px] > 0.5f;
}

// Per-surface dispatch: only triggers for APERTURE_IMAGE; falls through for
// CIRCLE / POLYGON (already handled inside check_aperture()).
inline bool check_image_aperture(const OpticalSystem& lens, int s, const Vec3f& hit)
{
    const Surface& surf = lens.surfaces[s];
    if (surf.aperture_shape != APERTURE_IMAGE) return true;
    if ((size_t)s >= lens.aperture_images.size()) return true; // defensively passthrough
    const Vec3f local = world_to_surface_point(hit, surf);
    return sample_aperture_image(lens.aperture_images[(size_t)s], local,
                                 surf.aperture_aspect,
                                 surf.aperture_semi_diameter);
}

// Outcome of one per-surface interaction: CONTINUE to the next surface, or DIED
// (the caller returns `result`, whose status is set here).
enum class StepOutcome { CONTINUE, DIED };

// Shared per-surface interaction for all four trace functions below.  Runs
// intersect -> image-aperture -> weight -> attenuator -> refract/reflect for
// surface s, mutating ray / current_ior / result.  Returns DIED on a vignette
// (intersect miss or image-aperture clip) or TIR, having set result.status and
// invoked the recorder for that event.
//
//   n2          : refractive index on the far side of surface s — ior_at(s) for
//                 a forward hit, ior_before(s) for the return leg. The
//                 caller supplies it so this helper stays direction-agnostic.
//   reflecting  : true at a ghost bounce surface (reflect); false to transmit.
//   reflect_ior : current_ior after a reflection — the (unchanged) incoming
//                 medium on the outbound leg, or the bounce surface's ior_at on
//                 the return leg. Ignored when !reflecting.
//   record      : per-surface event sink.  NoRecord{} on the fast paths, so the
//                 recording calls compile away; the diagnostic overloads pass a
//                 lambda that appends a TraceEvent.
template <class Recorder>
StepOutcome trace_surface_step(Ray& ray, const OpticalSystem& lens, int s,
                               float lambda_nm, float n2, bool reflecting,
                               float reflect_ior, float& current_ior,
                               TraceResult& result, Recorder&& record)
{
    float t = 0.0f; Vec3f norm;
    if (!intersect_surface(ray, lens.surfaces[s], t, norm))
    {
        record(s, ray.origin, Vec3f(0, 0, 0), current_ior, n2, 0.0f, false,
               TraceStatus::VIGNETTED);
        return StepOutcome::DIED;                 // result.status stays VIGNETTED
    }

    Vec3f hit = ray.origin + ray.dir * t;
    ray.origin = hit;
    if (!check_image_aperture(lens, s, hit))
    {
        record(s, hit, norm, current_ior, n2, 0.0f, false, TraceStatus::VIGNETTED);
        return StepOutcome::DIED;
    }

    const float n1 = current_ior;

    if (reflecting)
    {
        float fw = surface_weight(ray.dir, norm, n1, n2,
                                  lens.surfaces[s].coating, lambda_nm, true);
        result.weight *= fw;
        result.weight *= surface_attenuator(hit, lens.surfaces[s]);
        record(s, hit, norm, n1, n2, fw, true, TraceStatus::OK);
        ray.dir     = reflect_ray(ray.dir, norm);
        current_ior = reflect_ior;
    }
    else
    {
        Vec3f new_dir;
        if (!refract_ray(ray.dir, norm, n1, n2, new_dir))
        {
            record(s, hit, norm, n1, n2, 0.0f, false, TraceStatus::TIR);
            result.status = TraceStatus::TIR;
            return StepOutcome::DIED;
        }
        float fw = surface_weight(ray.dir, norm, n1, n2,
                                  lens.surfaces[s].coating, lambda_nm, false);
        result.weight *= fw;
        result.weight *= surface_attenuator(hit, lens.surfaces[s]);
        record(s, hit, norm, n1, n2, fw, false, TraceStatus::OK);
        ray.dir     = new_dir;
        current_ior = n2;
    }
    return StepOutcome::CONTINUE;
}

// No-op recorder for the fast (non-diagnostic) trace paths.
struct NoRecord {
    void operator()(int, const Vec3f&, const Vec3f&, float, float, float, bool,
                    TraceStatus) const {}
};

// Sensor-plane propagation, shared tail of every trace.  Sets result.position +
// status.  The image plane is the virtual z = 0 plane; on-axis rays land at
// exactly t_sensor == 0 when the last surface sits at z = 0, so the guard admits
// t == 0 (using !(t >= 0) to also reject NaN).
inline void propagate_to_sensor(const Ray& ray, TraceResult& result)
{
    if (std::abs(ray.dir.z) < 1e-12f)
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return;
    }
    float t_sensor = -ray.origin.z / ray.dir.z;
    if (!(t_sensor >= 0.0f))  // catches negative t and NaN (t == 0 lands on-sensor)
    {
        result.status = TraceStatus::MISSED_SURFACE;
        return;
    }
    result.position = ray.origin + ray.dir * t_sensor;
    result.status   = TraceStatus::OK;
}

} // namespace

// ============================================================================
// Fast path — no per-surface diagnostic recording.
// Used by filter_ghost_pairs() and any other hot path.
// ============================================================================

TraceResult trace_ghost_ray(const Ray& ray_in, const OpticalSystem& lens,
                            int bounce_a, int bounce_b)
{
    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;
    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    const int N = lens.num_surfaces();
    if (bounce_a < 0 || bounce_b < 0 || bounce_a >= bounce_b || bounce_b >= N)
    {
        result.status = TraceStatus::INVALID_INPUT;
        result.weight = 0.0f;
        return result;
    }
    float current_ior = 1.0f;
    NoRecord rec;

    // Outbound leg: forward through 0..bounce_b, transmitting except at bounce_b
    // (reflect). Muted surfaces are skipped: the ray passes through unbent and
    // the medium stack stays put. Ghost enumeration guarantees bounce_a /
    // bounce_b are active, so the reflect branch never fires on a muted surface.
    // The reflect at bounce_b leaves current_ior unchanged (still in the medium
    // before bounce_b), so reflect_ior == current_ior.
    for (int s = 0; s <= bounce_b; ++s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, s == bounce_b,
                               current_ior, current_ior, result, rec)
            == StepOutcome::DIED)
            return result;
    }

    // Return leg: backward through bounce_b-1..bounce_a, transmitting except at
    // bounce_a (reflect). Backward hit: n2 is the medium on the other side.
    // Reflecting at bounce_a puts the ray in the medium to the right of bounce_a.
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2          = lens.ior_before(s, lambda_nm);
        const float reflect_ior = (s == bounce_a)
                                ? lens.surfaces[bounce_a].ior_at(lambda_nm)
                                : current_ior;
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, s == bounce_a,
                               reflect_ior, current_ior, result, rec)
            == StepOutcome::DIED)
            return result;
    }

    // Sensor leg: forward through bounce_a+1..N-1 — the ray exits bounce_a going
    // forward and passes through all remaining surfaces (including those between
    // bounce_a and bounce_b a second time — correct for a double-bounce ghost).
    for (int s = bounce_a + 1; s < N; ++s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, false,
                               current_ior, current_ior, result, rec)
            == StepOutcome::DIED)
            return result;
    }

    propagate_to_sensor(ray, result);
    return result;
}

// ============================================================================
// trace_primary_ray — pure forward refraction, no bounces.
// ============================================================================

TraceResult trace_primary_ray(const Ray& ray_in, const OpticalSystem& lens)
{
    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;
    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    const int N = lens.num_surfaces();
    float current_ior = 1.0f;
    NoRecord rec;

    for (int s = 0; s < N; ++s)
    {
        // Muted surfaces are transparent — pass straight through.
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, false,
                               current_ior, current_ior, result, rec)
            == StepOutcome::DIED)
            return result;
    }

    propagate_to_sensor(ray, result);
    return result;
}

// ============================================================================
// trace_primary_ray diagnostic overload — records per-surface events.
// ============================================================================

TraceResult trace_primary_ray(const Ray& ray_in, const OpticalSystem& lens,
                              RayPath& path_out)
{
    path_out.events.clear();

    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;
    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    const int N = lens.num_surfaces();
    float current_ior = 1.0f;

    auto record = [&](int s, const Vec3f& hit, const Vec3f& norm,
                      float n1, float n2, float fw, bool reflected,
                      TraceStatus st)
    {
        TraceEvent ev;
        ev.surface_index  = s;
        ev.hit_point      = hit;
        ev.surface_normal = norm;
        ev.ior_before     = n1;
        ev.ior_after      = n2;
        ev.fresnel_weight = fw;
        ev.status         = st;
        ev.reflected      = reflected;
        path_out.events.push_back(ev);
    };

    for (int s = 0; s < N; ++s)
    {
        // Muted surfaces are transparent — no recorded event, no medium change.
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, false,
                               current_ior, current_ior, result, record)
            == StepOutcome::DIED)
        {
            path_out.result = result;
            return result;
        }
    }

    propagate_to_sensor(ray, result);
    path_out.result = result;
    return result;
}

// ============================================================================
// Diagnostic path — identical three-leg loop; appends a TraceEvent at
// every surface the ray reaches.
// ============================================================================

TraceResult trace_ghost_ray(const Ray& ray_in, const OpticalSystem& lens,
                            int bounce_a, int bounce_b,
                            RayPath& path_out)
{
    path_out.events.clear();

    Ray ray = ray_in;
    const float lambda_nm = ray_in.lambda;
    TraceResult result{};
    result.status = TraceStatus::VIGNETTED;
    result.weight = 1.0f;

    const int N = lens.num_surfaces();
    if (bounce_a < 0 || bounce_b < 0 || bounce_a >= bounce_b || bounce_b >= N)
    {
        result.status = TraceStatus::INVALID_INPUT;
        result.weight = 0.0f;
        path_out.result = result;
        return result;
    }
    float current_ior = 1.0f;

    auto record = [&](int s, const Vec3f& hit, const Vec3f& norm,
                      float n1, float n2, float fw, bool reflected,
                      TraceStatus st)
    {
        TraceEvent ev;
        ev.surface_index   = s;
        ev.hit_point       = hit;
        ev.surface_normal  = norm;
        ev.ior_before      = n1;
        ev.ior_after       = n2;
        ev.fresnel_weight  = fw;
        ev.status          = st;
        ev.reflected       = reflected;
        path_out.events.push_back(ev);
    };

    // Outbound leg: forward 0..bounce_b, reflect at bounce_b (current_ior unchanged).
    // Muted surfaces are transparent across every leg. Ghost enumeration
    // ensures bounce_a / bounce_b are active, so the reflect branches always have
    // a real surface.
    for (int s = 0; s <= bounce_b; ++s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, s == bounce_b,
                               current_ior, current_ior, result, record)
            == StepOutcome::DIED)
        {
            path_out.result = result;
            return result;
        }
    }

    // Return leg: backward bounce_b-1..bounce_a, reflect at bounce_a.
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2          = lens.ior_before(s, lambda_nm);
        const float reflect_ior = (s == bounce_a)
                                ? lens.surfaces[bounce_a].ior_at(lambda_nm)
                                : current_ior;
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, s == bounce_a,
                               reflect_ior, current_ior, result, record)
            == StepOutcome::DIED)
        {
            path_out.result = result;
            return result;
        }
    }

    // Sensor leg: forward bounce_a+1..N-1.
    for (int s = bounce_a + 1; s < N; ++s)
    {
        if (!lens.surfaces[s].is_active) continue;
        const float n2 = lens.surfaces[s].ior_at(lambda_nm);
        if (trace_surface_step(ray, lens, s, lambda_nm, n2, false,
                               current_ior, current_ior, result, record)
            == StepOutcome::DIED)
        {
            path_out.result = result;
            return result;
        }
    }

    propagate_to_sensor(ray, result);
    path_out.result = result;
    return result;
}
