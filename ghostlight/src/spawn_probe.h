// ============================================================================
// spawn_probe.h — Probing the disc the renderers actually spawn.
//
// Measures the area and survivor-mean landing of the same shifted entrance
// disc used by the render kernels. Calibration and source mapping share these
// primitives so their forward and inverse maps use one landing definition.
// ============================================================================
#pragma once

#include "aperture_sampler.h"   // PupilMask
#include "optical_system.h"
#include "trace.h"              // TraceResult

// ---------------------------------------------------------------------------
// Trace a ray from a given entrance-plane height at a given field angle.
// The entrance plane is SPAWN_OFFSET ahead of the first surface — the same spawn
// plane the flare/PSF renderers use — so the marginal-ray solve in
// lens_calibration.cpp measures the pupil in the coordinates the diffraction
// pupil builder samples.
//
// This single-ray primitive is shift-free. Bundle probes apply spawn_shift()
// to their sample positions; paraxial and marginal-ray solves do not.
// ---------------------------------------------------------------------------
bool trace_from_ok(const OpticalSystem& lens, float hx, float hy,
                   float angle_h, float angle_v, float lambda_nm,
                   TraceResult* result_out = nullptr);

// A tracked sampling window concentrates resolution on deep pupils. It is
// carried across field angles and widened whenever survivors reach its edge.
struct Window {
    float cu = 0.0f, cv = 0.0f;   // centre, in normalised disc coords
    float r  = 1.0f;              // half-extent of the sampled square
};

struct DiscArea {
    float area   = 0.0f;   // absolute normalised-disc area, coverage-weighted
    float mean_x = 0.0f;   // area-weighted mean sensor landing of the survivors
    float mean_y = 0.0f;
    bool  at_edge = false; // a survivor reached the sampled edge: window too small
    Window fit;            // survivor bounding box, to seed the next angle
};

// The window a search should start from, given how much of the front element
// admits light (entrance_pupil_semi / front_semi_aperture, the smaller axis).
// A lens that fills its front element gets the whole disc — restricting the
// window would only add work. A deep-stop lens starts from its measured axial
// pupil, dilated to hold the pupil's walk between two probe angles.
Window initial_probe_window(float pupil_fill);

// Surviving area of the renderers' spawn disc at one field angle, over `win`.
//
// Boundary cells are subsampled for fractional area coverage.
//
// `grid` is the sample count per axis; 0 means the calibrated default (17,
// sized by measurement — see spawn_probe.cpp). The source-map solve passes a
// smaller one: it reads only mean_x/mean_y, which converge far faster than the
// area does, so it buys back the cost of running per render.
//
// `edge_ss` is the boundary-cell subsampling rate; 0 means the calibrated
// default (2). Refining it sharpens the survivor MEAN specifically: the mean
// steps whenever a cell changes weight, and a finer boundary splits each of
// those steps into smaller ones without touching interior sampling.
DiscArea probe_window(const OpticalSystem& lens, const PupilMask& mask,
                      float front_R, float angle_h, float angle_v,
                      float lambda_nm, const Window& win, int grid = 0, int edge_ss = 0);

// probe_window() with the window management around it: re-centre on the beam
// from angle to angle, and widen when the beam outruns the sampled square.
//
// `track` carries the window across calls and is left holding whatever size
// this call settled on. That size only ever grows within a search, and a caller
// comparing areas across calls must re-measure its reference whenever it does —
// a window's cell area is (2r/n)^2, so an area measured at one size and divided
// by one measured at another compares two different quantisations, and on a
// deep-stop lens a single boundary cell is a large fraction of the answer.
// (A caller that only reads mean_x/mean_y, as the source map does, is immune:
// the mean is a ratio of two sums sharing the same cell area.)
DiscArea probe_disc(const OpticalSystem& lens, const PupilMask& mask,
                    float front_R, float angle_h, float angle_v,
                    float lambda_nm, Window* track, int grid = 0, int edge_ss = 0);
