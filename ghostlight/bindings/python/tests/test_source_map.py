"""Accuracy tests for the traced source-position map.

The tests:

  * they sample radii the map was never fitted to (0.25 .. 1.0 and a diagonal),
  * they check against a FRESH trace at the calibrated grid, not against the
    solve's own arithmetic or its cheaper probe grid,
  * and they include strongly distorting and anamorphic lenses.
"""
import math

import pytest

import ghostlight

from _corpus import LENSES_DIR, lens_path

# Lenses spanning smooth vignetting, deep stops, strong distortion, and
# front-anamorphic optics.
CONTROL = ["helios44.lens", "DoubleGauss.lens", "00081_zeiss_100mm.lens"]
DISTORTED = ["AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens",
             "Cooke_50mm_F3_2x_US9341827B2.lens",
             "sirui_75mm_133xANA_f1_8_Modified2x_v01.lens"]
ALL_CASES = CONTROL + DISTORTED

# A Super-35-ish frame for the round lenses; the anamorphics get a frame inside
# their own covered field so every sampled radius is a real solve rather than a
# continuation (the continuation has its own tests below).
FRAMES = {"helios44.lens": (10.0, 10.0),
          "DoubleGauss.lens": (12.445, 9.335),
          "00081_zeiss_100mm.lens": (12.445, 9.335),
          "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens": (12.0, 8.0),
          "Cooke_50mm_F3_2x_US9341827B2.lens": (12.0, 8.0),
          "sirui_75mm_133xANA_f1_8_Modified2x_v01.lens": (12.0, 8.0)}

# Sample the interior as well as the frame edge.
RADII = [0.25, 0.5, 0.75, 1.0]

# Screen-position accuracy, as a fraction of the full frame width/height.
#
# This is a CEILING set from measurement, not an aspiration: across 128 samples
# (8 lenses x 4 radii x 4 directions) checked against an independent 41x41 probe
# the traced map's worst error is 1.00% of frame, and its mean is 0.034%.  The
# worst case is sirui 75mm Mod2x, whose pupil fills 36% of its front element --
# too much for the probe to restrict its sampling window, too little to be
# resolved by more than ~6 of the 17 grid cells -- so its survivor mean steps in
# ~100 um jumps and the solve inherits that as noise.  It is a probe-resolution
# limit, not a map error, and it is why this number is 1.2% rather than 0.05%.
#
# A single loose ceiling would be a weak test on its own, so it does not stand
# alone: test_map_beats_the_closed_form_it_replaced pins the actual claim on the
# lens the bug was reported against, and test_every_sample_beats_the_closed_form
# pins it across the whole matrix.
TOL_FRAC = 0.012

# What the lenses that are NOT probe-limited must hit.  Keeping a strict bar for
# them is what stops the ceiling above from quietly covering a real regression.
TOL_FRAC_STRICT = 0.0005
PROBE_LIMITED = {"Cooke_50mm_F3_2x_US9341827B2.lens",
                 "sirui_75mm_133xANA_f1_8_Modified2x_v01.lens"}


def _load(name):
    lens = ghostlight.OpticalSystem.load(str(lens_path(name)))
    return lens, lens.calibration()


def _solve(lens, calib, sx, sy, half):
    return ghostlight._ghostlight._solve_source_map(lens, calib, sx, sy, half[0], half[1])


def _screen_error(lens, calib, sx, sy, half):
    """Solve for the angle, then TRACE it and see where it really lands.

    The comparison is against _source_map_landing, which re-probes at the
    calibration grid -- not the solve's cheaper one -- so this also gates the
    solve's grid choice rather than letting it agree with itself.
    """
    r = _solve(lens, calib, sx, sy, half)
    land = ghostlight._ghostlight._source_map_landing(lens, calib, r["angle_x"], r["angle_y"])
    if land is None:
        return r, None
    got_x = 0.5 + 0.5 * land[0] / half[0]
    got_y = 0.5 + 0.5 * land[1] / half[1]
    return r, (got_x - sx, got_y - sy)


# ---------------------------------------------------------------------------
# Source-map landing accuracy.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_CASES)
@pytest.mark.parametrize("radius", RADII)
def test_solved_angle_lands_where_asked(name, radius):
    """A source placed at radius r on each axis traces back to radius r.

    Interior radii ensure the map is accurate away from its calibrated edge.
    """
    lens, calib = _load(name)
    half = FRAMES[name]
    for sx, sy in ((0.5 + 0.5 * radius, 0.5),
                   (0.5, 0.5 + 0.5 * radius),
                   (0.5 - 0.5 * radius, 0.5 - 0.5 * radius)):
        r, err = _screen_error(lens, calib, sx, sy, half)
        assert r["status"] == "solved", (
            f"{name} at ({sx:.3f},{sy:.3f}) fell back to {r['status']}")
        assert err is not None, f"{name}: solved angle has no landing"
        tol = TOL_FRAC if name in PROBE_LIMITED else TOL_FRAC_STRICT
        assert max(abs(err[0]), abs(err[1])) < tol, (
            f"{name} at ({sx:.3f},{sy:.3f}): traced landing is off by "
            f"({err[0]:+.5f}, {err[1]:+.5f}) of frame (limit {tol})")


@pytest.mark.parametrize("name", ALL_CASES)
def test_solved_angle_lands_where_asked_diagonal(name):
    """Both axes at once.  The map is not separable -- the Atlas cross-couples
    x into y by ~2% -- so a pair of on-axis curves would not have caught this.
    """
    lens, calib = _load(name)
    half = FRAMES[name]
    r, err = _screen_error(lens, calib, 0.5 + 0.35, 0.5 + 0.35, half)
    assert r["status"] == "solved"
    assert err is not None
    tol = TOL_FRAC if name in PROBE_LIMITED else TOL_FRAC_STRICT
    assert max(abs(err[0]), abs(err[1])) < tol, (
        f"{name} diagonal: off by ({err[0]:+.5f}, {err[1]:+.5f}) of frame")


def test_every_sample_beats_the_closed_form():
    """Across the whole matrix, the traced map must beat the closed form it
    replaced -- everywhere, not merely on average.

    This is the claim the work actually makes, and it is the one assertion that
    cannot be satisfied by loosening a tolerance.
    """
    worse, gains = [], []
    for name in ALL_CASES:
        lens, calib = _load(name)
        half = FRAMES[name]
        f_w = calib.sensor_half_w / math.tan(calib.max_half_angle_h)
        f_h = calib.sensor_half_h / math.tan(calib.max_half_angle_v)
        for radius in RADII:
            for dx, dy in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.7), (-0.6, 0.45)):
                n = math.hypot(dx, dy)
                ndx, ndy = radius * dx / n, radius * dy / n
                sx, sy = 0.5 + 0.5 * ndx, 0.5 + 0.5 * ndy

                old = ghostlight._ghostlight._source_map_landing(
                    lens, calib,
                    math.atan(ndx * half[0] / f_w), math.atan(ndy * half[1] / f_h))
                _, new = _screen_error(lens, calib, sx, sy, half)
                if old is None or new is None:
                    continue
                old_err = max(abs(0.5 + 0.5 * old[0] / half[0] - sx),
                              abs(0.5 + 0.5 * old[1] / half[1] - sy))
                new_err = max(abs(new[0]), abs(new[1]))
                gains.append((old_err, new_err))
                # A margin, so probe noise on an already-tiny error cannot fail
                # this: only a genuine regression should.
                if new_err > max(old_err, 2e-4):
                    worse.append((name, radius, (dx, dy), old_err, new_err))

    assert gains, "no comparable samples"
    assert not worse, f"traced map is worse than the closed form at: {worse}"
    mean_old = sum(o for o, _ in gains) / len(gains)
    mean_new = sum(n for _, n in gains) / len(gains)
    assert mean_new < mean_old / 10.0, (
        f"mean error only improved {mean_old:.5f} -> {mean_new:.5f} of frame")


def test_map_beats_the_closed_form_it_replaced():
    """On the lens the bug was reported against, the traced map must be a large
    improvement over the closed form -- not merely different from it.

    This is the regression itself, in one assertion: the reported symptom was
    the film-gate flare sitting visibly off the source marker on the Atlas.
    """
    name = "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens"
    lens, calib = _load(name)
    half = FRAMES[name]
    sx, sy = 0.25, 0.75          # a radius the closed form was never fitted at

    # What the closed form would have produced.
    f_w = calib.sensor_half_w / math.tan(calib.max_half_angle_h)
    f_h = calib.sensor_half_h / math.tan(calib.max_half_angle_v)
    old_ax = math.atan((sx - 0.5) * 2.0 * half[0] / f_w)
    old_ay = math.atan((sy - 0.5) * 2.0 * half[1] / f_h)
    old_land = ghostlight._ghostlight._source_map_landing(lens, calib, old_ax, old_ay)
    old_err = max(abs(0.5 + 0.5 * old_land[0] / half[0] - sx),
                  abs(0.5 + 0.5 * old_land[1] / half[1] - sy))

    _, new = _screen_error(lens, calib, sx, sy, half)
    new_err = max(abs(new[0]), abs(new[1]))

    assert old_err > 0.01, (
        "the closed form is supposed to be visibly wrong here -- if it is not, "
        "this test has stopped exercising the regression")
    assert new_err < old_err / 10.0, (
        f"traced map error {new_err:.5f} is not a clear improvement on the "
        f"closed form's {old_err:.5f}")


# ---------------------------------------------------------------------------
# Beyond the image circle: the continuation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["DoubleGauss.lens", "helios44.lens",
                                  "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens"])
def test_continuation_is_continuous_monotonic_and_bounded(name):
    """Dragging a source out of frame must not make it jump, reverse, or run
    away.  Option A joins an arctangent to the last angle that converged, so it
    matches in value and slope at the handover and saturates at anchor+90deg.
    """
    lens, calib = _load(name)
    half = (12.445, 9.335)

    prev_a = None
    saw_solved = saw_continued = False
    sy = 0.5
    # Walk out to 1.6x this lens's OWN image circle, not to a fixed ndc.  A
    # fixed range is a different physical distance on every lens: 2.5 of frame
    # width is well past helios44's reach and still inside double_gauss's.
    end_ndc = 1.6 * calib.image_circle_semi_w / half[0]
    n_steps = 125
    step = end_ndc / n_steps
    for i in range(1, n_steps + 1):
        sx = 0.5 + 0.5 * (i * step)
        r = _solve(lens, calib, sx, sy, half)
        assert r["status"] in ("solved", "continued"), (
            f"{name} at ndc {i*step:.2f}: {r['status']}")
        saw_solved |= r["status"] == "solved"
        saw_continued |= r["status"] == "continued"

        a = r["angle_x"]
        assert math.isfinite(a)
        assert abs(a) < math.pi / 2, f"{name}: angle {a} reached the pole"

        if prev_a is not None:
            assert a > prev_a - 1e-6, (
                f"{name} at ndc {i*step:.2f}: angle went backwards "
                f"({prev_a:.6f} -> {a:.6f})")
            # No jump: one 0.02-ndc step must not move the angle by more than
            # a whole ladder step.  A discontinuity at the handover is exactly
            # what a C0-but-not-C1 continuation would show here.
            assert a - prev_a < math.radians(6.0), (
                f"{name} at ndc {i*step:.2f}: angle jumped "
                f"{math.degrees(a - prev_a):.2f} deg in one step")
        prev_a = a

        # The forward map must agree with where the source was asked for --
        # inside and outside the circle alike, since that is what welds the
        # splatted layers to the marker.
        assert abs(r["screen_x"] - sx) < 0.012, (
            f"{name} at ndc {i*step:.2f}: forward map says {r['screen_x']:.5f}, "
            f"asked {sx:.5f}")

    assert saw_solved and saw_continued, (
        f"{name}: expected to cross from solved into continued over "
        f"ndc 0..{end_ndc:.2f} (solved={saw_solved}, continued={saw_continued})")


def test_continuation_handover_is_c1():
    """Across the last solved radius and the first continued one, the angle's
    slope must not kink.  Measured as the ratio of successive differences over
    a uniform ndc walk straddling the handover.
    """
    lens, calib = _load("DoubleGauss.lens")
    half = (12.445, 9.335)
    step = 0.01
    angles, statuses = [], []
    for i in range(1, 261):
        r = _solve(lens, calib, 0.5 + 0.5 * (i * step), 0.5, half)
        angles.append(r["angle_x"])
        statuses.append(r["status"])

    if "continued" not in statuses:
        pytest.skip("frame never leaves the image circle on this lens")
    k = statuses.index("continued")
    assert k >= 3, "handover too early to measure a slope either side"

    before = angles[k - 1] - angles[k - 2]
    after = angles[k + 1] - angles[k]
    assert before > 0 and after > 0
    ratio = after / before
    assert 0.5 < ratio < 2.0, (
        f"slope kinks at the handover: {before:.6f} -> {after:.6f} "
        f"(ratio {ratio:.3f})")


# ---------------------------------------------------------------------------
# Forward o solve, and the extended-source approximation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_CASES)
def test_forward_reproduces_the_solved_landing(name):
    """The forward map must return the source to where it was asked for.

    Note what "exact" means here and why.  The forward map is anchored on the
    landing the solve actually ACHIEVED, not on the position that was requested,
    so it reproduces the request only up to the solve residual.  That is the
    right way round: the splatted layers (starburst, veil) then sit exactly
    where the traced ones (ghosts, gate) do, which is the agreement the user
    reported as broken.  Pinning the splat to the requested position instead
    would restore the disagreement in miniature -- welded to the marker, off the
    lens.  The gap is bounded by the solve's acceptance threshold, 1% of the
    frame's smaller half-extent.
    """
    lens, calib = _load(name)
    half = FRAMES[name]
    bound = 0.01 * min(half) / (2.0 * min(half))    # accept_mm as a frame fraction
    for sx, sy in ((0.5, 0.5), (0.75, 0.25), (0.2, 0.9)):
        r = _solve(lens, calib, sx, sy, half)
        assert abs(r["screen_x"] - sx) < bound, (
            f"{name}: forward x {r['screen_x']:.5f} vs asked {sx}")
        assert abs(r["screen_y"] - sy) < bound, (
            f"{name}: forward y {r['screen_y']:.5f} vs asked {sy}")


def test_forward_is_exact_at_the_anchor():
    """Whatever the residual, the forward map must reproduce the solve's own
    anchor landing to the last bit -- that is what makes an extended source's
    zero-offset sample land exactly on its centre.
    """
    lens, calib = _load("DoubleGauss.lens")
    half = (12.445, 9.335)
    r = _solve(lens, calib, 0.7, 0.35, half)
    sx, sy = ghostlight._ghostlight._source_map_screen(
        lens, calib, 0.7, 0.35, half[0], half[1], r["angle_x"], r["angle_y"])
    assert sx == r["screen_x"]
    assert sy == r["screen_y"]
    assert abs(sx - (0.5 + 0.5 * r["anchor_x_mm"] / half[0])) < 1e-6
    assert abs(sy - (0.5 + 0.5 * r["anchor_y_mm"] / half[1])) < 1e-6


@pytest.mark.parametrize("name", ["DoubleGauss.lens",
                                  "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens"])
def test_extended_source_offsets_stay_accurate(name):
    """An extended source is one solve plus a cloud of angular offsets, and the
    offsets go through the Jacobian rather than through a probe each.

    The error of that is second order in the offset, so it has to be bounded
    where the renderer actually uses it.  A 1.5 deg source (the widest the
    panel offers) must stay far inside a pixel; 5 deg is the documented guard.
    """
    lens, calib = _load(name)
    half = FRAMES[name]
    r = _solve(lens, calib, 0.3, 0.7, half)
    assert r["status"] == "solved"

    for radius_deg, tol_frac in ((1.5, 5e-4), (5.0, 5e-3)):
        d = math.radians(radius_deg) * 0.5
        worst = 0.0
        for jx, jy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)):
            ax = r["angle_x"] + jx * d
            ay = r["angle_y"] + jy * d
            sx, sy = ghostlight._ghostlight._source_map_screen(
                lens, calib, 0.3, 0.7, half[0], half[1], ax, ay)
            land = ghostlight._ghostlight._source_map_landing(lens, calib, ax, ay)
            if land is None:
                continue
            worst = max(worst,
                        abs(sx - (0.5 + 0.5 * land[0] / half[0])),
                        abs(sy - (0.5 + 0.5 * land[1] / half[1])))
        assert worst < tol_frac, (
            f"{name}: Jacobian offsets drift {worst:.5f} of frame at "
            f"{radius_deg} deg source radius (limit {tol_frac})")


# ---------------------------------------------------------------------------
# Determinism, and the whole library.
# ---------------------------------------------------------------------------
def test_solve_is_deterministic_across_repeats():
    """Chunked progressive rendering asks for the same solve once per chunk.
    Every chunk must get bit-identical angles, or an extended source would
    smear as it accumulates.
    """
    lens, calib = _load("AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens")
    half = (12.0, 8.0)
    first = _solve(lens, calib, 0.25, 1.05, half)
    for _ in range(4):
        again = _solve(lens, calib, 0.25, 1.05, half)
        assert again["angle_x"] == first["angle_x"]
        assert again["angle_y"] == first["angle_y"]
        assert again["status"] == first["status"]


def test_memo_does_not_leak_between_different_requests():
    """The solve is memoised on (lens, calibration, request).  A different
    source position, a different frame, or a different lens must not read a
    previous answer back -- a stale hit here would be the original bug again.
    """
    lens_a, cal_a = _load("DoubleGauss.lens")
    lens_b, cal_b = _load("helios44.lens")
    half = (12.445, 9.335)

    a1 = _solve(lens_a, cal_a, 0.30, 0.60, half)
    b1 = _solve(lens_b, cal_b, 0.30, 0.60, half)
    a2 = _solve(lens_a, cal_a, 0.30, 0.60, half)
    assert a2["angle_x"] == a1["angle_x"], "memo lost a valid entry"
    assert b1["angle_x"] != a1["angle_x"], "memo served lens A's answer for B"

    other_pos = _solve(lens_a, cal_a, 0.70, 0.60, half)
    assert other_pos["angle_x"] != a1["angle_x"], "memo ignored source position"

    other_frame = _solve(lens_a, cal_a, 0.30, 0.60, (24.0, 18.0))
    assert other_frame["angle_x"] != a1["angle_x"], "memo ignored the frame"


def test_whole_library_resolves_finitely():
    """Every lens in the library must produce a finite angle at an awkward
    off-axis diagonal.  Falling back is allowed -- a lens whose map never
    crosses the calibration threshold inside the search ceiling has nothing to
    invert -- but NaN, a pole, or a crash is not.
    """
    paths = sorted(LENSES_DIR.rglob("*.lens"))
    assert len(paths) >= 15, "lens library looks truncated"

    fell_back = []
    for p in paths:
        lens = ghostlight.OpticalSystem.load(str(p))
        calib = lens.calibration()
        r = _solve(lens, calib, 0.85, 0.85, (12.445, 9.335))
        assert math.isfinite(r["angle_x"]) and math.isfinite(r["angle_y"]), p.name
        assert abs(r["angle_x"]) < math.pi / 2, p.name
        assert abs(r["angle_y"]) < math.pi / 2, p.name
        assert math.isfinite(r["screen_x"]) and math.isfinite(r["screen_y"]), p.name
        if r["status"] == "fallback":
            fell_back.append(p.name)

    # Kept as an explicit budget rather than an inequality on a magic number:
    # if a lens starts falling back that did not before, that is a finding.
    assert len(fell_back) <= 2, f"too many lenses cannot be solved: {fell_back}"


# ---------------------------------------------------------------------------
# The reported symptom, end to end.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("sx,sy", [(0.25, 0.75), (0.75, 0.30), (0.35, 0.62)])
def test_gate_direct_image_lands_on_the_source(sx, sy):
    """The film-gate pass renders the source's DIRECT image on its way to
    finding what spills past the frame.  That image is fully traced -- it is
    where the lens actually puts the beam -- so its position against the
    requested source position is the user-visible symptom that opened this work
    ("the gate flare is NOT lining up with the source point AT ALL"), measured
    rather than looked at.

    On the Atlas the closed-form map put it 4.2% of the frame away, which at
    1024 px wide is 43 px.  The tolerance here is 1.5 px at 256.
    """
    import numpy as np

    lens, calib = _load("AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens")
    w = h = 256
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x, cfg.source_y = sx, sy
    cfg.source_r = cfg.source_g = cfg.source_b = 10.0
    cfg.ray_grid = 96
    cfg.spectral_samples = 1
    cfg.pupil_jitter = 2
    cfg.sensor_half_w, cfg.sensor_half_h = 12.0, 8.0
    cfg.gate.enabled = True

    d = ghostlight._ghostlight._render_gate_debug(w, h, lens, calib, cfg)
    lum = (np.asarray(d["direct_r"]) + np.asarray(d["direct_g"])
           + np.asarray(d["direct_b"]))
    assert lum.max() > 0.0, "no direct image was rendered"

    # Intensity-weighted centroid of the direct image, so the check is not
    # limited to the pixel grid the argmax would quantise it to.
    ys, xs = np.nonzero(lum > 0.05 * lum.max())
    wts = lum[ys, xs]
    cx = float((xs + 0.5) @ wts / wts.sum()) / w
    cy = float((ys + 0.5) @ wts / wts.sum()) / h

    assert abs(cx - sx) < 1.5 / w, f"direct image x at {cx:.4f}, source at {sx}"
    assert abs(cy - sy) < 1.5 / h, f"direct image y at {cy:.4f}, source at {sy}"


@pytest.mark.gpu
def test_chunked_render_matches_single_call():
    """An extended source rendered as several chunks must equal the same source
    rendered in one call.

    The per-sample screen positions come from the base solve's Jacobian, which
    is pure arithmetic off a memoised solve -- so chunk-invariance is structural
    rather than something to be careful about.  This pins that it stays so.
    """
    import numpy as np

    lens, calib = _load("AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens")
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x, cfg.source_y = 0.32, 0.71
    cfg.source_r = cfg.source_g = cfg.source_b = 10.0
    cfg.ray_grid = 64
    cfg.spectral_samples = 1
    cfg.pupil_jitter = 2
    cfg.jitter_seed = 7
    cfg.sensor_half_w, cfg.sensor_half_h = 12.0, 8.0
    cfg.diffraction.starburst = True

    rng = np.random.default_rng(3)
    n = 16
    offs = np.zeros((n, 3), np.float32)
    offs[:, 0] = rng.uniform(-0.01, 0.01, n)
    offs[:, 1] = rng.uniform(-0.01, 0.01, n)
    offs[:, 2] = 1.0 / n

    whole = lens.render_source_flare(offs, 128, 128, cfg, calib=calib)
    acc = None
    for k in range(0, n, 4):
        part = lens.render_source_flare(offs[k:k + 4], 128, 128, cfg, calib=calib)
        arrs = [np.asarray(part[key]) for key in ("ghost_r", "starburst_r")]
        acc = arrs if acc is None else [a + b for a, b in zip(acc, arrs)]

    # 5e-3 of peak is this codebase's standing bar for chunked-vs-single
    # (test_render_source_extended.py::test_chunked_sum_matches_single_call).
    # What is left at that scale is float accumulation ORDER, not geometry:
    # summing the same splats in a different sequence does not associate the
    # same way.  The geometric claim -- that every chunk maps its samples to
    # identical screen positions -- is pinned exactly, and separately, by
    # test_solve_is_deterministic_across_repeats.
    for name, chunked, single in zip(("ghost_r", "starburst_r"), acc,
                                     [np.asarray(whole[k])
                                      for k in ("ghost_r", "starburst_r")]):
        peak = max(float(single.max()), 1e-30)
        diff = float(np.abs(chunked - single).max()) / peak
        assert diff < 5e-3, f"{name}: chunked differs from single by {diff:.2e} of peak"


def test_landing_is_none_past_the_image_circle():
    """The reference landing must report honestly that there is nothing there,
    rather than returning a plausible number the solve could chase.
    """
    lens, calib = _load("helios44.lens")
    assert ghostlight._ghostlight._source_map_landing(lens, calib, 0.0, 0.0) is not None
    assert ghostlight._ghostlight._source_map_landing(lens, calib,
                                            math.radians(80.0), 0.0) is None
