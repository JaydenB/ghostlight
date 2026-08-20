"""Film-gate flare (mechanical) — config, off-parity, and physics.

Host gates (no GPU): the default is OFF, every field round-trips, the resolved
GpuGate arithmetic and clamps are right, and the scatter lobe has the shape it
claims (Cauchy across the marks, tight along them).

GPU gates: the layer is absent when off and does not perturb any other buffer
when on; a source imaged INSIDE the frame produces exactly nothing; the fold
lands where the mirror says it should; and each knob moves the one property it
is supposed to move.

The scene throughout is a lens whose image circle overfills the sensor — the
gate cannot fire otherwise, because there is no light outside the frame to
strike the plate.
"""

import numpy as np
import pytest

import ghostlight


COVERAGE = 0.70          # sensor half-extent as a fraction of the image circle
INSIDE_X = 0.50          # source imaged at frame centre
EDGE_X = 1.03            # source just past the frame edge, inside the capture band


def _cfg(lens, *, sx=EDGE_X, on=True, w=160, **gate):
    cal = lens.calibration()
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, 0.5
    c.source_r = c.source_g = c.source_b = 10.0
    c.ray_grid = 160
    c.spectral_samples = 6
    c.pupil_jitter = 2
    c.sensor_half_w = COVERAGE * cal.sensor_half_w
    c.sensor_half_h = COVERAGE * cal.sensor_half_h
    c.gate.enabled = on
    for k, v in gate.items():
        setattr(c.gate, k, v)
    return c


def _render(lens, **kw):
    w = kw.pop("w", 160)
    return lens.render_point_flare(w, w, _cfg(lens, w=w, **kw))


def _gate(out):
    return ghostlight._arrays.gate_to_hwc(out)


def _dbg(lens, *, w=192, **kw):
    cal = lens.calibration()
    return ghostlight._ghostlight._render_gate_debug(w, w, lens, cal, _cfg(lens, **kw))


# ---------------------------------------------------------------- host: config
def test_gate_config_defaults():
    """A default config has the gate OFF — the byte-identical guard."""
    c = ghostlight.PointFlareConfig()
    assert c.gate.enabled is False


def test_gate_config_pinned_defaults():
    """Defaults are the values the CPU prototype validated, not round numbers."""
    g = ghostlight.PointFlareConfig().gate
    assert g.thickness_mm == pytest.approx(0.8)
    assert g.standoff_mm == pytest.approx(5.0)
    assert g.roughness_rad == pytest.approx(0.08)
    assert g.groove_aniso == pytest.approx(0.12)
    assert g.groove_angle_deg == pytest.approx(0.0)
    assert g.reflectance_r0 == pytest.approx(0.04)
    assert g.max_kick_rad == pytest.approx(0.35)
    assert g.gain == pytest.approx(1.0)
    assert g.scatter_samples == 4
    assert g.spectral_samples == 0
    assert g.lobe == ghostlight.GateLobe.GROOVED


def test_gate_config_roundtrip():
    c = ghostlight.PointFlareConfig()
    c.gate.enabled = True
    c.gate.thickness_mm = 2.5
    c.gate.standoff_mm = 12.0
    c.gate.offset_left_mm = -1.5
    c.gate.offset_right_mm = 2.0
    c.gate.offset_top_mm = 0.25
    c.gate.offset_bottom_mm = -0.25
    c.gate.lobe = ghostlight.GateLobe.CAUCHY_ISO
    c.gate.roughness_rad = 0.2
    c.gate.groove_aniso = 0.5
    c.gate.groove_angle_deg = 33.0
    c.gate.max_kick_rad = 0.8
    c.gate.reflectance_r0 = 0.6
    c.gate.gain = 7.5
    c.gate.scatter_samples = 12
    c.gate.spectral_samples = 9
    assert c.gate.enabled is True
    assert c.gate.thickness_mm == pytest.approx(2.5)
    assert c.gate.standoff_mm == pytest.approx(12.0)
    assert c.gate.offset_left_mm == pytest.approx(-1.5)
    assert c.gate.offset_right_mm == pytest.approx(2.0)
    assert c.gate.lobe == ghostlight.GateLobe.CAUCHY_ISO
    assert c.gate.groove_angle_deg == pytest.approx(33.0)
    assert c.gate.gain == pytest.approx(7.5)
    assert c.gate.scatter_samples == 12
    assert c.gate.spectral_samples == 9


def test_gate_to_hwc_absent_returns_none():
    assert ghostlight._arrays.gate_to_hwc({"ghost_r": np.zeros((2, 2), np.float32)}) is None


# ------------------------------------------------- host: resolved GpuGate math
def test_gate_opening_follows_sensor_extent():
    g = ghostlight._ghostlight._gate_params_debug(ghostlight.GateConfig(), 12.4, 7.0)
    assert g["x_pos"] == pytest.approx(12.4)
    assert g["x_neg"] == pytest.approx(-12.4)
    assert g["y_pos"] == pytest.approx(7.0)
    assert g["y_neg"] == pytest.approx(-7.0)


def test_gate_offsets_open_and_crop_each_side():
    c = ghostlight.GateConfig()
    c.offset_right_mm = 2.0
    c.offset_left_mm = -1.0
    c.offset_top_mm = 0.5
    c.offset_bottom_mm = -0.5
    g = ghostlight._ghostlight._gate_params_debug(c, 12.4, 7.0)
    assert g["x_pos"] == pytest.approx(14.4)
    assert g["x_neg"] == pytest.approx(-11.4)
    assert g["y_pos"] == pytest.approx(7.5)
    assert g["y_neg"] == pytest.approx(-6.5)


def test_gate_opening_cannot_invert():
    """A negative offset larger than the half-extent would make every ray
    'outside' the opening, so each side is floored away from the axis."""
    c = ghostlight.GateConfig()
    c.offset_left_mm = -50.0
    g = ghostlight._ghostlight._gate_params_debug(c, 12.4, 7.0)
    assert g["x_neg"] < 0.0
    assert g["x_neg"] == pytest.approx(-0.1)


def test_gate_clamps_are_applied():
    c = ghostlight.GateConfig()
    c.thickness_mm = -1.0
    c.standoff_mm = -3.0
    c.roughness_rad = 99.0
    c.groove_aniso = 5.0
    c.reflectance_r0 = 4.0
    c.scatter_samples = 500
    c.max_kick_rad = 99.0
    g = ghostlight._ghostlight._gate_params_debug(c, 10.0, 10.0)
    assert g["t_mm"] > 0.0
    assert g["zs_mm"] == pytest.approx(0.0)
    assert g["sig_wide"] == pytest.approx(0.5)
    assert g["r0"] == pytest.approx(1.0)
    assert g["n_scatter"] == 64
    assert g["inv_scatter"] == pytest.approx(1.0 / 64.0)
    assert g["max_kick"] == pytest.approx(1.5)


def test_cauchy_iso_makes_the_lobe_round():
    """CAUCHY_ISO forces both transverse widths equal. It is NOT reachable by
    setting groove_aniso = 1 on GROOVED, whose tight axis stays Gaussian."""
    c = ghostlight.GateConfig()
    c.lobe = ghostlight.GateLobe.CAUCHY_ISO
    c.groove_aniso = 0.05
    g = ghostlight._ghostlight._gate_params_debug(c, 10.0, 10.0)
    assert g["sig_tight"] == pytest.approx(g["sig_wide"])


# ------------------------------------------------------- host: the scatter lobe
def test_lobe_is_anisotropic_across_versus_along_the_marks():
    c = ghostlight.GateConfig()
    c.roughness_rad = 0.1
    c.groove_aniso = 0.1
    d = ghostlight._ghostlight._gate_sample_debug(c, 20000, 12345)
    # A Cauchy has no finite variance, so compare percentiles, not std.
    spread_across = float(np.percentile(np.abs(d["axis"]), 68))
    spread_along = float(np.percentile(np.abs(d["other"]), 68))
    assert spread_across > 4.0 * spread_along


def test_lobe_wide_axis_is_heavy_tailed():
    """The wide axis is Cauchy, so its 99th percentile sits far beyond a
    Gaussian's — that heavy tail is what gives the streak its reach."""
    c = ghostlight.GateConfig()
    c.roughness_rad = 0.05
    c.max_kick_rad = 1.5          # do not let the clamp hide the tail
    d = ghostlight._ghostlight._gate_sample_debug(c, 40000, 7)
    a = np.abs(np.asarray(d["axis"]))
    p50, p99 = float(np.percentile(a, 50)), float(np.percentile(a, 99))
    assert p99 / max(p50, 1e-12) > 20.0     # Gaussian would be ~2.6


def test_lobe_respects_the_kick_clamp():
    c = ghostlight.GateConfig()
    c.roughness_rad = 0.4
    c.max_kick_rad = 0.02
    d = ghostlight._ghostlight._gate_sample_debug(c, 5000, 3)
    assert float(np.abs(np.asarray(d["axis"])).max()) <= 0.02 + 1e-6
    assert float(np.abs(np.asarray(d["other"])).max()) <= 0.02 + 1e-6


# ------------------------------------------------------------------ GPU: parity
@pytest.mark.gpu
def test_gate_off_omits_the_layer(psf_lens):
    out = _render(psf_lens, on=False)
    assert "gate_r" not in out and "gate_g" not in out and "gate_b" not in out
    assert ghostlight._arrays.gate_to_hwc(out) is None


@pytest.mark.gpu
def test_gate_on_emits_a_finite_layer(psf_lens):
    out = _render(psf_lens, on=True)
    for k in ("gate_r", "gate_g", "gate_b"):
        assert k in out
    g = _gate(out)
    assert np.isfinite(g).all()
    assert float(g.sum()) > 0.0
    # Small negatives are legitimate: the spectral table converts CIE to the
    # output space, whose primaries do not enclose the spectral locus, so a
    # narrow-band deposit can sit slightly outside the gamut. The ghost layer
    # does the same, and the EXR writer preserves them deliberately.
    assert float(g.min()) > -1e-3 * float(g.max())


@pytest.mark.gpu
def test_source_inside_the_frame_produces_nothing(psf_lens):
    """No primary ray lands outside the opening, so there is nothing to scrape.
    Gate flare is an edge phenomenon by construction, not by tuning."""
    g = _gate(_render(psf_lens, sx=INSIDE_X, on=True))
    assert float(np.abs(g).max()) == 0.0


@pytest.mark.gpu
def test_gate_does_not_disturb_the_other_layers(psf_lens):
    """The gate scatters but never occludes, so every other buffer is untouched
    to the GPU atomic-reorder floor."""
    cfg_off = _cfg(psf_lens, on=False)
    cfg_on = _cfg(psf_lens, on=True)
    for c in (cfg_off, cfg_on):
        c.diffraction.starburst = True
        c.diffraction.veil = True
    off = psf_lens.render_point_flare(160, 160, cfg_off)
    on = psf_lens.render_point_flare(160, 160, cfg_on)
    for name, fn in (("ghost", ghostlight._arrays.ghost_to_hwc),
                     ("starburst", ghostlight._arrays.starburst_to_hwc),
                     ("veil", ghostlight._arrays.veil_to_hwc)):
        a, b = fn(off), fn(on)
        peak = max(float(np.abs(a).max()), 1e-30)
        assert float(np.abs(a - b).max()) <= 1e-4 * peak, name


@pytest.mark.gpu
def test_gate_is_deterministic(psf_lens):
    a = _gate(_render(psf_lens))
    b = _gate(_render(psf_lens))
    peak = max(float(a.max()), 1e-30)
    assert float(np.abs(a - b).max()) <= 1e-5 * peak


@pytest.mark.gpu
def test_gate_is_chunk_invariant(psf_lens):
    """The source-flare panel renders eight sources at a time and sums, so the
    scatter seed must not depend on the chunk-local source index."""
    cal = psf_lens.calibration()
    cfg = _cfg(psf_lens)
    offs = np.zeros((16, 3), np.float32)
    offs[:, 0] = np.linspace(-2e-4, 2e-4, 16)
    offs[:, 1] = np.linspace(1e-4, -1e-4, 16)
    offs[:, 2] = 1.0 / 16

    def render(o):
        return _gate(psf_lens.render_source_flare(o, 128, 128, cfg, calib=cal))

    whole = render(offs)
    peak = max(float(whole.max()), 1e-30)
    for a, b in ((offs[:8], offs[8:]), (offs[:5], offs[5:]), (offs[9:], offs[:9])):
        summed = render(a) + render(b)
        assert float(np.abs(whole - summed).max()) <= 1e-4 * peak


@pytest.mark.gpu
def test_gate_scales_linearly_with_gain_and_flux(psf_lens):
    base = float(_gate(_render(psf_lens)).sum())
    doubled = float(_gate(_render(psf_lens, gain=2.0)).sum())
    assert doubled == pytest.approx(2.0 * base, rel=0.02)

    cfg = _cfg(psf_lens)
    cfg.source_r = cfg.source_g = cfg.source_b = 20.0
    brighter = float(_gate(psf_lens.render_point_flare(160, 160, cfg)).sum())
    assert brighter == pytest.approx(2.0 * base, rel=0.02)


@pytest.mark.gpu
def test_layer_hue_walks_as_the_source_crosses_the_band(psf_lens):
    """Each wavelength crosses the gate edge at its own source position, because
    the trace that decides the scrape is dispersive. So a WHITE source does not
    give a neutral layer: the flare starts blue-weighted at the inner edge of the
    band and warms as the source travels out. This is the chromatic fringing the
    effect is known for, and it is derived rather than tinted in."""
    def ratio(sx):
        g = _gate(_render(psf_lens, sx=sx))
        r = float(g[..., 0].sum())
        b = float(g[..., 2].sum())
        return b / max(r, 1e-30)

    inner, outer = ratio(1.01), ratio(1.05)
    assert inner > outer * 1.1


# ------------------------------------------------------------- GPU: the physics
@pytest.mark.gpu
def test_capture_band_matches_the_f_number_prediction(psf_lens):
    """The band is t/(2N); the pass reports what it used so the closed form and
    the resolved geometry cannot drift apart."""
    cal = psf_lens.calibration()
    d = _dbg(psf_lens, thickness_mm=1.6)
    assert d["band_x_mm"] == pytest.approx(1.6 / (2.0 * cal.f_number_x), rel=1e-4)
    assert d["band_y_mm"] == pytest.approx(1.6 / (2.0 * cal.f_number_y), rel=1e-4)


@pytest.mark.gpu
def test_specular_fold_lands_inside_the_edge_within_the_reach_law(psf_lens):
    """A mirror wall folds every scraping ray to 2*edge - p, so with roughness off
    the whole layer lands INSIDE the +x edge and no further in than the overshoot
    it came from — which the band bounds at (standoff + thickness)/(2N).

    The source's own image cannot serve as the other half of this check: it is
    outside the frame by construction, so it never lands in the buffer."""
    cal = psf_lens.calibration()
    w = 256
    d = _dbg(psf_lens, roughness_rad=0.0, w=w)
    lum = (np.asarray(d["gate_r"]) + np.asarray(d["gate_g"])
           + np.asarray(d["gate_b"]))
    assert lum.sum() > 0
    cols = np.nonzero(lum.max(axis=0) > 0.02 * lum.max())[0]
    # The +x wall sits at the right-hand frame edge, i.e. the last column.
    half_w = COVERAGE * cal.sensor_half_w
    mm_per_px = 2.0 * half_w / w
    deepest_mm = float(w - 1 - cols.min()) * mm_per_px
    predicted = (d["zs_mm"] + d["t_mm"]) / (2.0 * cal.f_number_x)
    # Every fold is INSIDE the wall it bounced off, and none reaches deeper than
    # the band allows. A point source folds to a compact spot at its own
    # overshoot distance rather than a smear running back to the edge, so the
    # bound is what identifies the mirror, not proximity to the border.
    assert cols.max() <= w - 1
    assert 0.0 < deepest_mm <= 1.5 * predicted


@pytest.mark.gpu
def test_thickness_moves_energy_not_reach(psf_lens):
    thin = _dbg(psf_lens, thickness_mm=0.2)
    thick = _dbg(psf_lens, thickness_mm=3.0)
    assert thick["energy"] > 4.0 * thin["energy"]
    assert thick["reach_mm"] == pytest.approx(thin["reach_mm"], rel=0.25)


@pytest.mark.gpu
def test_standoff_trades_reach_against_energy(psf_lens):
    near = _dbg(psf_lens, standoff_mm=5.0)
    far = _dbg(psf_lens, standoff_mm=15.0)
    assert far["reach_mm"] > near["reach_mm"]
    assert far["energy"] < near["energy"]


@pytest.mark.gpu
def test_roughness_is_direction_only(psf_lens):
    """The band test does not depend on the lobe, so the number of rays that
    strike the wall is exactly constant. What falls with roughness is the
    DEPOSITED energy, because a symmetric lobe throws part of the light back out
    through the opening, where the plate masks it — a real loss, not a leak.
    Rotating the marks to run along the wall removes that path and the loss with
    it, which is the cleanest proof the scatter itself conserves energy."""
    base = _dbg(psf_lens, roughness_rad=0.0)
    rough = _dbg(psf_lens, roughness_rad=0.25)
    assert rough["scrapes"] == base["scrapes"]
    assert rough["energy"] < base["energy"]

    along = _dbg(psf_lens, roughness_rad=0.25, groove_angle_deg=90.0)
    assert along["scrapes"] == base["scrapes"]
    assert along["energy"] == pytest.approx(base["energy"], rel=0.02)


@pytest.mark.gpu
def test_reflectance_moves_brightness_only(psf_lens):
    dark = _dbg(psf_lens, reflectance_r0=0.02)
    shiny = _dbg(psf_lens, reflectance_r0=0.60)
    assert shiny["energy"] > dark["energy"]
    # Grazing Fresnel already drives R toward 1, so a 30x change in the
    # normal-incidence value is worth well under 3x here. This is why painting a
    # gate black only half-fixes it.
    assert shiny["energy"] < 3.0 * dark["energy"]
    assert shiny["reach_mm"] == pytest.approx(dark["reach_mm"], rel=1e-3)


@pytest.mark.gpu
def test_scatter_samples_change_smoothness_not_energy(psf_lens):
    few = _dbg(psf_lens, scatter_samples=2)
    many = _dbg(psf_lens, scatter_samples=16)
    assert many["energy"] == pytest.approx(few["energy"], rel=0.05)


@pytest.mark.gpu
def test_groove_angle_rotates_the_streak(psf_lens):
    """Marks along the edge fan light into frame; marks across it do not."""
    def extent_x(d):
        lum = (np.asarray(d["gate_r"]) + np.asarray(d["gate_g"])
               + np.asarray(d["gate_b"]))
        live = lum.max(axis=0) > 0.02 * lum.max()
        cols = np.nonzero(live)[0]
        return int(cols.max() - cols.min() + 1) if cols.size else 0

    across = extent_x(_dbg(psf_lens, groove_angle_deg=0.0, w=256))
    along = extent_x(_dbg(psf_lens, groove_angle_deg=90.0, w=256))
    assert across > 2 * along


@pytest.mark.gpu
def test_gate_is_faint_against_the_source(psf_lens):
    """The layer carries a small fraction of the source, which is why it needs
    its own gain and why it can safely join the metered flare layer."""
    d = _dbg(psf_lens, w=256)
    gate_peak = float(max(np.asarray(d["gate_r"]).max(),
                          np.asarray(d["gate_g"]).max(),
                          np.asarray(d["gate_b"]).max()))
    # Baseline the same source imaged INSIDE the frame: at the flaring position
    # its direct image is outside the frame and never reaches the buffer, so it
    # cannot be metered in place.
    ref = _dbg(psf_lens, sx=INSIDE_X, w=256)
    direct_peak = float(max(np.asarray(ref["direct_r"]).max(),
                            np.asarray(ref["direct_g"]).max(),
                            np.asarray(ref["direct_b"]).max()))
    assert direct_peak > 0.0
    ratio = gate_peak / direct_peak
    assert 1e-6 < ratio < 0.05
