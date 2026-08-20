"""Tests for V2 coating models: artist tint, spectral / angular / spectral-
angular reflectance tables, physical TMM layer stacks, and their round-trip.

Reflectance is isolated per surface through the diagnostic primary trace,
whose per-event ``fresnel_weight`` is the coating-aware transmittance (1 - R)
at that surface (see trace.cpp:347).  So R(lambda) = 1 - fresnel_weight[0]
for a coating placed on surface 0.
"""

import math
import pathlib
import struct
import tempfile

import numpy as np
import pytest

import ghostlight

from _corpus import lens_path  # noqa: E402

_D_LINE = 587.56
EXAMPLE_LENS = lens_path("example_doublet.lens")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coated_singlet():
    """A BK7-ish singlet (glass front + flat air rear + stop) whose surface 0
    is the coating target.  Mirrors conftest.simple_system but returned fresh
    so tests can mutate the coating freely."""
    sys = ghostlight.OpticalSystem()
    sys.name = "coated_singlet"

    s0 = ghostlight.Surface()
    s0.radius = 60.0
    s0.thickness = 5.0
    s0.ior = 1.5168
    s0.abbe_v = 64.17
    s0.semi_aperture = 15.0
    s0.disp_model = ghostlight.DispersionModel.ABBE
    sys.surfaces.append(s0)

    s1 = ghostlight.Surface()
    s1.radius = 0.0
    s1.thickness = 10.0
    s1.ior = 1.0
    s1.abbe_v = 0.0
    s1.semi_aperture = 15.0
    s1.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(s1)

    s2 = ghostlight.Surface()
    s2.radius = 0.0
    s2.thickness = 0.0
    s2.ior = 1.0
    s2.abbe_v = 0.0
    s2.semi_aperture = 8.0
    s2.is_stop = True
    s2.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(s2)

    sys.finalize()
    return sys


def _on_axis(wavelength):
    return ghostlight.Ray(ghostlight.Vec3f(0.0, 0.0, -200.0), ghostlight.Vec3f(0.0, 0.0, 1.0), wavelength)


def _surface0_reflectance(sys, wavelength):
    """R at surface 0 = 1 - transmittance, read from the diagnostic trace."""
    path = ghostlight.trace_primary_ray_diagnostic(_on_axis(wavelength), sys)
    assert len(path.events) >= 1
    return 1.0 - path.events[0].fresnel_weight


def test_surface_structure_edits_keep_coating_tables_aligned():
    sys = _coated_singlet()
    table = np.array([[450.0, 0.1], [650.0, 0.2]], dtype=np.float32)
    sys.set_coating_spectral_table(0, table)
    sys.insert_surface(0, ghostlight.Surface(), "inserted")
    np.testing.assert_array_equal(sys.get_coating_table(1), table)
    assert len(sys.surface_ids) == len(sys.aperture_images) == sys.num_surfaces()
    sys.erase_surface(0)
    np.testing.assert_array_equal(sys.get_coating_table(0), table)


def test_failed_layer_update_preserves_existing_coating():
    sys = _coated_singlet()
    table = np.array([[450.0, 0.1], [650.0, 0.2]], dtype=np.float32)
    sys.set_coating_spectral_table(0, table)
    invalid = {
        "material": "invalid",
        "thickness_nm": 0.0,
        "nk_table": np.array([[0.55, 1.4, 0.0]], dtype=np.float32),
    }
    with pytest.raises(ValueError):
        sys.set_coating_layers(0, [invalid])
    assert sys.surfaces[0].coating.model == ghostlight.CoatingModel.SPECTRAL
    np.testing.assert_array_equal(sys.get_coating_table(0), table)


def test_gaussian_attenuator_uses_surface_local_coordinates():
    sys = ghostlight.OpticalSystem()
    surface = ghostlight.Surface()
    surface.thickness = 0.0
    surface.semi_aperture = 10.0
    surface.decenter_x = 3.0
    surface.coating.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
    surface.coating.gauss_sigma = 0.5
    surface.coating.gauss_background = 0.1
    surface.coating.gauss_peak = 0.9
    sys.surfaces.append(surface)
    sys.finalize()

    centered = ghostlight.trace_primary_ray(
        ghostlight.Ray(ghostlight.Vec3f(3.0, 0.0, -10.0), ghostlight.Vec3f(0.0, 0.0, 1.0)),
        sys,
    )
    off_center = ghostlight.trace_primary_ray(
        ghostlight.Ray(ghostlight.Vec3f(0.0, 0.0, -10.0), ghostlight.Vec3f(0.0, 0.0, 1.0)),
        sys,
    )
    assert centered.weight == pytest.approx(1.0)
    assert off_center.weight < 0.11


def test_zero_sigma_attenuator_is_finite():
    sys = _coated_singlet()
    coating = sys.surfaces[0].coating
    coating.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
    coating.gauss_sigma = 0.0
    coating.gauss_background = 0.25
    coating.gauss_peak = 0.75
    result = ghostlight.trace_primary_ray(_on_axis(_D_LINE), sys)
    assert math.isfinite(result.weight)


# ---------------------------------------------------------------------------
# ARTIST
# ---------------------------------------------------------------------------

def test_artist_white_is_flat():
    """A white tint gives R(lambda) == strength at every wavelength."""
    sys = _coated_singlet()
    c = sys.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ARTIST
    c.tint_r = c.tint_g = c.tint_b = 1.0
    c.tint_strength = 0.10

    r450 = _surface0_reflectance(sys, 450.0)
    r550 = _surface0_reflectance(sys, 550.0)
    r650 = _surface0_reflectance(sys, 650.0)
    assert r450 == pytest.approx(0.10, abs=1e-4)
    assert r550 == pytest.approx(0.10, abs=1e-4)
    assert r650 == pytest.approx(0.10, abs=1e-4)


def test_artist_red_tint_peaks_red():
    """A red tint reflects more at 650 than 550 than 450 nm."""
    sys = _coated_singlet()
    c = sys.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ARTIST
    c.tint_r, c.tint_g, c.tint_b = 1.0, 0.0, 0.0
    c.tint_strength = 0.10

    r450 = _surface0_reflectance(sys, 450.0)
    r550 = _surface0_reflectance(sys, 550.0)
    r650 = _surface0_reflectance(sys, 650.0)
    assert r650 > r550 > r450


def test_artist_strength_monotone():
    """Higher strength gives higher reflectance for the same tint."""
    sys = _coated_singlet()
    c = sys.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ARTIST
    c.tint_r = c.tint_g = c.tint_b = 1.0

    c.tint_strength = 0.02
    lo = _surface0_reflectance(sys, 550.0)
    c.tint_strength = 0.20
    hi = _surface0_reflectance(sys, 550.0)
    assert hi > lo
    assert hi == pytest.approx(0.20, abs=1e-4)


def test_artist_matches_formula():
    """R matches the analytic Gaussian-basis formula at an arbitrary lambda."""
    sys = _coated_singlet()
    c = sys.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ARTIST
    tr, tg, tb, strength = 0.8, 0.3, 0.1, 0.12
    c.tint_r, c.tint_g, c.tint_b, c.tint_strength = tr, tg, tb, strength

    lam = 512.0
    inv = 1.0 / (2.0 * 65.0 * 65.0)
    br = math.exp(-((lam - 650.0) ** 2) * inv)
    bg = math.exp(-((lam - 550.0) ** 2) * inv)
    bb = math.exp(-((lam - 450.0) ** 2) * inv)
    expected = strength * (tr * br + tg * bg + tb * bb) / (br + bg + bb)
    assert _surface0_reflectance(sys, lam) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# SPECTRAL
# ---------------------------------------------------------------------------

def test_spectral_exact_at_nodes_and_midpoints():
    sys = _coated_singlet()
    table = np.array([[400.0, 0.02], [550.0, 0.005], [700.0, 0.04]], dtype=np.float32)
    sys.set_coating_spectral_table(0, table)
    assert int(sys.surfaces[0].coating.model) == int(ghostlight.CoatingModel.SPECTRAL)

    assert _surface0_reflectance(sys, 400.0) == pytest.approx(0.02, abs=1e-4)
    assert _surface0_reflectance(sys, 550.0) == pytest.approx(0.005, abs=1e-4)
    assert _surface0_reflectance(sys, 700.0) == pytest.approx(0.04, abs=1e-4)
    # Midpoint between 400 and 550 -> mean of endpoints
    mid = _surface0_reflectance(sys, 475.0)
    assert mid == pytest.approx(0.5 * (0.02 + 0.005), abs=1e-4)


def test_spectral_clamp_vs_discard():
    sys = _coated_singlet()
    table = np.array([[500.0, 0.01], [600.0, 0.03]], dtype=np.float32)

    # Clamp: a wavelength below the table uses the edge value.
    sys.set_coating_spectral_table(0, table, out_of_range_discard=False)
    assert _surface0_reflectance(sys, 450.0) == pytest.approx(0.01, abs=1e-4)

    # Discard: below-range wavelength kills the ray entirely.
    sys.set_coating_spectral_table(0, table, out_of_range_discard=True)
    path = ghostlight.trace_primary_ray_diagnostic(_on_axis(450.0), sys)
    # Weight collapses to 0 (discarded) — no OK primary energy survives.
    assert path.result.weight == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ANGULAR (Snell-invariant reference angle)
# ---------------------------------------------------------------------------

def test_angular_normal_incidence_lookup():
    """On-axis ray hits surface 0 at normal incidence -> table's 0-deg row."""
    sys = _coated_singlet()
    table = np.array([[0.0, 0.01], [45.0, 0.05], [80.0, 0.30]], dtype=np.float32)
    sys.set_coating_angular_table(0, table, angle_ref_ior=1.0)
    assert int(sys.surfaces[0].coating.model) == int(ghostlight.CoatingModel.ANGULAR)
    assert _surface0_reflectance(sys, _D_LINE) == pytest.approx(0.01, abs=1e-4)


# ---------------------------------------------------------------------------
# SPECTRAL_ANGULAR
# ---------------------------------------------------------------------------

def test_sa_bilinear_at_normal_incidence():
    sys = _coated_singlet()
    wl = np.array([400.0, 700.0], dtype=np.float32)
    ang = np.array([0.0, 60.0], dtype=np.float32)
    r = np.array([[0.02, 0.10], [0.06, 0.20]], dtype=np.float32)
    sys.set_coating_sa_table(0, wl, ang, r, angle_ref_ior=1.0)
    assert int(sys.surfaces[0].coating.model) == int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)

    # Normal incidence, lambda = 400 -> r[0,0]; lambda 550 -> midway on wl axis.
    assert _surface0_reflectance(sys, 400.0) == pytest.approx(0.02, abs=1e-4)
    assert _surface0_reflectance(sys, 550.0) == pytest.approx(0.04, abs=2e-3)


# ---------------------------------------------------------------------------
# TMM layer stacks
# ---------------------------------------------------------------------------

def _mgf2_nk(thickness_nm, n=1.38):
    nk = np.array([[0.40, n, 0.0], [0.55, n, 0.0], [0.70, n, 0.0]], dtype=np.float32)
    return {"material": "MgF2", "thickness_nm": thickness_nm, "nk_table": nk}


def test_tmm_bakes_sa_table():
    sys = _coated_singlet()
    sys.set_coating_layers(0, [_mgf2_nk(99.6)])
    assert int(sys.surfaces[0].coating.model) == int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)
    wl, ang, r = sys.get_coating_sa_table(0)
    assert r.shape == (31, 18)
    assert wl[0] == pytest.approx(400.0) and wl[-1] == pytest.approx(700.0)
    assert ang[0] == pytest.approx(0.0) and ang[-1] == pytest.approx(85.0)
    assert np.all((r >= 0.0) & (r <= 1.0))


def test_tmm_single_layer_matches_airy():
    """A single quarter-wave MgF2 layer baked via TMM must match the analytic
    Airy formula used by the SIMPLE ar_layers=1 model, at normal incidence,
    across the visible range.

    We compare the two by tracing the same singlet with (a) the TMM stack and
    (b) the SIMPLE single-layer coating, and requiring their reflectances to
    agree to a few thousandths.
    """
    lambdas = [450.0, 550.0, 650.0]

    # (a) SIMPLE single-layer MgF2 (quarter-wave at 550 nm ~ 99.6 nm)
    simple = _coated_singlet()
    simple.surfaces[0].coating.ar_layers = 1
    r_simple = [_surface0_reflectance(simple, l) for l in lambdas]

    # (b) TMM stack: one MgF2 quarter-wave layer
    tmm = _coated_singlet()
    tmm.set_coating_layers(0, [_mgf2_nk(550.0 / (4.0 * 1.38))])
    r_tmm = [_surface0_reflectance(tmm, l) for l in lambdas]

    for a, b in zip(r_simple, r_tmm):
        assert a == pytest.approx(b, abs=3e-3)


def test_tmm_zero_reflection_at_design_wavelength():
    """A quarter-wave layer whose index is sqrt(n_substrate) gives ~0 R at the
    design wavelength (ideal single-layer AR)."""
    sys = _coated_singlet()
    n_sub = 1.5168
    n_coat = math.sqrt(n_sub)          # ideal AR index
    design = 550.0
    thickness = design / (4.0 * n_coat)
    nk = np.array([[0.40, n_coat, 0.0], [0.55, n_coat, 0.0], [0.70, n_coat, 0.0]],
                  dtype=np.float32)
    sys.set_coating_layers(0, [{"material": "ideal", "thickness_nm": thickness,
                                "nk_table": nk}])
    assert _surface0_reflectance(sys, design) == pytest.approx(0.0, abs=2e-3)


# ---------------------------------------------------------------------------
# coating_state_hash / cache invalidation
# ---------------------------------------------------------------------------

def test_coating_state_hash_reacts_to_edits():
    sys = _coated_singlet()
    h0 = sys.coating_state_hash()

    sys.surfaces[0].coating.model = ghostlight.CoatingModel.ARTIST
    sys.surfaces[0].coating.tint_r = 0.9
    h1 = sys.coating_state_hash()
    assert h1 != h0

    sys.set_coating_spectral_table(0, np.array([[400, 0.02], [700, 0.05]], dtype=np.float32))
    h2 = sys.coating_state_hash()
    assert h2 != h1

    # No-op re-read is stable.
    assert sys.coating_state_hash() == h2


def test_clear_coating_resets_to_simple():
    sys = _coated_singlet()
    sys.set_coating_spectral_table(0, np.array([[400, 0.02], [700, 0.05]], dtype=np.float32))
    sys.clear_coating(0)
    assert int(sys.surfaces[0].coating.model) == int(ghostlight.CoatingModel.SIMPLE)
    assert sys.surfaces[0].coating.table_count == 0
    assert sys.get_coating_table(0).shape[0] == 0


# ---------------------------------------------------------------------------
# Round-trip through the writer + parser (this guards designer undo)
# ---------------------------------------------------------------------------

def _roundtrip(sys):
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "rt.lens")
        sys.save(p)
        return ghostlight.OpticalSystem.load(p), p


def test_version_is_not_content_derived():
    """The emitted version must not depend on what the lens contains.

    It used to: a rich coating (or a non-zero element pivot) bumped the file
    to major 2 and its absence dropped it back to 1, so the same lens flipped
    format across saves in both directions. One format, one version.
    """
    import json

    def _saved_version(mutate=None):
        sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
        if mutate is not None:
            mutate(sys)
        with tempfile.TemporaryDirectory() as d:
            p = str(pathlib.Path(d) / "rt.lens")
            sys.save(p)
            doc = json.loads(pathlib.Path(p).read_text())
        return doc["version"]

    def _artist(sys):
        c = sys.surfaces[0].coating
        c.model = ghostlight.CoatingModel.ARTIST
        c.tint_r, c.tint_g, c.tint_b, c.tint_strength = 0.7, 0.3, 0.1, 0.09

    plain = _saved_version()
    rich = _saved_version(_artist)
    assert plain == rich
    assert plain == {"major": ghostlight.LENS_FORMAT_MAJOR,
                     "minor": ghostlight.LENS_FORMAT_MINOR}


def test_roundtrip_artist():
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    idx = 0
    c = sys.surfaces[idx].coating
    c.model = ghostlight.CoatingModel.ARTIST
    c.tint_r, c.tint_g, c.tint_b, c.tint_strength = 0.7, 0.3, 0.1, 0.09

    import json
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "rt.lens")
        sys.save(p)
        doc = json.loads(pathlib.Path(p).read_text())
        reload = ghostlight.OpticalSystem.load(p)

    rc = reload.surfaces[idx].coating
    assert int(rc.model) == int(ghostlight.CoatingModel.ARTIST)
    assert rc.tint_r == pytest.approx(0.7, abs=1e-4)
    assert rc.tint_g == pytest.approx(0.3, abs=1e-4)
    assert rc.tint_strength == pytest.approx(0.09, abs=1e-4)


def test_roundtrip_spectral_table():
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    table = np.array([[420, 0.03], [540, 0.006], [660, 0.028]], dtype=np.float32)
    sys.set_coating_spectral_table(0, table, out_of_range_discard=True)

    reload, _ = _roundtrip(sys)
    rc = reload.surfaces[0].coating
    assert int(rc.model) == int(ghostlight.CoatingModel.SPECTRAL)
    assert bool(rc.out_of_range_discard) is True
    np.testing.assert_allclose(reload.get_coating_table(0), table, atol=1e-3)


def test_roundtrip_layer_stack_reemits_layers():
    """A layer stack must round-trip as 'layers' intent (not the baked table),
    so it stays editable and re-bakes identically on reload."""
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    sys.set_coating_layers(0, [_mgf2_nk(99.6)])
    _, baked_before = sys.get_coating_sa_table(0), sys.get_coating_sa_table(0)[2]

    reload, _ = _roundtrip(sys)
    rc = reload.surfaces[0].coating
    assert int(rc.model) == int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)
    layers = reload.get_coating_layers(0)
    assert len(layers) == 1
    assert layers[0]["material"] == "MgF2"
    assert layers[0]["thickness_nm"] == pytest.approx(99.6, abs=1e-3)
    # Re-baked table matches the original bake.
    np.testing.assert_allclose(reload.get_coating_sa_table(0)[2], baked_before, atol=1e-4)


def test_roundtrip_attenuator():
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    c = sys.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
    c.gauss_sigma = 8.0
    c.gauss_background = 0.1
    c.gauss_peak = 0.7
    c.gauss_decenter_x = 1.5

    reload, _ = _roundtrip(sys)
    rc = reload.surfaces[0].coating
    assert int(rc.model) == int(ghostlight.CoatingModel.ATTENUATOR_GAUSS)
    assert rc.gauss_sigma == pytest.approx(8.0, abs=1e-4)
    assert rc.gauss_peak == pytest.approx(0.7, abs=1e-4)
    assert rc.gauss_decenter_x == pytest.approx(1.5, abs=1e-4)


@pytest.mark.parametrize("major", [0, 2, 3])
def test_version_gate_rejects_other_majors(major):
    """Only the current major loads.

    2 is in this list on purpose: it is the superseded format, and a stale
    file has to fail loudly rather than load as something subtly different.
    """
    import json
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "bad.lens"
        sys.save(str(p))
        doc = json.loads(p.read_text())
        doc["version"]["major"] = major
        p.write_text(json.dumps(doc))
        with pytest.raises(Exception):
            ghostlight.OpticalSystem.load(str(p))


def test_coating_without_model_is_a_hard_error():
    """A coating modifier must declare its model.

    The TMM layer stack must not be discriminated by the mere presence of a
    ``layers`` array, and an unrecognised model degraded silently to
    "uncoated" — a physics change hiding in a typo.
    """
    import json
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    sys.set_coating_layers(0, [_mgf2_nk(99.6)])
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "nomodel.lens"
        sys.save(str(p))
        doc = json.loads(p.read_text())
        for el in doc["optical_system"]:
            for surf in el["surfaces"]:
                for mod in surf.get("modifiers", []):
                    if mod.get("type") == "coating":
                        mod.pop("model", None)
        p.write_text(json.dumps(doc))
        with pytest.raises(Exception):
            ghostlight.OpticalSystem.load(str(p))


def test_unknown_coating_model_is_a_hard_error():
    import json
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    sys.surfaces[0].coating.ar_layers = 2
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "badmodel.lens"
        sys.save(str(p))
        doc = json.loads(p.read_text())
        for el in doc["optical_system"]:
            for surf in el["surfaces"]:
                for mod in surf.get("modifiers", []):
                    if mod.get("type") == "coating":
                        mod["model"] = "no_such_model"
        p.write_text(json.dumps(doc))
        with pytest.raises(Exception):
            ghostlight.OpticalSystem.load(str(p))


# ---------------------------------------------------------------------------
# GPU ghost tint
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_artist_red_tints_ghost_red():
    """A red ARTIST coat on an interior surface must push the ghost buffer's
    red channel well above blue, versus a neutral baseline."""
    def render(apply_red):
        sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
        if apply_red:
            for i in range(sys.num_surfaces()):
                c = sys.surfaces[i].coating
                c.model = ghostlight.CoatingModel.ARTIST
                c.tint_r, c.tint_g, c.tint_b = 1.0, 0.05, 0.05
                c.tint_strength = 0.20
        cfg = ghostlight.PointFlareConfig()
        cfg.ray_grid = 24
        cfg.spectral_samples = 16
        cfg.source_r = cfg.source_g = cfg.source_b = 5.0
        return sys.render_point_flare(48, 48, cfg)

    red = render(True)
    base = render(False)
    red_sum = float(np.sum(red["ghost_r"]))
    blue_sum = float(np.sum(red["ghost_b"]))
    assert red_sum > 0.0
    # Red-tinted ghosts: red channel dominates blue.
    assert red_sum > 2.0 * blue_sum
    # And redder than the neutral baseline's channel ratio.
    base_r = float(np.sum(base["ghost_r"])) + 1e-9
    base_b = float(np.sum(base["ghost_b"])) + 1e-9
    assert (red_sum / (blue_sum + 1e-9)) > (base_r / base_b)


@pytest.mark.gpu
def test_spectral_samples_3_legacy_still_tints():
    """The legacy 3-sample path (650/550/450) must still show tint direction."""
    sys = ghostlight.OpticalSystem.load(str(EXAMPLE_LENS))
    for i in range(sys.num_surfaces()):
        c = sys.surfaces[i].coating
        c.model = ghostlight.CoatingModel.ARTIST
        c.tint_r, c.tint_g, c.tint_b = 1.0, 0.05, 0.05
        c.tint_strength = 0.20
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 24
    cfg.spectral_samples = 3
    cfg.source_r = cfg.source_g = cfg.source_b = 5.0
    out = sys.render_point_flare(48, 48, cfg)
    assert float(np.sum(out["ghost_r"])) > 2.0 * float(np.sum(out["ghost_b"]))
