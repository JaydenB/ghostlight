"""The entrance-pupil sampling mask derives its shape from
the is_stop surface, and FlareConfig.aperture_blades / aperture_rotation act
as overrides.

These tests cover the binding surface for those knobs:
  - aperture_blades / aperture_rotation live on FlareConfig (and therefore
    on PointFlareConfig by inheritance), with override semantics
  - RenderConfig does not expose them

The actual mask-shape effect on ghost output is a GPU-side optimisation; it
shapes which rays get traced before check_aperture(), so for the no-shape lens
the override engaging vs not is observable by counting kept rays.  We don't
drive that GPU path from a Python unit test — the sibling aperture-shape and
aperture-image tests already exercise the trace correctness against the
resolved aperture shape.
"""

import pytest
import ghostlight


# ---------------------------------------------------------------------------
# RenderConfig does not carry the override knobs
# ---------------------------------------------------------------------------

def test_render_config_has_no_aperture_blades():
    rc = ghostlight.RenderConfig()
    assert not hasattr(rc, "aperture_blades")


def test_render_config_has_no_aperture_rotation():
    rc = ghostlight.RenderConfig()
    assert not hasattr(rc, "aperture_rotation")


# ---------------------------------------------------------------------------
# FlareConfig + subclasses do carry them
# ---------------------------------------------------------------------------

def test_flare_config_default_aperture_blades_is_zero():
    fc = ghostlight.FlareConfig()
    assert fc.aperture_blades == 0
    assert fc.aperture_rotation == pytest.approx(0.0)


def test_flare_config_aperture_blades_round_trip():
    fc = ghostlight.FlareConfig()
    fc.aperture_blades = 6
    fc.aperture_rotation = 30.0
    assert fc.aperture_blades == 6
    assert fc.aperture_rotation == pytest.approx(30.0)


def test_point_flare_config_inherits_aperture_blades():
    pc = ghostlight.PointFlareConfig()
    pc.aperture_blades = 8
    pc.aperture_rotation = 22.5
    assert pc.aperture_blades == 8
    assert pc.aperture_rotation == pytest.approx(22.5)


# ---------------------------------------------------------------------------
# Smoke: the GPU ghost render still runs with the moved overrides set.
# Mostly a regression guard that the kernel launch / sampling-mask code
# correctly reads from FlareConfig now.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_render_point_flare_with_override_blades_runs(simple_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = 0.5
    cfg.spectral_samples = 3
    cfg.aperture_blades = 8  # override the lens default
    cfg.aperture_rotation = 15.0
    out = simple_lens.render_point_flare(64, 64, cfg)
    assert "ghost_r" in out


@pytest.mark.gpu
def test_render_point_flare_with_polygon_stop_no_override(simple_system):
    """A lens with a polygon is_stop should drive the mask shape when the
    override is 0 — and the render should complete without error."""
    # Mutate the stop surface to be a hexagon.
    for s in simple_system.surfaces:
        if s.is_stop:
            s.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
            s.aperture_blades = 6
            break
    lens = simple_system
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = 0.3
    cfg.spectral_samples = 3
    # config.aperture_blades = 0 → mask derived from the lens (hexagonal stop)
    out = lens.render_point_flare(64, 64, cfg)
    assert "ghost_r" in out
