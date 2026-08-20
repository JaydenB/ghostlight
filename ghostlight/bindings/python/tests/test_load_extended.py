"""Extended tests for lens file loading edge cases and complex lenses."""

import pathlib
import tempfile
import pytest
import ghostlight


_D_LINE = 587.56
_F_LINE = 486.13
_C_LINE = 656.27


# ---------------------------------------------------------------------------
# Loading known complex lenses
# ---------------------------------------------------------------------------

def test_load_doublegauss(doublegauss_lens):
    assert doublegauss_lens.num_surfaces() > 6


def test_load_cooketriplet(cooketriplet_lens):
    assert cooketriplet_lens.num_surfaces() >= 6


def test_doublegauss_chains_to_origin(doublegauss_lens):
    surfaces = doublegauss_lens.surfaces
    assert surfaces[0].z < 0.0
    last = surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0, abs=1e-3)


def test_cooketriplet_chains_to_origin(cooketriplet_lens):
    surfaces = cooketriplet_lens.surfaces
    assert surfaces[0].z < 0.0
    last = surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0, abs=1e-3)


def test_doublegauss_surface_ids_length(doublegauss_lens):
    n = doublegauss_lens.num_surfaces()
    assert len(doublegauss_lens.surface_ids) == n


def test_cooketriplet_surface_ids_length(cooketriplet_lens):
    n = cooketriplet_lens.num_surfaces()
    assert len(cooketriplet_lens.surface_ids) == n


def test_doublegauss_surface_z_ascending(doublegauss_lens):
    zs = [s.z for s in doublegauss_lens.surfaces]
    assert zs == sorted(zs), "surface z positions must be non-decreasing"


def test_cooketriplet_surface_z_ascending(cooketriplet_lens):
    zs = [s.z for s in cooketriplet_lens.surfaces]
    assert zs == sorted(zs)


def test_doublegauss_focal_length_positive(doublegauss_lens):
    assert doublegauss_lens.focal_length > 0.0


def test_cooketriplet_focal_length_positive(cooketriplet_lens):
    assert cooketriplet_lens.focal_length > 0.0


# ---------------------------------------------------------------------------
# IOR / dispersion of loaded lenses
# ---------------------------------------------------------------------------

def test_ior_before_first_surface_is_air(doublegauss_lens):
    """Air must precede the first optical surface."""
    assert doublegauss_lens.ior_before(0) == pytest.approx(1.0)


def test_ior_before_glass_surface_gt_one(doublegauss_lens):
    """At least one surface must have IOR > 1.0 (glass)."""
    found_glass = False
    for i in range(doublegauss_lens.num_surfaces()):
        if doublegauss_lens.ior_before(i) > 1.0:
            found_glass = True
            break
    assert found_glass, "No glass surface found"


def test_ior_before_at_f_gt_c_for_glass(doublegauss_lens):
    """For glass surfaces, F-line IOR must exceed C-line IOR (normal dispersion)."""
    sys = doublegauss_lens
    for i in range(sys.num_surfaces()):
        ior = sys.ior_before(i)
        if ior > 1.1:
            ior_f = sys.ior_before_at(i, _F_LINE)
            ior_c = sys.ior_before_at(i, _C_LINE)
            assert ior_f > ior_c, f"Surface {i}: expected F > C, got {ior_f} vs {ior_c}"
            break  # one glass surface is sufficient


# ---------------------------------------------------------------------------
# Aperture stop
# ---------------------------------------------------------------------------

def test_doublegauss_has_aperture_stop(doublegauss_lens):
    """Complex lens must have at least one aperture stop surface."""
    found_stop = any(s.is_stop for s in doublegauss_lens.surfaces)
    assert found_stop


def test_cooketriplet_has_aperture_stop(cooketriplet_lens):
    found_stop = any(s.is_stop for s in cooketriplet_lens.surfaces)
    assert found_stop


# ---------------------------------------------------------------------------
# finalize() is idempotent
# ---------------------------------------------------------------------------

def test_finalize_idempotent(doublegauss_lens):
    """Calling finalize() twice must produce the same z positions."""
    sys = doublegauss_lens
    zs_before = [s.z for s in sys.surfaces]
    sys.finalize()
    zs_after = [s.z for s in sys.surfaces]
    assert zs_before == pytest.approx(zs_after)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_nonexistent_file():
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load("nonexistent_file_xyz.lens")


def test_load_invalid_json():
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w", delete=False) as f:
        f.write("this is not valid json {{{")
        tmp_path = f.name
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load(tmp_path)


def test_load_empty_json_object():
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w", delete=False) as f:
        f.write("{}")
        tmp_path = f.name
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load(tmp_path)


# ---------------------------------------------------------------------------
# Surface semi-apertures are positive
# ---------------------------------------------------------------------------

def test_all_semi_apertures_positive(doublegauss_lens):
    for s in doublegauss_lens.surfaces:
        assert s.semi_aperture > 0.0


def test_all_semi_apertures_positive_cooke(cooketriplet_lens):
    for s in cooketriplet_lens.surfaces:
        assert s.semi_aperture > 0.0


# ---------------------------------------------------------------------------
# Aperture-shape defaults for lenses without aperture modifiers
# ---------------------------------------------------------------------------

def test_doublegauss_default_aperture_shapes(doublegauss_lens):
    """A lens with no aperture modifiers must leave every surface circular,
    aspect 1.0 — regression guard against parser mis-defaults."""
    for s in doublegauss_lens.surfaces:
        assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE
        assert s.aperture_blades == 0
        assert s.aperture_rotation_rad == 0.0
        assert s.aperture_aspect == pytest.approx(1.0)


def test_cooketriplet_default_aperture_shapes(cooketriplet_lens):
    for s in cooketriplet_lens.surfaces:
        assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE
        assert s.aperture_blades == 0
        assert s.aperture_rotation_rad == 0.0
        assert s.aperture_aspect == pytest.approx(1.0)
