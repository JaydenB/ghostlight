"""Tests for programmatic lens construction."""

import pytest
import ghostlight


def test_finalize_places_sensor_at_origin(simple_system):
    # Convention: sensor at z=0. Chain ends at z=0 via the last surface's
    # vertex + its thickness (the BFD). simple_system thicknesses are
    # 5, 10, 0, so total=15 and the last vertex sits exactly at z=0.
    last = simple_system.surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0)


def test_finalize_z_positions(simple_system):
    # New convention: surfaces are laid backward from 0, total=15.
    expected_z = [-15.0, -10.0, 0.0]
    actual_z   = [s.z for s in simple_system.surfaces]
    for expected, actual in zip(expected_z, actual_z):
        assert actual == pytest.approx(expected)


def test_finalize_surface_ids_resized(simple_system):
    # finalize() must resize surface_ids to match surfaces length
    assert len(simple_system.surface_ids) == simple_system.num_surfaces()


def test_finalize_empty_raises():
    sys = ghostlight.OpticalSystem()
    with pytest.raises(RuntimeError):
        sys.finalize()


def test_mutate_thickness_and_refinalize(simple_system):
    # Change thickness of first surface and refinalize.
    # New total = 8 + 10 + 0 = 18; chain ends at 0, so z = [-18, -10, 0].
    simple_system.surfaces[0].thickness = 8.0
    simple_system.finalize()
    assert simple_system.surfaces[0].z == pytest.approx(-18.0)
    assert simple_system.surfaces[1].z == pytest.approx(-10.0)
    assert simple_system.surfaces[2].z == pytest.approx(0.0)


def test_finalize_invalidates_cache(simple_system):
    # OpticalSystem.finalize() invalidates the cached calibration key
    lens = simple_system
    # Touch calibration to populate cache
    # (We don't call full calibration here; just test invalidation via cache key)
    initial_key = lens._current_key()
    lens.surfaces[0].radius = 50.0
    lens.finalize()
    new_key = lens._current_key()
    assert initial_key != new_key
