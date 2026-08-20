"""Tests for loading a lens from a JSON file."""

import pytest
import ghostlight


def test_load_example_lens(example_lens_path):
    lens = ghostlight.OpticalSystem.load(example_lens_path)
    # example_doublet.lens: 3 surfaces in doublet + 1 stop + 2 rear singlet = 6
    assert lens.num_surfaces() == 6
    assert lens.name == "Example Doublet + Singlet"


def test_load_chains_to_origin(example_lens_path):
    lens = ghostlight.OpticalSystem.load(example_lens_path)
    # Sensor convention: chain ends at z=0; surfaces extend into negative z.
    surfaces = lens.surfaces
    assert surfaces[0].z < 0.0
    last = surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0, abs=1e-3)


def test_load_surface_z_ascending(example_lens_path):
    lens = ghostlight.OpticalSystem.load(example_lens_path)
    zs = [s.z for s in lens.surfaces]
    assert zs == sorted(zs), "surface z positions must be non-decreasing"


def test_load_bad_path():
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load("does_not_exist.lens")


def test_lens_repr(example_lens_path):
    lens = ghostlight.OpticalSystem.load(example_lens_path)
    r = repr(lens)
    assert "OpticalSystem(" in r
    assert "surfaces=6" in r
