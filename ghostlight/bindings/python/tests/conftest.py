"""Shared fixtures for Ghostlight Python binding tests."""

import pathlib
import sys

# The built extension lives in this package directory.  Inserting it lets the
# suite run from any working directory: from the repository root, `ghostlight`
# would otherwise resolve to the C++ source directory of the same name, which
# has no __init__.py and imports as an empty namespace package.
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import pytest  # noqa: E402
import numpy as np  # noqa: E402

import ghostlight  # noqa: E402

from _corpus import FIXTURES_DIR, LENSES_DIR, lens_path  # noqa: E402,F401

EXAMPLE_LENS = lens_path("example_doublet.lens")
DOUBLEGAUSS_LENS = lens_path("doublegauss_wide.lens")
COOKETRIPLET_LENS = lens_path("cooketriplet.lens")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA-capable GPU")


def pytest_collection_modifyitems(config, items):
    """Skip gpu-marked tests when no CUDA device is available."""
    if not ghostlight._cuda_available():
        skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def example_lens_path() -> str:
    return str(EXAMPLE_LENS)


@pytest.fixture
def loaded_lens(example_lens_path) -> ghostlight.OpticalSystem:
    return ghostlight.OpticalSystem.load(example_lens_path)


@pytest.fixture
def psf_lens() -> ghostlight.OpticalSystem:
    """Double Gauss 58mm f/2 — strong but *contained* coma off-axis, used for
    PSF centring / orientation tests. Distinct from the wide-angle
    `doublegauss_lens` fixture, whose ~5 mm image circle and heavy aberration
    overflow the diagnostic tile.

    Its illuminated edge reaches ~38 mm, which is `calib.image_circle_semi_w`.
    `calib.sensor_half_w` is a different number — the covered field, where
    vignetting begins — so read the former when asking what the lens covers."""
    return ghostlight.OpticalSystem.load(str(lens_path("DoubleGauss.lens")))


@pytest.fixture
def simple_system() -> ghostlight.OpticalSystem:
    """Minimal three-surface lens built programmatically (singlet + stop)."""
    sys = ghostlight.OpticalSystem()
    sys.name = "test_singlet"

    # Front surface: BK7-like sphere, radius 47 mm, 5 mm thick
    s0 = ghostlight.Surface()
    s0.radius = 47.0
    s0.thickness = 5.0
    s0.ior = 1.5168
    s0.abbe_v = 64.17
    s0.semi_aperture = 15.0
    s0.disp_model = ghostlight.DispersionModel.ABBE
    sys.surfaces.append(s0)

    # Rear surface: flat, air, 10 mm to stop
    s1 = ghostlight.Surface()
    s1.radius = 0.0
    s1.thickness = 10.0
    s1.ior = 1.0
    s1.abbe_v = 0.0
    s1.semi_aperture = 15.0
    s1.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(s1)

    # Aperture stop
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


@pytest.fixture
def simple_lens(simple_system) -> ghostlight.OpticalSystem:
    """Alias for `simple_system`; the OpticalSystem is the unit under test."""
    return simple_system


@pytest.fixture
def doublegauss_lens_path() -> str:
    return str(DOUBLEGAUSS_LENS)


@pytest.fixture
def cooketriplet_lens_path() -> str:
    return str(COOKETRIPLET_LENS)


@pytest.fixture
def doublegauss_lens() -> ghostlight.OpticalSystem:
    return ghostlight.OpticalSystem.load(str(DOUBLEGAUSS_LENS))


@pytest.fixture
def cooketriplet_lens() -> ghostlight.OpticalSystem:
    return ghostlight.OpticalSystem.load(str(COOKETRIPLET_LENS))


@pytest.fixture
def bk7_surface() -> ghostlight.Surface:
    """BK7 glass surface using Abbe dispersion model (n_d=1.5168, V=64.17)."""
    s = ghostlight.Surface()
    s.radius = 50.0
    s.thickness = 5.0
    s.ior = 1.5168
    s.abbe_v = 64.17
    s.semi_aperture = 20.0
    s.disp_model = ghostlight.DispersionModel.ABBE
    return s


@pytest.fixture
def sellmeier_bk7_surface() -> ghostlight.Surface:
    """BK7 glass surface using Sellmeier dispersion (Schott catalogue coefficients)."""
    s = ghostlight.Surface()
    s.radius = 50.0
    s.thickness = 5.0
    s.ior = 1.5168
    s.abbe_v = 64.17
    s.semi_aperture = 20.0
    s.disp_model = ghostlight.DispersionModel.SELLMEIER
    # Schott BK7 Sellmeier coefficients
    import numpy as np
    s.sellmeier_B[:] = [1.03961212, 0.23179234, 1.01046945]
    s.sellmeier_C[:] = [6.00069867e-3, 2.00179144e-2, 1.03560653e2]
    return s
