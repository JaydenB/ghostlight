"""Sensor at the z=0 convention, and the set_sensor_size API.

Convention: the sensor lives at world z=0 by design. ``SensorSpec`` builds
its quad/border at z=0; the viewport's ``set_sensor_size`` API resizes the
quad in place without disturbing the label.
"""

from __future__ import annotations

import pytest

from _helpers import load_example_lens


def test_from_calibration_builds_quad_at_origin():
    from ghostlight_viewport import SensorSpec
    lens = load_example_lens()
    calib = lens.calibration()
    spec = SensorSpec.from_calibration(calib)
    verts, _ = spec.build_quad()
    # Every vertex of the sensor quad sits on z=0.
    assert (verts[:, 2] == 0.0).all()
    # Half-extents still come from the calibration.
    assert spec.half_w == pytest.approx(float(calib.sensor_half_w))
    assert spec.half_h == pytest.approx(float(calib.sensor_half_h))


def test_set_sensor_size_api_exists_on_viewport():
    """The widget exposes ``set_sensor_size`` (smoke-level — full GL behaviour
    is covered by the demo)."""
    from ghostlight_viewport.widget import LensViewport
    assert callable(getattr(LensViewport, "set_sensor_size", None)), (
        "LensViewport.set_sensor_size missing"
    )
