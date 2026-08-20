"""Tests for image-aperture stops.

Covers:
  - Surface POD default for aperture_semi_diameter
  - ApertureImage struct round-trip (numpy pixels in/out)
  - OpticalSystem.aperture_images plumbing (parser + Python)
  - CPU sample_aperture_image() behaviour via trace_primary_ray, with the
    2x2 / aspect=2 tables from the spec
  - load_aperture_images() helper end-to-end through a tmpdir PNG
"""

import json
import math
import os
import pathlib
import tempfile
import numpy as np
import pytest
import ghostlight


_D_LINE = 587.56


# ---------------------------------------------------------------------------
# Surface defaults
# ---------------------------------------------------------------------------

def test_surface_default_aperture_semi_diameter_is_zero():
    s = ghostlight.Surface()
    assert s.aperture_semi_diameter == pytest.approx(0.0)


def test_aperture_shape_image_enum_value():
    assert int(ghostlight.ApertureShape.IMAGE) == 2


def test_aperture_image_default_construct():
    img = ghostlight.ApertureImage()
    assert img.width == 0
    assert img.height == 0
    assert img.semi_diameter == pytest.approx(0.0)
    assert img.source_path == ""


def test_aperture_image_pixels_roundtrip():
    img = ghostlight.ApertureImage()
    img.semi_diameter = 5.0
    pixels = np.array([[1.0, 0.0],
                       [0.0, 1.0]], dtype=np.float32)
    img.pixels = pixels
    assert img.width  == 2
    assert img.height == 2
    out = np.asarray(img.pixels)
    assert out.shape == (2, 2)
    assert np.allclose(out, pixels)


def test_lens_system_aperture_images_list_grows_with_surfaces(simple_system):
    """finalize() must size aperture_images to match surfaces.size()."""
    # simple_system has 3 surfaces; the conftest finalize() should have sized
    # the parallel vectors.
    assert len(simple_system.aperture_images) == simple_system.num_surfaces()


# ---------------------------------------------------------------------------
# CPU sampling via the trace path
# ---------------------------------------------------------------------------

def _make_image_stop_system(pixels: np.ndarray,
                            semi_diameter: float = 5.0,
                            aspect: float = 1.0,
                            semi_aperture: float = 10.0) -> ghostlight.OpticalSystem:
    """Single-surface flat stop, image aperture, populated pixels."""
    sys = ghostlight.OpticalSystem()
    sys.name = "image_stop_only"

    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = semi_aperture
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    stop.aperture_aspect = aspect
    stop.aperture_semi_diameter = semi_diameter
    sys.surfaces.append(stop)

    sys.finalize()

    # finalize() sized aperture_images parallel to surfaces; populate slot 0.
    img = sys.aperture_images[0]
    img.semi_diameter = semi_diameter
    img.pixels = pixels.astype(np.float32, copy=False)
    return sys


def _hit_status(system: ghostlight.OpticalSystem, x: float, y: float) -> ghostlight.TraceStatus:
    ray = ghostlight.Ray(ghostlight.Vec3f(x, y, -10.0), ghostlight.Vec3f(0.0, 0.0, 1.0), _D_LINE)
    return ghostlight.trace_primary_ray(ray, system).status


def test_image_aperture_2x2_diagonal_pattern():
    """2x2 image, (0,0)+(1,1) white, (0,1)+(1,0) black, semi_diameter=5."""
    # Image convention: pixels[row, col] where row is Y, col is X.
    # (0,0) → top-left pixel, (1,1) → bottom-right.
    # u = 0.5 + hx/(2*sd); v = 0.5 + hy/(2*sd).
    # (-2.4, -2.4) → u=v≈0.26 → nearest (col=0,row=0)=1 (passes)
    # (-2.4,  2.4) → u≈0.26,v≈0.74 → (col=0,row=1)=0 (blocked)
    # ( 2.4, -2.4) → u≈0.74,v≈0.26 → (col=1,row=0)=0 (blocked)
    # ( 2.4,  2.4) → u≈0.74,v≈0.74 → (col=1,row=1)=1 (passes)
    pixels = np.array([[1.0, 0.0],
                       [0.0, 1.0]], dtype=np.float32)
    sys = _make_image_stop_system(pixels, semi_diameter=5.0)
    assert _hit_status(sys, -2.4, -2.4) == ghostlight.TraceStatus.OK
    assert _hit_status(sys, -2.4,  2.4) == ghostlight.TraceStatus.VIGNETTED
    assert _hit_status(sys,  2.4, -2.4) == ghostlight.TraceStatus.VIGNETTED
    assert _hit_status(sys,  2.4,  2.4) == ghostlight.TraceStatus.OK


def test_image_aperture_out_of_bounds_blocked():
    pixels = np.array([[1.0, 1.0],
                       [1.0, 1.0]], dtype=np.float32)
    sys = _make_image_stop_system(pixels, semi_diameter=5.0)
    # Inside bounding circle, inside image bounds: passes.
    assert _hit_status(sys, 0.0, 0.0) == ghostlight.TraceStatus.OK
    # Outside image bounds (y=6 > semi_diameter=5): blocked.
    assert _hit_status(sys, 0.0, 6.0) == ghostlight.TraceStatus.VIGNETTED


def test_image_aperture_aspect2_x_stretched():
    """aspect=2: image is 2x stretched in X.  hx = x/aspect → x=4.8 sees u≈0.74."""
    pixels = np.array([[1.0, 0.0],
                       [0.0, 1.0]], dtype=np.float32)
    sys = _make_image_stop_system(pixels, semi_diameter=5.0, aspect=2.0)
    # x=4.8, y=2.4 → hx=2.4, hy=2.4 → u≈0.74, v≈0.74 → (col=1,row=1)=1 (passes)
    assert _hit_status(sys,  4.8,  2.4) == ghostlight.TraceStatus.OK
    # x=4.8, y=-2.4 → hx=2.4, hy=-2.4 → u≈0.74, v≈0.26 → (col=1,row=0)=0 (blocked)
    assert _hit_status(sys,  4.8, -2.4) == ghostlight.TraceStatus.VIGNETTED


def test_image_aperture_uses_surface_local_coordinates():
    pixels = np.zeros((3, 3), dtype=np.float32)
    pixels[1, 1] = 1.0
    sys = _make_image_stop_system(pixels, semi_diameter=5.0)
    sys.surfaces[0].decenter_x = 3.0
    assert _hit_status(sys, 3.0, 0.0) == ghostlight.TraceStatus.OK
    assert _hit_status(sys, 0.0, 0.0) == ghostlight.TraceStatus.VIGNETTED


def test_image_aperture_passthrough_when_pixels_empty():
    """Surface marked APERTURE_IMAGE but with no pixels loaded — passthrough."""
    sys = ghostlight.OpticalSystem()
    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.semi_aperture = 10.0
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    stop.aperture_semi_diameter = 5.0
    sys.surfaces.append(stop)
    sys.finalize()
    # No pixels loaded — every ray inside the bounding circle should pass.
    assert _hit_status(sys, 0.0, 0.0) == ghostlight.TraceStatus.OK
    assert _hit_status(sys, 2.0, 0.0) == ghostlight.TraceStatus.OK


# ---------------------------------------------------------------------------
# Parser: "shape": "image" round-trip
# ---------------------------------------------------------------------------

def _doublet_doc_with_image_modifier(modifier: dict) -> dict:
    return {
        "format": "ghostlight-optical",
        "version": {"major": 1, "minor": 0},
        "metadata": {"name": "Image aperture test", "focal_length_mm": 50.0},
        "glass_catalogue": {
            "N-BK7": {
                "name": "N-BK7",
                "dispersion": {
                    "model": "sellmeier",
                    "B": [1.03961212, 0.23179234, 1.01046945],
                    "C": [0.00600069867, 0.02001791440, 103.560653],
                },
            },
        },
        "optical_system": [
            {
                "type": "element",
                "name": "Front",
                "transform": {"position": {"x": 0, "y": 0, "z": 0}},
                "materials": [{"glass": "N-BK7"}],
                "surfaces": [
                    {"semi_aperture": 25.0, "thickness": 4.0,
                     "form": {"type": "sphere", "radius": 50.0}},
                    {"semi_aperture": 25.0,
                     "form": {"type": "sphere", "radius": -50.0}},
                ],
            },
            {
                "type": "element",
                "name": "Stop",
                "transform": {"position": {"x": 0, "y": 0, "z": 10.0}},
                "materials": [],
                "surfaces": [
                    {
                        "semi_aperture": 10.0,
                        "is_stop": True,
                        "form": {"type": "sphere", "radius": 0.0},
                        "modifiers": [modifier],
                    },
                ],
            },
        ],
    }


def _write_and_load(doc: dict) -> ghostlight.OpticalSystem:
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w",
                                     delete=False) as f:
        json.dump(doc, f)
        path = f.name
    return ghostlight.OpticalSystem.load(path)


def _find_stop_index(lens: ghostlight.OpticalSystem) -> int:
    for i, s in enumerate(lens.surfaces):
        if s.is_stop:
            return i
    raise AssertionError("no stop surface")


def test_parse_image_modifier_records_path_and_semi_diameter():
    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "stop.png",
        "semi_diameter": 7.5,
        "aperture_aspect": 1.5,
    })
    lens = _write_and_load(doc)
    idx = _find_stop_index(lens)
    s = lens.surfaces[idx]
    assert s.aperture_shape == ghostlight.ApertureShape.IMAGE
    assert s.aperture_semi_diameter == pytest.approx(7.5)
    assert s.aperture_aspect == pytest.approx(1.5)
    img = lens.aperture_images[idx]
    assert img.source_path == "stop.png"
    assert img.semi_diameter == pytest.approx(7.5)
    # Pixels remain empty until load_aperture_images() runs.
    assert np.asarray(img.pixels).size == 0


def test_image_aperture_survives_a_save_reload_round_trip(tmp_path):
    """The writer must re-emit the mask path.

    Emitting an empty ``"image_path"`` would mean an image aperture was
    lost on every save — and, because designer undo round-trips the whole
    document through this writer, on every Ctrl+Z as well. The reload then
    warned and silently fell back to a circular aperture.
    """
    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "stop.png",
        "semi_diameter": 7.5,
        "aperture_aspect": 1.5,
    })
    lens = _write_and_load(doc)
    idx = _find_stop_index(lens)

    out = tmp_path / "rt.lens"
    lens.save(str(out))

    written = json.loads(out.read_text(encoding="utf-8"))
    mods = [m
            for el in written["optical_system"]
            for surf in el["surfaces"]
            for m in surf.get("modifiers", [])
            if m.get("type") == "aperture" and m.get("shape") == "image"]
    assert len(mods) == 1, "image aperture modifier was dropped on save"
    assert mods[0]["image_path"] == "stop.png"
    assert mods[0]["semi_diameter"] == pytest.approx(7.5)

    reloaded = ghostlight.OpticalSystem.load(str(out))
    s = reloaded.surfaces[idx]
    assert s.aperture_shape == ghostlight.ApertureShape.IMAGE
    assert s.aperture_semi_diameter == pytest.approx(7.5)
    assert s.aperture_aspect == pytest.approx(1.5)
    assert reloaded.aperture_images[idx].source_path == "stop.png"


def test_parse_image_modifier_missing_path_falls_back_to_circle():
    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "semi_diameter": 5.0,
    })
    lens = _write_and_load(doc)
    idx = _find_stop_index(lens)
    assert lens.surfaces[idx].aperture_shape == ghostlight.ApertureShape.CIRCLE


def test_parse_image_modifier_missing_semi_diameter_falls_back_to_circle():
    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "stop.png",
    })
    lens = _write_and_load(doc)
    idx = _find_stop_index(lens)
    assert lens.surfaces[idx].aperture_shape == ghostlight.ApertureShape.CIRCLE


# ---------------------------------------------------------------------------
# load_aperture_images() helper end-to-end (PIL-backed PNG round-trip)
# ---------------------------------------------------------------------------

def test_load_aperture_images_decodes_png_and_invalidates_cache(tmp_path):
    pil = pytest.importorskip("PIL")
    from PIL import Image

    # Build a tiny lens file referencing a 4x4 PNG mask we write next to it.
    mask_png = tmp_path / "mask.png"
    arr = (np.array([[255, 255,   0,   0],
                     [255, 255,   0,   0],
                     [  0,   0, 255, 255],
                     [  0,   0, 255, 255]], dtype=np.uint8))
    Image.fromarray(arr, mode="L").save(mask_png)

    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "mask.png",
        "semi_diameter": 5.0,
    })
    lens_path = tmp_path / "image_stop.lens"
    with open(lens_path, "w") as f:
        json.dump(doc, f)

    lens = ghostlight.OpticalSystem.load(str(lens_path))
    idx = _find_stop_index(lens)
    img_before = lens.aperture_images[idx]
    assert np.asarray(img_before.pixels).size == 0

    lens.load_aperture_images(root_dir=str(tmp_path))
    img_after = lens.aperture_images[idx]
    loaded = np.asarray(img_after.pixels)
    assert loaded.shape == (4, 4)
    # Greyscale 0/255 → 0.0/1.0 after the helper's normalisation step.
    assert loaded.min() == pytest.approx(0.0)
    assert loaded.max() == pytest.approx(1.0)
    # Non-zero entries match the top-left / bottom-right quadrants.
    assert loaded[0, 0] == pytest.approx(1.0)
    assert loaded[3, 3] == pytest.approx(1.0)
    assert loaded[0, 3] == pytest.approx(0.0)
    assert loaded[3, 0] == pytest.approx(0.0)


def test_load_aperture_images_is_idempotent(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    mask_png = tmp_path / "mask.png"
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8), mode="L").save(mask_png)
    doc = _doublet_doc_with_image_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "mask.png",
        "semi_diameter": 5.0,
    })
    lens_path = tmp_path / "x.lens"
    with open(lens_path, "w") as f:
        json.dump(doc, f)

    lens = ghostlight.OpticalSystem.load(str(lens_path))
    lens.load_aperture_images(root_dir=str(tmp_path))
    lens.load_aperture_images(root_dir=str(tmp_path))  # second call: no-op
    idx = _find_stop_index(lens)
    loaded = np.asarray(lens.aperture_images[idx].pixels)
    assert loaded.shape == (2, 2)


# ---------------------------------------------------------------------------
# Cache-key invalidation
# ---------------------------------------------------------------------------

def test_lens_cache_invalidates_on_aperture_semi_diameter_change(simple_lens):
    cal1 = simple_lens.calibration()
    for s in simple_lens.surfaces:
        if s.is_stop:
            s.aperture_semi_diameter = 7.0
            break
    cal2 = simple_lens.calibration()
    assert cal1 is not cal2
