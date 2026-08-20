"""ID-buffer picking helpers.

Encodes scene object IDs into an RGBA8 fragment so a single
:func:`glReadPixels` call resolves a click into ``(tag, element_index,
surface_index, face_index)``.

The actual FBO is owned by the widget; this module is pure encoding logic
plus tag-byte constants so picking shaders and the widget agree on the wire
format.

Wire format
-----------
For element-body picks (``tag = TAG_ELEMENT_BODY``):

* ``R + G << 8`` holds ``element_index + 1`` (16 bits → up to 65534 elements).
* ``B`` holds ``surface_index + 1`` (8 bits → up to 254 surfaces), or 0 when
  the region has no specific surface attribution.

The ``+ 1`` offset means an all-zero pixel (the cleared state of an empty
FBO) decodes as "nothing here" rather than "element 0 / surface 0".

For view-cube picks (``tag = TAG_VIEW_CUBE``): ``R`` holds ``face_index + 1``;
G and B unused.
"""

from __future__ import annotations

from typing import Optional, Tuple


# Tag byte semantics (alpha channel of the picking buffer)
TAG_ELEMENT_BODY = 0x00
TAG_VIEW_CUBE    = 0xFF


def encode_element_surface_id(
    element_index: int,
    surface_index: Optional[int] = None,
    tag: int = TAG_ELEMENT_BODY,
) -> Tuple[float, float, float, float]:
    """Pack a 16-bit element index + 8-bit surface index + 8-bit tag.

    Both indices are stored ``+ 1`` so the cleared-FBO pixel ``(0, 0, 0, 0)``
    decodes to "nothing here".  ``surface_index = None`` writes ``B = 0``,
    which decodes back to ``None``.
    """
    el = max(0, min(0xFFFE, int(element_index))) + 1
    if surface_index is None:
        surf = 0
    else:
        surf = max(0, min(0xFE, int(surface_index))) + 1
    tag_v = max(0, min(0xFF, int(tag)))
    r = (el & 0xFF) / 255.0
    g = ((el >> 8) & 0xFF) / 255.0
    b = surf / 255.0
    a = tag_v / 255.0
    return r, g, b, a


def encode_element_id(element_index: int, tag: int = TAG_ELEMENT_BODY) -> Tuple[float, float, float, float]:
    """Backwards-compatible element-only encoder.

    Equivalent to :func:`encode_element_surface_id` with no surface index.
    """
    return encode_element_surface_id(element_index, surface_index=None, tag=tag)


def encode_view_cube_face(face_index: int) -> Tuple[float, float, float, float]:
    """View-cube faces use TAG_VIEW_CUBE in alpha; face index goes in R.

    Stored ``face_index + 1`` so an empty pixel never decodes as face 0.
    """
    f = max(0, min(0xFE, int(face_index))) + 1
    return f / 255.0, 0.0, 0.0, TAG_VIEW_CUBE / 255.0


def decode_pixel(rgba: Tuple[int, int, int, int]) -> dict:
    """Inverse of the encoders.

    Returns ``{"tag", "element_index", "surface_index", "face_index",
    "is_empty"}``.  ``is_empty`` is ``True`` and other fields are ``None``
    when the pixel was cleared (no geometry rendered there).
    """
    r, g, b, a = (int(v) & 0xFF for v in rgba)
    tag = a

    if tag == TAG_ELEMENT_BODY and r == 0 and g == 0 and b == 0:
        return {
            "tag": tag,
            "element_index": None,
            "surface_index": None,
            "face_index": None,
            "is_empty": True,
        }

    element_index: Optional[int] = None
    surface_index: Optional[int] = None
    face_index: Optional[int] = None

    if tag == TAG_VIEW_CUBE:
        face_index = r - 1
    elif tag == TAG_ELEMENT_BODY:
        el_raw = r | (g << 8)
        if el_raw > 0:
            element_index = el_raw - 1
        if b > 0:
            surface_index = b - 1

    return {
        "tag": tag,
        "element_index": element_index,
        "surface_index": surface_index,
        "face_index": face_index,
        "is_empty": False,
    }
