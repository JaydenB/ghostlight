"""Picking encoding never returns a spurious element 0 on empty pixels.

The bug was that an FBO pixel of ``(0,0,0,0)`` decoded to ``tag=0,
element_index=0`` and the click handler immediately selected element 0.
After the +1 offset fix, an empty pixel must decode as "no element".
"""

from __future__ import annotations


def test_empty_pixel_decodes_as_no_element():
    from ghostlight_viewport import picking
    info = picking.decode_pixel((0, 0, 0, 0))
    assert info["is_empty"] is True
    assert info["element_index"] is None
    assert info["face_index"] is None


def test_element_index_round_trips():
    from ghostlight_viewport import picking
    # Range capped at 65534 — bits 16-23 now carry surface_index, leaving
    # 16 bits for element_index (one slot reserved for the empty-pixel
    # signal, so max usable is 0xFFFE).
    for idx in (0, 1, 17, 255, 256, 65534):
        r, g, b, a = picking.encode_element_id(idx)
        info = picking.decode_pixel((
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
            int(round(a * 255)),
        ))
        assert info["element_index"] == idx
        assert info["surface_index"] is None
        assert info["tag"] == picking.TAG_ELEMENT_BODY
        assert info["is_empty"] is False


def test_element_surface_id_round_trips():
    from ghostlight_viewport import picking
    for el_idx in (0, 1, 17, 4000):
        for surf_idx in (None, 0, 1, 9, 254):
            r, g, b, a = picking.encode_element_surface_id(el_idx, surf_idx)
            info = picking.decode_pixel((
                int(round(r * 255)),
                int(round(g * 255)),
                int(round(b * 255)),
                int(round(a * 255)),
            ))
            assert info["element_index"] == el_idx
            assert info["surface_index"] == surf_idx
            assert info["tag"] == picking.TAG_ELEMENT_BODY
            assert info["is_empty"] is False


def test_view_cube_face_round_trips():
    from ghostlight_viewport import picking
    for face in range(6):
        r, g, b, a = picking.encode_view_cube_face(face)
        info = picking.decode_pixel((
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
            int(round(a * 255)),
        ))
        assert info["face_index"] == face
        assert info["tag"] == picking.TAG_VIEW_CUBE
        assert info["is_empty"] is False
