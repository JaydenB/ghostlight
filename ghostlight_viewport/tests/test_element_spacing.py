"""Element spacing is not double-offset.

After the fix, ``build_element_mesh`` must NOT apply ``element.position`` —
the cumulative ``surface.z`` set by ghostlight's loader is the only positional
truth.  This test loads each element from the example doublet and verifies
that the mesh's centroid Z matches the mean of the resolved surfaces'
``z`` values within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import load_example_lens, load_example_elements


def test_element_centroids_match_surface_z():
    from ghostlight_viewport import geometry
    lens = load_example_lens()
    elements = load_example_elements()
    for el in elements:
        mesh = geometry.build_element_mesh(el, lens)
        if mesh.vertex_count == 0:
            continue
        indices = el.resolve_surfaces(lens)
        surface_zs = [float(lens.surfaces[i].z) for i in indices]
        expect_z = float(np.mean(surface_zs))
        got_z = float(mesh.vertices[:, 2].mean())
        # Mesh centroid is biased by side-wall geometry, so allow a few mm.
        assert abs(got_z - expect_z) < 5.0, (
            f"{el.name}: centroid z={got_z:.3f} vs expected {expect_z:.3f}; "
            f"element transform probably re-applied (double-offset)"
        )


def test_element_position_not_double_applied():
    """When ``element.position.z`` is non-zero but the underlying surfaces are
    already at cumulative z, the mesh must NOT shift by that position."""
    from ghostlight_viewport import geometry
    lens = load_example_lens()
    elements = load_example_elements()
    # Aperture stop in the doublet has element.transform.z = 12.5; the
    # underlying surface has surface.z = 8.1 (cumulative).  Adding 12.5 on top
    # would give a centroid near 20.6, which is wrong.
    stop = next((el for el in elements if el.kind.name == "STOP"), None)
    if stop is None:
        pytest.skip("no stop element in example lens")
    mesh = geometry.build_element_mesh(stop, lens)
    indices = stop.resolve_surfaces(lens)
    expect_z = float(lens.surfaces[indices[0]].z)
    got_z = float(mesh.vertices[:, 2].mean())
    assert abs(got_z - expect_z) < 1.0
