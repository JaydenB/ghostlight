"""Angular sample-point generators for extended (area) light sources.

An extended source is a shape in *angle space*: a patch of collimated source
directions around a center direction (the tracer models sources at infinity,
so spatial extent is expressed as angular extent — e.g. the sun is a disk of
angular radius ~0.265 degrees).

Samplers return an (N, 3) float32 array of [d_angle_x, d_angle_y, weight]
rows for OpticalSystem.render_source_flare.  Offsets are radians; weights are
uniform and sum to 1, so the render is the area-average of point flares and
total intensity is independent of shape size and sample count.  Shapes:
point, disk/ellipse, rectangle, square, and regular polygon; rotate_offsets
applies an orientation to any of them.

Points come from the Halton sequence (bases 2, 3), so any prefix of the rows
is itself well distributed over the shape: callers may render the rows in
chunks and display the running weighted mean as a progressive preview.
"""

from __future__ import annotations

import numpy as np


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """Van der Corput radical inverse of integer indices in the given base."""
    result = np.zeros(indices.shape, dtype=np.float64)
    frac = 1.0 / base
    idx = indices.copy()
    while idx.max(initial=0) > 0:
        result += (idx % base) * frac
        idx //= base
        frac /= base
    return result


def _halton_2d(n: int) -> np.ndarray:
    """(n, 2) Halton points in [0, 1)^2, starting at index 1."""
    indices = np.arange(1, n + 1, dtype=np.int64)
    return np.stack(
        [_radical_inverse(indices, 2), _radical_inverse(indices, 3)], axis=1
    )


def _with_weights(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    n = dx.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    out[:, 0] = dx
    out[:, 1] = dy
    out[:, 2] = 1.0 / n
    return out


def sample_point() -> np.ndarray:
    """The degenerate source: one full-weight sample at the center."""
    return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)


def sample_disk(radius_x: float, radius_y: float | None = None, n: int = 64) -> np.ndarray:
    """Uniform-by-area samples over a disk (or ellipse) of angular radius radians.

    radius_y defaults to radius_x (a circle); differing radii give an ellipse.
    """
    if n < 1:
        raise ValueError("sample_disk: n must be >= 1")
    if radius_y is None:
        radius_y = radius_x
    if n == 1:
        return sample_point()
    uv = _halton_2d(n)
    r = np.sqrt(uv[:, 0])
    theta = 2.0 * np.pi * uv[:, 1]
    return _with_weights(radius_x * r * np.cos(theta), radius_y * r * np.sin(theta))


def sample_rect(half_width: float, half_height: float, n: int = 64) -> np.ndarray:
    """Uniform-by-area samples over a rectangle of angular half-extents radians."""
    if n < 1:
        raise ValueError("sample_rect: n must be >= 1")
    if n == 1:
        return sample_point()
    uv = _halton_2d(n)
    return _with_weights(
        half_width * (uv[:, 0] * 2.0 - 1.0),
        half_height * (uv[:, 1] * 2.0 - 1.0),
    )


def sample_square(half_size: float, n: int = 64) -> np.ndarray:
    """Uniform-by-area samples over a square of angular half-extent radians."""
    return sample_rect(half_size, half_size, n)


def polygon_vertices(radius: float, n_sides: int, rotation: float = 0.0) -> np.ndarray:
    """Corner offsets (n_sides, 2) of a regular polygon of circumradius radians.

    Vertex 0 points along -Y (up on screen) at ``rotation`` 0; ``rotation`` is
    a CCW angle in radians. Callers can use the same vertices for sampling and
    outlines.
    """
    n_sides = max(3, int(n_sides))
    base = -np.pi / 2.0 + float(rotation)
    ang = base + 2.0 * np.pi * np.arange(n_sides) / n_sides
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def sample_polygon(radius: float, n_sides: int = 6, n: int = 64) -> np.ndarray:
    """Uniform-by-area samples over a regular polygon of circumradius radians.

    Built axis-aligned (see :func:`polygon_vertices`); apply :func:`rotate_offsets`
    for orientation.  The polygon is split into its ``n_sides`` congruent
    center triangles; samples are dealt round-robin across them so any prefix
    stays balanced over the whole shape (progressive-preview friendly).
    """
    if n < 1:
        raise ValueError("sample_polygon: n must be >= 1")
    n_sides = max(3, int(n_sides))
    if n == 1:
        return sample_point()
    verts = polygon_vertices(radius, n_sides, 0.0)
    uv = _halton_2d(n)
    tri = np.arange(n) % n_sides
    # Uniform sampling inside triangle (O, A, B): p = sqrt(u)*(1-v)*A + sqrt(u)*v*B.
    su = np.sqrt(uv[:, 0])
    v = uv[:, 1]
    a = verts[tri]
    b = verts[(tri + 1) % n_sides]
    dx = su * (1.0 - v) * a[:, 0] + su * v * b[:, 0]
    dy = su * (1.0 - v) * a[:, 1] + su * v * b[:, 1]
    return _with_weights(dx, dy)


def rotate_offsets(offsets: np.ndarray, rotation: float) -> np.ndarray:
    """Rotate the angular offsets of an (N, 3) array by ``rotation`` radians.

    Weights are untouched.  A no-op (returns the input) when rotation is 0.
    """
    rotation = float(rotation)
    if rotation == 0.0:
        return offsets
    c, s = np.cos(rotation), np.sin(rotation)
    out = np.array(offsets, dtype=np.float32, copy=True)
    dx, dy = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = dx * c - dy * s
    out[:, 1] = dx * s + dy * c
    return out


def chunk_offsets(offsets: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """Split an (N, 3) offsets array into row chunks for progressive rendering.

    Each chunk keeps its rows' global weights, so summing the per-chunk render
    outputs reproduces the single-call result; dividing a partial sum by the
    weight rendered so far gives a correctly exposed progressive preview.
    """
    if chunk_size < 1:
        raise ValueError("chunk_offsets: chunk_size must be >= 1")
    return [offsets[i : i + chunk_size] for i in range(0, offsets.shape[0], chunk_size)]
