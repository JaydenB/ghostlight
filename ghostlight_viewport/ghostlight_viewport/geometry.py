"""Mesh generation for lens surfaces and lofted glass elements.

All routines are pure-numpy.  The output of each surface tessellator is a
``Mesh`` with float32 ``vertices`` (Nx3 positions), float32 ``normals``
(Nx3), and uint32 ``indices`` (M, triangles).  Lofting an element produces
one closed solid mesh per glass region, with a side-wall tube connecting
each pair of adjacent surfaces.

Conventions
-----------
* Optical axis is +Z.  Each surface's apex sits at ``surface.z``.
* Sag bulges away from the apex along +Z for a positive radius.
* Surface normals point *outward* from the glass — toward -Z on a front
  surface, toward +Z on a rear surface, regardless of curvature sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


N_RADIAL_DEFAULT = 32
N_AZIMUTH_DEFAULT = 64


@dataclass
class Mesh:
    """Vertex/normal/index triplet for a tessellated surface or solid.

    ``kinds`` is a per-vertex float used by the lens shader to colour the
    connecting side wall differently from the optical cap surfaces: 0.0 for
    cap vertices, 1.0 for side-wall vertices.  Constructors that omit it get
    an all-zero array sized to match ``vertices``.
    """
    vertices: np.ndarray   # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    indices: np.ndarray    # (M,)  uint32, triangles
    kinds: Optional[np.ndarray] = None  # (N,) float32; defaults to zeros

    def __post_init__(self) -> None:
        if self.kinds is None:
            self.kinds = np.zeros((self.vertices.shape[0],), dtype=np.float32)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)


# ---------------------------------------------------------------------------
# Sag formula
# ---------------------------------------------------------------------------

def sag(r: np.ndarray, radius: float, conic_k: float, asphere_terms: np.ndarray) -> np.ndarray:
    """Standard optical sag.

    ``asphere_terms[i]`` multiplies r**(2*(i+2)), i.e. r^4, r^6, r^8, ...
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    if abs(radius) > 1e-9:
        c = 1.0 / float(radius)
        k1 = 1.0 + float(conic_k)
        disc = 1.0 - k1 * c * c * r * r
        disc = np.clip(disc, 1e-12, None)
        out = (c * r * r) / (1.0 + np.sqrt(disc))
    for i, a in enumerate(np.asarray(asphere_terms, dtype=np.float64).ravel()):
        if a == 0.0:
            continue
        order = 2 * (i + 2)
        out = out + a * np.power(r, order)
    return out


def dsag_dr(r: np.ndarray, radius: float, conic_k: float, asphere_terms: np.ndarray) -> np.ndarray:
    """Analytic dsag/dr — used for surface normals."""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    if abs(radius) > 1e-9:
        c = 1.0 / float(radius)
        k1 = 1.0 + float(conic_k)
        disc = 1.0 - k1 * c * c * r * r
        disc = np.clip(disc, 1e-12, None)
        sqrt_disc = np.sqrt(disc)
        out = (c * r) / sqrt_disc
    for i, a in enumerate(np.asarray(asphere_terms, dtype=np.float64).ravel()):
        if a == 0.0:
            continue
        order = 2 * (i + 2)
        out = out + a * order * np.power(r, order - 1)
    return out


# ---------------------------------------------------------------------------
# Surface property extraction
# ---------------------------------------------------------------------------

# Enum-name lookup helpers.  Surface::form, ::cyl_axis, and ::aperture_shape
# are declared as plain ``int`` in the C++ struct, so pybind11 exposes them as
# Python ints rather than enum instances.  ``getattr(form, "name", default)``
# silently falls back to the default on real loaded data, which would
# tessellate every cylindrical surface as a sphere.  Use these
# helpers to compare against the canonical names regardless of whether the
# binding hands back an int or an enum.
_FORM_NAMES = {0: "SPHERE", 1: "ASPHERE", 2: "CYLINDRICAL"}
_CYL_AXIS_NAMES = {0: "AXIS_X", 1: "AXIS_Y"}
_APERTURE_SHAPE_NAMES = {0: "CIRCLE", 1: "POLYGON", 2: "IMAGE"}


def _enum_name(value, table: dict[int, str], default: str) -> str:
    if value is None:
        return default
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    try:
        return table.get(int(value), default)
    except (TypeError, ValueError):
        return default


def _form_name(surface) -> str:
    return _enum_name(getattr(surface, "form", None), _FORM_NAMES, "SPHERE")


def _cyl_axis_name(surface) -> str:
    return _enum_name(getattr(surface, "cyl_axis", None), _CYL_AXIS_NAMES, "AXIS_Y")


def _aperture_shape_name(surface) -> str:
    return _enum_name(
        getattr(surface, "aperture_shape", None), _APERTURE_SHAPE_NAMES, "CIRCLE"
    )


def _asphere_array(surface) -> np.ndarray:
    """Return the active aspheric coefficient list as a numpy array."""
    terms = getattr(surface, "asphere_terms", None)
    n = int(getattr(surface, "n_asphere_terms", 0))
    if terms is None or n <= 0:
        return np.zeros((0,), dtype=np.float64)
    try:
        arr = np.array([float(terms[i]) for i in range(n)], dtype=np.float64)
    except (IndexError, TypeError):
        return np.zeros((0,), dtype=np.float64)
    return arr


def _aperture_radial_test(theta: np.ndarray, surface) -> np.ndarray:
    """For each azimuth, return the maximum radius allowed by the aperture.

    Used both to clip the spherical/aspheric ring rim and to carve the iris
    hole for stops.
    """
    semi = float(getattr(surface, "semi_aperture", 1.0))
    aspect = float(getattr(surface, "aperture_aspect", 1.0)) or 1.0
    blades = int(getattr(surface, "aperture_blades", 0))
    rot = float(getattr(surface, "aperture_rotation_rad", 0.0))

    is_polygon = _aperture_shape_name(surface) == "POLYGON" and blades >= 3

    # Aspect ratio applied as ellipse along X
    cos = np.cos(theta)
    sin = np.sin(theta)
    r_ellipse = semi / np.sqrt((cos / aspect) ** 2 + sin ** 2)

    if not is_polygon:
        return r_ellipse

    sector = 2.0 * math.pi / blades
    apothem = math.cos(math.pi / blades) * semi
    sec = np.mod(theta - rot, sector)
    half = sector * 0.5
    angle_from_edge = np.abs(sec - half)
    r_poly = apothem / np.cos(angle_from_edge)
    return np.minimum(r_ellipse, r_poly)


# ---------------------------------------------------------------------------
# Surface pose (canonical frame -> world)
#
# EVERY builder that emits vertices for a surface must go through these.
# Open-coding the placement lets the copies drift — a cap that applies the
# rotation while the side-wall rims and the cylinder cap apply only decenter
# tears a tilted element's walls off the caps they join. One implementation,
# five callers.
# ---------------------------------------------------------------------------

def surface_rotation(surface) -> Optional[np.ndarray]:
    """The surface's local->world 3x3, or ``None`` when it's identity.

    ``None`` is the fast path, not an error — it lets callers skip the matmul
    for the overwhelmingly common untilted case.
    """
    rot_flat = getattr(surface, "rot", None)
    if rot_flat is None:
        return None
    try:
        rot_m = np.array(
            [float(rot_flat[i]) for i in range(9)], dtype=np.float64
        ).reshape(3, 3)
    except (IndexError, TypeError, ValueError):
        return None
    if np.allclose(rot_m, np.eye(3)):
        return None
    return rot_m


def surface_axis(surface) -> np.ndarray:
    """World-space direction of the surface's local +Z (its optical axis)."""
    rot_m = surface_rotation(surface)
    if rot_m is None:
        return np.array([0.0, 0.0, 1.0])
    return rot_m @ np.array([0.0, 0.0, 1.0])


def place_surface_vertices(local: np.ndarray, surface) -> np.ndarray:
    """Map canonical-frame vertices onto the surface's world pose.

    ``local`` is ``(n, 3)`` with the vertex at the origin and sag along +Z —
    i.e. NOT yet offset by ``surface.z``. The mapping is the inverse of the
    world->canonical transform the tracer applies in ``intersect_surface``:

        world = rot @ local + (decenter_x, decenter_y, z)

    Rotation first, then the offset: the vertex is the centre of rotation, so
    rotating after translating would swing the whole surface around the world
    origin instead of turning it in place.
    """
    verts = np.asarray(local, dtype=np.float64)
    rot_m = surface_rotation(surface)
    if rot_m is not None:
        verts = verts @ rot_m.T
    else:
        verts = verts.copy()
    verts[:, 0] += float(getattr(surface, "decenter_x", 0.0))
    verts[:, 1] += float(getattr(surface, "decenter_y", 0.0))
    verts[:, 2] += float(getattr(surface, "z", 0.0))
    return verts


def place_surface_normals(local: np.ndarray, surface) -> np.ndarray:
    """Rotate canonical-frame normals into world space.

    The rotation is orthonormal, so the plain matrix applies — no inverse
    transpose needed — and lengths are preserved.
    """
    rot_m = surface_rotation(surface)
    if rot_m is None:
        return np.asarray(local, dtype=np.float32)
    return (np.asarray(local, dtype=np.float64) @ rot_m.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Tessellators
# ---------------------------------------------------------------------------

def tessellate_sphere_or_asphere_cap(
    surface,
    *,
    outward_sign: float = -1.0,
    n_radial: int = N_RADIAL_DEFAULT,
    n_theta: int = N_AZIMUTH_DEFAULT,
) -> Mesh:
    """Revolve the surface profile to make a cap mesh.

    ``outward_sign`` is +1 for back surfaces (normal points toward +Z) and
    -1 for front surfaces (normal toward -Z), and is applied to the analytic
    normal so glass shells light correctly without back-face culling.

    Vertices are emitted in the surface's canonical frame (apex at the origin,
    sag along +Z) and then placed by :func:`place_surface_vertices`.
    """
    radius = float(getattr(surface, "radius", 0.0))
    conic_k = float(getattr(surface, "conic_k", 0.0))
    asphere = _asphere_array(surface)
    semi = float(getattr(surface, "semi_aperture", 1.0))

    if semi <= 0.0 or n_radial < 2 or n_theta < 6:
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    max_r_per_theta = _aperture_radial_test(theta, surface)

    # Build ring rows from r=0 outward; each ring has its own r_per_theta.
    radial_frac = np.linspace(0.0, 1.0, n_radial)
    n_ring = n_theta

    vertex_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []

    # Apex vertex (r=0): one shared vertex with normal = (0,0,outward_sign)
    apex = np.array([[0.0, 0.0, sag(np.array([0.0]), radius, conic_k, asphere)[0]]])
    apex_normal = np.array([[0.0, 0.0, outward_sign]])

    for ri, frac in enumerate(radial_frac):
        if ri == 0:
            continue  # apex handled separately
        r_per_theta = frac * max_r_per_theta
        sag_vals = sag(r_per_theta, radius, conic_k, asphere)
        x = r_per_theta * np.cos(theta)
        y = r_per_theta * np.sin(theta)
        z = sag_vals
        ring = np.stack([x, y, z], axis=1)

        # Normals: gradient is (dsag/dr * cos, dsag/dr * sin, -1) on the
        # front; we want it scaled by outward_sign so it points outward.
        slope = dsag_dr(r_per_theta, radius, conic_k, asphere)
        nx = slope * np.cos(theta)
        ny = slope * np.sin(theta)
        nz = np.full_like(slope, -1.0)
        n = np.stack([nx, ny, nz], axis=1)
        n_len = np.linalg.norm(n, axis=1, keepdims=True)
        n_len = np.where(n_len > 1e-12, n_len, 1.0)
        n = (n / n_len) * outward_sign

        vertex_rows.append(ring)
        normal_rows.append(n)

    vertices = np.vstack([apex] + vertex_rows).astype(np.float32)
    normals = np.vstack([apex_normal] + normal_rows).astype(np.float32)

    # Canonical frame -> world (decenter + tilt + axial position).
    vertices = place_surface_vertices(vertices, surface).astype(np.float32)
    normals = place_surface_normals(normals, surface)

    # Triangle indices
    # Apex fan (apex_idx=0) -> first ring
    apex_idx = 0
    first_ring_start = 1
    indices: list[int] = []
    # Apex fan triangles
    for j in range(n_ring):
        a = first_ring_start + j
        b = first_ring_start + (j + 1) % n_ring
        # Winding: for outward_sign=-1, we want CCW seen from -Z; for +1, CCW from +Z.
        if outward_sign < 0.0:
            indices.extend([apex_idx, b, a])
        else:
            indices.extend([apex_idx, a, b])
    # Quad rings.  Winding must match the apex fan: CCW seen from the
    # OUTWARD direction.  In world XY (looking from +Z), an outward=+Z cap
    # wants CCW, which for these vertex labels is [a, c, b] + [a, d, c]
    # (the [a, b, c] path winds CW because r2 > r1 reverses the sweep
    # direction).  The outward=-Z branch is the mirror.  Back-face culling
    # was off until this was fixed, hiding the inconsistency with the apex
    # fan; with culling on, the swapped order put a hole at every cap
    # ring while the apex fan rendered correctly.
    for r_idx in range(1, n_radial - 1):
        ring0 = 1 + (r_idx - 1) * n_ring
        ring1 = 1 + (r_idx) * n_ring
        for j in range(n_ring):
            a = ring0 + j
            b = ring0 + (j + 1) % n_ring
            c = ring1 + (j + 1) % n_ring
            d = ring1 + j
            if outward_sign < 0.0:
                indices.extend([a, b, c, a, c, d])
            else:
                indices.extend([a, c, b, a, d, c])

    return Mesh(
        vertices=vertices,
        normals=normals,
        indices=np.array(indices, dtype=np.uint32),
    )


def _signed_slope(r_signed: np.ndarray, radius: float, conic_k: float,
                   asphere: np.ndarray) -> np.ndarray:
    """``dsag/dr`` taken at ``|r|`` and signed by the sign of ``r``.

    ghostlight's tracer evaluates ``sag(|r|)`` symmetrically and the geometric
    normal at radius ``r`` has slope ``dsag/d|r| * sign(r)``.  At ``r == 0`` the
    slope is exactly zero (no kink) since ``dsag/dr`` is finite there.
    """
    r = np.asarray(r_signed, dtype=np.float64)
    mag = np.abs(r)
    s = dsag_dr(mag, radius, conic_k, asphere)
    return s * np.sign(r)


def tessellate_cylinder(
    surface,
    *,
    outward_sign: float = -1.0,
    n_radial: int = N_RADIAL_DEFAULT,
    n_extrude: int = N_RADIAL_DEFAULT,
) -> Mesh:
    """Extrude the 1-D sag profile along the cylinder axis.

    Cylindrical surfaces are flat along ``cyl_axis`` and curved in the
    perpendicular direction.  The aperture clipping uses the same
    polygon / aspect rules as :func:`tessellate_sphere_or_asphere_cap` —
    truncating the rim of the rectangular footprint via the curved-axis
    half-width and the flat-axis half-height.
    """
    radius = float(getattr(surface, "radius", 0.0))
    conic_k = float(getattr(surface, "conic_k", 0.0))
    asphere = _asphere_array(surface)
    semi = float(getattr(surface, "semi_aperture", 1.0))
    aspect = float(getattr(surface, "aperture_aspect", 1.0)) or 1.0

    along_y = _cyl_axis_name(surface) == "AXIS_Y"

    if semi <= 0.0 or n_radial < 2 or n_extrude < 2:
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )

    # Footprint half-extents — aspect stretches X regardless of cyl_axis;
    # ``semi_aperture`` sets the un-stretched extent.  Same convention used
    # by ``_cylinder_rim_vertices``.
    x_max = semi * aspect
    y_max = semi

    # Profile across the curved direction (signed so we get full ± domain)
    if along_y:
        r = np.linspace(-x_max, x_max, n_radial)         # X is curved
        extrude = np.linspace(-y_max, y_max, n_extrude)  # Y is flat
    else:
        r = np.linspace(-y_max, y_max, n_radial)         # Y is curved
        extrude = np.linspace(-x_max, x_max, n_extrude)  # X is flat

    profile_sag = sag(np.abs(r), radius, conic_k, asphere)
    profile_slope = _signed_slope(r, radius, conic_k, asphere)

    if along_y:
        # X = r, Y = extrude, Z = sag(|r|)
        x_grid, y_grid = np.meshgrid(r, extrude, indexing="xy")
        z_grid = np.tile(profile_sag, (n_extrude, 1))
        slope_grid = np.tile(profile_slope, (n_extrude, 1))
        nx = slope_grid
        ny = np.zeros_like(slope_grid)
        nz = np.full_like(slope_grid, -1.0)
    else:
        x_grid, y_grid = np.meshgrid(extrude, r, indexing="xy")
        z_grid = np.tile(profile_sag[:, None], (1, n_extrude))
        slope_grid = np.tile(profile_slope[:, None], (1, n_extrude))
        nx = np.zeros_like(slope_grid)
        ny = slope_grid
        nz = np.full_like(slope_grid, -1.0)

    # Polygonal aperture: project each vertex's (x, y) through the same
    # apothem test used for round surfaces, clipping the rim.  Vertices
    # outside the polygon stay on the rectangle but their world Z is set to
    # NaN-equivalent so we can drop their triangles below.  For the moment
    # the rectangular footprint stays — polygonal anamorphic apertures are
    # rare but the math is in place.
    blades = int(getattr(surface, "aperture_blades", 0))
    rot = float(getattr(surface, "aperture_rotation_rad", 0.0))
    if _aperture_shape_name(surface) == "POLYGON" and blades >= 3:
        sector = 2.0 * math.pi / blades
        apothem = math.cos(math.pi / blades) * semi
        # Test against the polygon in (x/aspect, y) ellipse-stretched space.
        sx = x_grid / aspect
        sy = y_grid
        theta = np.arctan2(sy, sx)
        rad = np.sqrt(sx * sx + sy * sy)
        sec = np.mod(theta - rot, sector)
        half = sector * 0.5
        angle_from_edge = np.abs(sec - half)
        r_poly = apothem / np.cos(angle_from_edge)
        outside = rad > r_poly
        # Snap outside vertices radially inward to the polygon boundary so
        # the rim follows the polygon shape.  Avoid divide-by-zero at origin.
        safe_rad = np.where(rad > 1e-9, rad, 1.0)
        scale = np.where(outside, r_poly / safe_rad, 1.0)
        sx_new = sx * scale
        sy_new = sy * scale
        x_grid = sx_new * aspect
        y_grid = sy_new
        # Recompute z for the moved points (along the curved direction).
        if along_y:
            new_r = x_grid          # signed curved-direction position
        else:
            new_r = y_grid
        z_grid = sag(np.abs(new_r), radius, conic_k, asphere)
        slope_signed = _signed_slope(new_r, radius, conic_k, asphere)
        if along_y:
            nx = slope_signed
            ny = np.zeros_like(slope_signed)
            nz = np.full_like(slope_signed, -1.0)
        else:
            nx = np.zeros_like(slope_signed)
            ny = slope_signed
            nz = np.full_like(slope_signed, -1.0)

    vertices = np.stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1
    ).astype(np.float64)
    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    n_len = np.where(n_len > 1e-12, n_len, 1.0)
    normals = (normals / n_len) * outward_sign

    # Canonical frame -> world. Decenter alone is not enough: a tilted
    # anamorphic element would draw its cap flat while the tracer refracts
    # off a turned one.
    vertices = place_surface_vertices(vertices, surface)
    normals = place_surface_normals(normals, surface)

    indices: list[int] = []
    n_cols = n_radial if along_y else n_extrude
    n_rows = n_extrude if along_y else n_radial
    for j in range(n_rows - 1):
        for i in range(n_cols - 1):
            a = j * n_cols + i
            b = j * n_cols + i + 1
            c = (j + 1) * n_cols + i + 1
            d = (j + 1) * n_cols + i
            if outward_sign < 0.0:
                indices.extend([a, c, b, a, d, c])
            else:
                indices.extend([a, b, c, a, c, d])

    return Mesh(
        vertices=vertices.astype(np.float32),
        normals=normals.astype(np.float32),
        indices=np.array(indices, dtype=np.uint32),
    )


def tessellate_surface(surface, *, outward_sign: float = -1.0) -> Mesh:
    """Dispatch on ``surface.form`` and tessellate as sphere/asphere/cylinder."""
    if _form_name(surface) == "CYLINDRICAL":
        return tessellate_cylinder(surface, outward_sign=outward_sign)
    return tessellate_sphere_or_asphere_cap(surface, outward_sign=outward_sign)


# ---------------------------------------------------------------------------
# Side wall + element solid
# ---------------------------------------------------------------------------

def _cylinder_rim_vertices(
    surface, n_theta: int = N_AZIMUTH_DEFAULT
) -> np.ndarray:
    """Rim vertices around a cylindrical surface's rectangular footprint.

    The cap mesh from :func:`tessellate_cylinder` is rectangular (or
    polygon-clipped); walking a polar ellipse here would inscribe a spherical
    silhouette inside that rectangle, so the side wall built between two
    cylindrical surfaces would render as a circular tube — making an
    anamorphic lens look spherical.  Instead we walk the rectangle perimeter
    at evenly spaced angles (preserving index correspondence with the
    spherical rim's polar parameterization) and evaluate cylindrical sag along
    only the curved axis at each rim point.
    """
    radius = float(getattr(surface, "radius", 0.0))
    conic_k = float(getattr(surface, "conic_k", 0.0))
    asphere = _asphere_array(surface)
    semi = float(getattr(surface, "semi_aperture", 1.0))
    aspect = float(getattr(surface, "aperture_aspect", 1.0)) or 1.0

    along_y = _cyl_axis_name(surface) == "AXIS_Y"

    x_max = semi * aspect
    y_max = semi

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos = np.cos(theta)
    sin = np.sin(theta)
    abs_cos = np.maximum(np.abs(cos), 1e-12)
    abs_sin = np.maximum(np.abs(sin), 1e-12)
    r = np.minimum(x_max / abs_cos, y_max / abs_sin)
    x = r * cos
    y = r * sin

    blades = int(getattr(surface, "aperture_blades", 0))
    rot = float(getattr(surface, "aperture_rotation_rad", 0.0))
    if _aperture_shape_name(surface) == "POLYGON" and blades >= 3:
        sector = 2.0 * math.pi / blades
        apothem = math.cos(math.pi / blades) * semi
        sx = x / aspect
        sy = y
        theta_p = np.arctan2(sy, sx)
        rad = np.sqrt(sx * sx + sy * sy)
        sec = np.mod(theta_p - rot, sector)
        half = sector * 0.5
        angle_from_edge = np.abs(sec - half)
        r_poly = apothem / np.cos(angle_from_edge)
        safe_rad = np.where(rad > 1e-9, rad, 1.0)
        scale = np.where(rad > r_poly, r_poly / safe_rad, 1.0)
        x = sx * scale * aspect
        y = sy * scale

    r_curved = x if along_y else y
    z = sag(np.abs(r_curved), radius, conic_k, asphere)

    local = np.stack([x, y, z], axis=1)
    return place_surface_vertices(local, surface).astype(np.float32)


def _surface_rim_vertices(
    surface, n_theta: int = N_AZIMUTH_DEFAULT
) -> np.ndarray:
    """Return ``(n_theta, 3)`` float32 rim vertices around the surface aperture.

    Dispatches on ``surface.form`` so cylindrical surfaces walk their
    rectangular footprint instead of the spherical polar parameterization.
    """
    if _form_name(surface) == "CYLINDRICAL":
        return _cylinder_rim_vertices(surface, n_theta)

    radius = float(getattr(surface, "radius", 0.0))
    conic_k = float(getattr(surface, "conic_k", 0.0))
    asphere = _asphere_array(surface)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    r_per_theta = _aperture_radial_test(theta, surface)
    z = sag(r_per_theta, radius, conic_k, asphere)
    x = r_per_theta * np.cos(theta)
    y = r_per_theta * np.sin(theta)
    local = np.stack([x, y, z], axis=1)
    return place_surface_vertices(local, surface).astype(np.float32)


def _wall_radial_normals(
    ra: np.ndarray, rb: np.ndarray, surface
) -> np.ndarray:
    """Outward radial normals for a side wall spanning rims ``ra`` -> ``rb``.

    "Outward" means perpendicular to the element's own optical axis, not to
    world +Z — otherwise a tilted element's wall lights as though it were
    still upright. The component along the local axis is projected out, and
    the centre is the rim centroid rather than a point on the world axis, so
    a decentred element is handled too.
    """
    centers = ((ra + rb) * 0.5).astype(np.float64)
    axis = surface_axis(surface)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    radial = centers - centers.mean(axis=0, keepdims=True)
    radial -= np.outer(radial @ axis, axis)
    rlen = np.linalg.norm(radial, axis=1, keepdims=True)
    rlen = np.where(rlen > 1e-12, rlen, 1.0)
    return (radial / rlen).astype(np.float32)


def build_side_wall(
    surface_a, surface_b, n_theta: int = N_AZIMUTH_DEFAULT
) -> Mesh:
    """Quad-ring tube connecting the rims of two consecutive surfaces."""
    ra = _surface_rim_vertices(surface_a, n_theta)
    rb = _surface_rim_vertices(surface_b, n_theta)
    n = ra.shape[0]
    vertices = np.vstack([ra, rb])
    radial = _wall_radial_normals(ra, rb, surface_a)
    normals = np.vstack([radial, radial])

    indices: list[int] = []
    for j in range(n):
        a = j
        b = (j + 1) % n
        c = (j + 1) % n + n
        d = j + n
        indices.extend([a, b, c, a, c, d])

    return Mesh(
        vertices=vertices.astype(np.float32),
        normals=normals.astype(np.float32),
        indices=np.array(indices, dtype=np.uint32),
        kinds=np.ones((vertices.shape[0],), dtype=np.float32),
    )


def build_side_wall_halves(
    surface_a, surface_b, n_theta: int = N_AZIMUTH_DEFAULT
) -> tuple[Mesh, Mesh]:
    """Split the side-wall tube into halves attributed to each cap surface.

    The wall is bisected at the axial midpoint of each rim pair.  ``half_a``
    is the half closer to ``surface_a`` (front cap); ``half_b`` is closer to
    ``surface_b`` (back cap).  Used by surface-mode picking so a click on the
    wall snaps to the nearer cap's surface rather than a generic "wall hit".
    Visually identical to :func:`build_side_wall` when both halves are drawn
    together with the same shader uniforms.
    """
    ra = _surface_rim_vertices(surface_a, n_theta)
    rb = _surface_rim_vertices(surface_b, n_theta)
    mid = ((ra + rb) * 0.5).astype(np.float32)
    n = ra.shape[0]

    radial = _wall_radial_normals(ra, rb, surface_a)

    def _half(top: np.ndarray, bottom: np.ndarray) -> Mesh:
        verts = np.vstack([top, bottom]).astype(np.float32)
        norms = np.vstack([radial, radial]).astype(np.float32)
        idx: list[int] = []
        for j in range(n):
            a = j
            b = (j + 1) % n
            c = (j + 1) % n + n
            d = j + n
            idx.extend([a, b, c, a, c, d])
        return Mesh(
            vertices=verts,
            normals=norms,
            indices=np.array(idx, dtype=np.uint32),
            kinds=np.ones((verts.shape[0],), dtype=np.float32),
        )

    return _half(ra.astype(np.float32), mid), _half(mid, rb.astype(np.float32))


def build_iris(surface, ring_outer_scale: float = 1.5) -> Mesh:
    """Annular (or polygonal-hole) iris in the surface's plane.

    Extends from the aperture rim to ``ring_outer_scale * semi_aperture``.
    The inner hole follows the same shape used to clip lens rims.
    """
    semi = float(getattr(surface, "semi_aperture", 1.0))
    n_theta = 96
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    r_inner = _aperture_radial_test(theta, surface)
    r_outer = semi * float(ring_outer_scale)

    x_in = r_inner * np.cos(theta)
    y_in = r_inner * np.sin(theta)
    x_out = r_outer * np.cos(theta)
    y_out = r_outer * np.sin(theta)
    zero = np.zeros_like(theta)
    inner = np.stack([x_in, y_in, zero], axis=1)
    outer = np.stack([x_out, y_out, zero], axis=1)

    # The iris ring is flat in the stop's own plane, so it follows the stop's
    # pose like every other surface geometry — ignoring decenter and tilt
    # would leave a decentred stop's ring behind on the axis.
    vertices = place_surface_vertices(
        np.vstack([inner, outer]), surface
    ).astype(np.float32)
    normals = place_surface_normals(
        np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
                (vertices.shape[0], 1)),
        surface,
    )
    indices: list[int] = []
    for j in range(n_theta):
        a = j
        b = (j + 1) % n_theta
        c = (j + 1) % n_theta + n_theta
        d = j + n_theta
        indices.extend([a, b, c, a, c, d])

    return Mesh(
        vertices=vertices,
        normals=normals,
        indices=np.array(indices, dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# Element-level lofting
# ---------------------------------------------------------------------------

def merge_meshes(meshes: list[Mesh]) -> Mesh:
    if not meshes:
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )
    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    idxs: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    offset = 0
    for m in meshes:
        verts.append(m.vertices)
        norms.append(m.normals)
        idxs.append(m.indices + offset)
        kinds.append(m.kinds)
        offset += m.vertex_count
    return Mesh(
        vertices=np.concatenate(verts, axis=0),
        normals=np.concatenate(norms, axis=0),
        indices=np.concatenate(idxs, axis=0),
        kinds=np.concatenate(kinds, axis=0).astype(np.float32),
    )


def loft_glass_solid(surfaces: list) -> Mesh:
    """Build one closed solid spanning two surfaces (front + back + side wall).

    For a cemented multi-element group, call this once per consecutive pair
    and merge the resulting meshes.
    """
    if len(surfaces) < 2:
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )
    front = tessellate_surface(surfaces[0], outward_sign=-1.0)
    back = tessellate_surface(surfaces[1], outward_sign=+1.0)
    wall = build_side_wall(surfaces[0], surfaces[1])
    return merge_meshes([front, back, wall])


def apply_element_transform(
    mesh: Mesh,
    position: tuple[float, float, float],
    rotation_euler_deg: tuple[float, float, float],
) -> Mesh:
    """Apply a position + Euler-XYZ rotation to mesh vertices and normals."""
    px, py, pz = position
    rx, ry, rz = (math.radians(d) for d in rotation_euler_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx

    if mesh.vertex_count == 0:
        return mesh

    v = mesh.vertices.astype(np.float64) @ R.T
    v[:, 0] += float(px)
    v[:, 1] += float(py)
    v[:, 2] += float(pz)
    n = mesh.normals.astype(np.float64) @ R.T
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    nl = np.where(nl > 1e-12, nl, 1.0)
    n = n / nl
    return Mesh(
        vertices=v.astype(np.float32),
        normals=n.astype(np.float32),
        indices=mesh.indices,
        kinds=mesh.kinds,
    )


@dataclass
class SubmeshRegion:
    """One drawable region of an element sub-solid with surface attribution.

    A glass sub-solid decomposes into ``front_cap`` (the optical surface at
    the front), ``back_cap`` (at the back), ``wall_a`` (the half of the side
    wall closer to the front cap), and ``wall_b`` (closer to the back cap).
    A stop has a single ``iris`` region.

    ``surface_index`` is the **global** index into ``system.surfaces`` — the
    same index the optical-editor tree uses for its :class:`SurfaceNode`, so
    a viewport pick can drive a tree selection without translation.
    Wall halves carry the surface index of their nearer cap so a click on
    the wall snaps to that surface.

    """
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    kinds: np.ndarray
    surface_index: int
    is_cap: bool

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


@dataclass
class SubSolid:
    """One closed sub-solid of an element (singlet body / cemented sub-glass / iris).

    ``centroid`` is the mean of all region vertices, used as the depth-sort
    key in the lens and picking passes.

    ``rim_loops`` is a list of (N, 3) float32 closed polylines that the
    viewport draws as stroke outlines on top of the lens — typically one
    loop per surface rim (where a cap meets its side wall) plus the
    inner / outer rings of an iris.  Each loop is closed implicitly: the
    renderer connects the last vertex back to the first.
    """
    regions: list[SubmeshRegion]
    centroid: tuple[float, float, float]
    rim_loops: list[np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rim_loops is None:
            self.rim_loops = []


def _region_centroid(regions: list[SubmeshRegion]) -> tuple[float, float, float]:
    verts = [r.vertices for r in regions if r.vertex_count > 0]
    if not verts:
        return (0.0, 0.0, 0.0)
    stacked = np.vstack(verts)
    c = stacked.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def build_element_subsolids(element, system) -> list[SubSolid]:
    """Build per-region sub-solids for an element.

    Mirrors :func:`build_element_submeshes` but splits each sub-solid into
    surface-attributed regions so the renderer and picking pass can act on
    a single optical surface (front cap vs back cap), not just the whole
    cemented body.
    """
    indices = element.resolve_surfaces(system)
    surfaces = [system.surfaces[i] for i in indices]

    kind = getattr(element.kind, "name", "GLASS")
    if kind == "STOP":
        if not surfaces:
            return []
        iris = build_iris(surfaces[0])
        region = SubmeshRegion(
            vertices=iris.vertices,
            normals=iris.normals,
            indices=iris.indices,
            kinds=iris.kinds,
            surface_index=int(indices[0]),
            is_cap=True,
        )
        # Iris stroke loops: inner aperture rim (the part the picker hits)
        # and outer ring perimeter (where the iris plate ends).
        surface = surfaces[0]
        semi = float(getattr(surface, "semi_aperture", 1.0))
        ring_outer_scale = 1.5
        n_theta = 96
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        r_inner = _aperture_radial_test(theta, surface)
        r_outer = semi * float(ring_outer_scale)
        zero = np.zeros_like(theta)
        # Placed through the same helper as the iris mesh itself, or the
        # stroke loops would peel away from the ring on a posed stop.
        inner = place_surface_vertices(
            np.stack([r_inner * np.cos(theta), r_inner * np.sin(theta), zero],
                     axis=1), surface).astype(np.float32)
        outer = place_surface_vertices(
            np.stack([r_outer * np.cos(theta), r_outer * np.sin(theta), zero],
                     axis=1), surface).astype(np.float32)
        return [SubSolid(
            regions=[region],
            centroid=_region_centroid([region]),
            rim_loops=[inner, outer],
        )]

    sub: list[SubSolid] = []
    for i in range(len(surfaces) - 1):
        front = tessellate_surface(surfaces[i], outward_sign=-1.0)
        back = tessellate_surface(surfaces[i + 1], outward_sign=+1.0)
        wall_a, wall_b = build_side_wall_halves(surfaces[i], surfaces[i + 1])
        si_front = int(indices[i])
        si_back = int(indices[i + 1])
        regions = [
            SubmeshRegion(
                vertices=front.vertices, normals=front.normals,
                indices=front.indices, kinds=front.kinds,
                surface_index=si_front, is_cap=True,
            ),
            SubmeshRegion(
                vertices=back.vertices, normals=back.normals,
                indices=back.indices, kinds=back.kinds,
                surface_index=si_back, is_cap=True,
            ),
            SubmeshRegion(
                vertices=wall_a.vertices, normals=wall_a.normals,
                indices=wall_a.indices, kinds=wall_a.kinds,
                surface_index=si_front, is_cap=False,
            ),
            SubmeshRegion(
                vertices=wall_b.vertices, normals=wall_b.normals,
                indices=wall_b.indices, kinds=wall_b.kinds,
                surface_index=si_back, is_cap=False,
            ),
        ]
        # Rim outlines at the cap/wall seam — the same rim vertices used to
        # stitch the wall halves, so they line up exactly with the lens
        # tessellation (no z-fighting gap when stroke is drawn over fill).
        rim_loops = [
            _surface_rim_vertices(surfaces[i]),
            _surface_rim_vertices(surfaces[i + 1]),
        ]
        sub.append(SubSolid(
            regions=regions,
            centroid=_region_centroid(regions),
            rim_loops=rim_loops,
        ))

    if not sub and surfaces:
        front = tessellate_surface(surfaces[0], outward_sign=-1.0)
        region = SubmeshRegion(
            vertices=front.vertices, normals=front.normals,
            indices=front.indices, kinds=front.kinds,
            surface_index=int(indices[0]), is_cap=True,
        )
        # Single-surface dummy — outline just its rim so the user still gets
        # a stroke at the surface boundary.
        rim_loops = [_surface_rim_vertices(surfaces[0])]
        sub.append(SubSolid(
            regions=[region],
            centroid=_region_centroid([region]),
            rim_loops=rim_loops,
        ))

    return sub


# ---------------------------------------------------------------------------
# Mesh / clip-plane intersection (cross-section silhouette outlines)
# ---------------------------------------------------------------------------

def slice_triangles_by_plane(
    vertices: np.ndarray,
    indices: np.ndarray,
    plane: tuple[float, float, float, float],
) -> np.ndarray:
    """Intersect a triangle mesh with the implicit plane ``ax + by + cz + d = 0``.

    Returns an ``(M, 3)`` float32 array of line-segment endpoints — two
    consecutive rows per segment, ready to upload as ``GL_LINES``.  Empty
    when the plane misses the mesh.  Triangles fully on either side
    contribute nothing; triangles straddling the plane contribute exactly
    one segment between the two edges that cross zero.
    """
    if vertices.shape[0] == 0 or indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float32)
    if float(n @ n) <= 0.5:
        return np.zeros((0, 3), dtype=np.float32)

    tris = indices.reshape(-1, 3)
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    d0 = v0 @ n + d
    d1 = v1 @ n + d
    d2 = v2 @ n + d

    # Robust-against-floating-point straddle test.  ``np.sin(np.pi)`` returns
    # 1.2e-16, not exactly 0 — so the cap mesh's vertex at θ=π has a tiny
    # positive y, while the vertex at θ=0 (where ``np.sin(0) == 0``) is at
    # y=0 exactly.  When the user drops a clip plane through y=0 the strict
    # ``> 0`` test passes for the drifted -X vertices but fails for the
    # clean-zero +X vertices, so the +X half of the chord generates zero
    # segments and the cross-section outline visibly cuts off at x=0.
    #
    # Fix: snap near-zero distances to exact zero, then bias them by a tiny
    # positive constant so "on-plane" vertices count as being on the same
    # (positive) side regardless of which direction float drift sent them.
    # Vertices genuinely off the plane keep their original sign.  The
    # tolerance is mm-scale; clip planes in ghostlight live in millimetres
    # and the smallest meaningful separation is many orders of magnitude
    # above this.
    _PLANE_EPS = 1e-6
    d0 = np.where(np.abs(d0) < _PLANE_EPS, _PLANE_EPS, d0)
    d1 = np.where(np.abs(d1) < _PLANE_EPS, _PLANE_EPS, d1)
    d2 = np.where(np.abs(d2) < _PLANE_EPS, _PLANE_EPS, d2)

    # Pick the straddling triangles: signs of distances aren't all the same.
    pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
    neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
    straddle = pos & neg
    if not np.any(straddle):
        return np.zeros((0, 3), dtype=np.float32)

    v0 = v0[straddle]; v1 = v1[straddle]; v2 = v2[straddle]
    d0 = d0[straddle]; d1 = d1[straddle]; d2 = d2[straddle]

    # For each edge (a, b), if d_a * d_b < 0 the edge crosses zero at
    # t = d_a / (d_a - d_b).  A straddling triangle has exactly two such
    # edges — collect both intersection points per triangle.
    def _edge_cross(va, vb, da, db):
        denom = da - db
        # Avoid div-by-zero where the edge lies exactly in the plane.
        safe = np.where(np.abs(denom) > 1e-20, denom, 1.0)
        t = (da / safe).reshape(-1, 1)
        return va + t * (vb - va), (da * db) < 0.0

    p01, m01 = _edge_cross(v0, v1, d0, d1)
    p12, m12 = _edge_cross(v1, v2, d1, d2)
    p20, m20 = _edge_cross(v2, v0, d2, d0)

    # Pack per-triangle: pick the two crossing edges into a (M, 2, 3) buffer.
    # Each triangle contributes one segment; ordering of the two endpoints
    # is irrelevant for GL_LINES.
    masks = np.stack([m01, m12, m20], axis=1)         # (M, 3) bool
    pts = np.stack([p01, p12, p20], axis=1)           # (M, 3, 3)

    # Order the masks so the True entries come first; segments[:, 0] and
    # segments[:, 1] then index those two endpoints reliably.
    order = np.argsort(~masks, axis=1, kind="stable")  # True first
    pts_sorted = np.take_along_axis(pts, order[..., None], axis=1)
    segments = pts_sorted[:, :2, :].astype(np.float32)  # (M, 2, 3)

    return segments.reshape(-1, 3)


def build_element_submeshes(element, system) -> list[Mesh]:
    """Build one :class:`Mesh` per visible sub-solid of an element.

    A singlet returns one mesh; a cemented doublet returns two; a triplet
    three; etc.  Stop elements return a single iris mesh.  The split lets the
    renderer depth-sort sub-solids independently — necessary because the
    lens pass draws transparent geometry with the depth test off, so within
    a draw call the triangle order is the blend order.  Without the split,
    cemented n-lets blend correctly from one optical-axis direction and
    incorrectly from the other.

    ``element.position`` / ``element.rotation_euler_deg`` are NOT applied — they
    are metadata.  ghostlight's loader bakes element-level transforms into each
    surface's ``decenter_x/y``, ``rot``, and cumulative ``z``, so applying them
    again here would double-offset every element.  Callers who want to instance
    sub-meshes at multiple poses can use :func:`apply_element_transform` on each.
    """
    indices = element.resolve_surfaces(system)
    surfaces = [system.surfaces[i] for i in indices]

    kind = getattr(element.kind, "name", "GLASS")
    if kind == "STOP":
        if not surfaces:
            return []
        return [build_iris(surfaces[0])]

    sub: list[Mesh] = []
    for i in range(len(surfaces) - 1):
        sub.append(loft_glass_solid([surfaces[i], surfaces[i + 1]]))
    if not sub and surfaces:
        # Single-surface dummy element — emit just the cap.
        sub.append(tessellate_surface(surfaces[0], outward_sign=-1.0))
    return sub


def build_element_mesh(element, system) -> Mesh:
    """Backwards-compatible wrapper around :func:`build_element_submeshes`.

    Returns the merged mesh — convenient for bbox computation, tests, and
    any consumer that doesn't care about sub-solid boundaries.  The render
    pipeline uses the sub-mesh form directly so it can depth-sort solids
    within a cemented group.
    """
    sub = build_element_submeshes(element, system)
    if not sub:
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )
    if len(sub) == 1:
        return sub[0]
    return merge_meshes(sub)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def mesh_bbox(meshes: list[Mesh]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(min, max)`` float32 (3,) bounding box across all meshes."""
    if not meshes or all(m.vertex_count == 0 for m in meshes):
        return (
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
        )
    mins = np.vstack([m.vertices.min(axis=0) for m in meshes if m.vertex_count > 0])
    maxs = np.vstack([m.vertices.max(axis=0) for m in meshes if m.vertex_count > 0])
    return mins.min(axis=0).astype(np.float32), maxs.max(axis=0).astype(np.float32)
