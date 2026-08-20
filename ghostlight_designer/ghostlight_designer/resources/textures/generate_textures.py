"""Generate the bundled example texture PNGs.

Run from anywhere; the images are written next to this file::

    python -m ghostlight_designer.resources.textures.generate_textures
    python ghostlight_designer/ghostlight_designer/resources/textures/generate_textures.py

Everything is seeded, so re-running reproduces the committed PNGs byte-for-byte.
Only numpy + Pillow are required (scipy is used for the blur when present).

Two families come out of here, and they are the same file format serving the two
code paths described in ``README.md``:

* ``aperture_*.png`` -- hard mattes for ``ApertureShape.IMAGE``.  The tracer
  thresholds at 0.5, so these are authored black/white with an anti-aliased
  edge (the grey boundary pixels feed the SDF bake and the MDFT pupil).  Every
  mask keeps a black margin: the GPU sampler clamps at the texture border, so a
  mask that runs to the edge would leak transmission outside the aperture.
* ``dirt_*.png`` -- graded front-glass transmission maps.  These multiply the
  pupil amplitude, so they sit near white (1.0 = clean glass) and only dip where
  there is dust, grease or a scratch.  No black margin: a dark border would
  clamp-darken the pupil rim.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# Output settings
# --------------------------------------------------------------------------- #

MASK_PX = 1024          # aperture matte resolution
MASK_SS = 4             # supersample factor used for anti-aliasing
DIRT_PX = 2048          # dirt / transmission map resolution

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur a float32 image.  scipy when available, Pillow otherwise."""
    if sigma <= 0.0:
        return a
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        from PIL import ImageFilter
        img = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), mode="F")
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(img, dtype=np.float32)
    return gaussian_filter(a.astype(np.float32), sigma, mode="nearest")


def _coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates on [-1, 1] as (x, y) with y increasing downward."""
    t = (np.arange(n, dtype=np.float32) + 0.5) / n * 2.0 - 1.0
    return np.meshgrid(t, t)


def _downsample(a: np.ndarray, factor: int) -> np.ndarray:
    """Box-average `a` by an integer factor (the anti-aliasing step)."""
    if factor == 1:
        return a
    h, w = a.shape
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def _value_noise(rng: np.random.Generator, n: int, cells: int) -> np.ndarray:
    """Smooth [0, 1) value noise: a small random grid resampled up bicubically."""
    small = rng.random((cells, cells)).astype(np.float32)
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    img = Image.fromarray(small, mode="F").resize((n, n), bicubic)
    return np.asarray(img, dtype=np.float32)


def _fbm(rng: np.random.Generator, n: int, cells: int, octaves: int) -> np.ndarray:
    """Fractional Brownian motion in [0, 1], normalised."""
    out = np.zeros((n, n), dtype=np.float32)
    amp, total = 1.0, 0.0
    for i in range(octaves):
        out += amp * _value_noise(rng, n, cells << i)
        total += amp
        amp *= 0.5
    out /= total
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / max(hi - lo, 1e-6)


def _save(a: np.ndarray, name: str) -> None:
    """Write a float [0, 1] image as an 8-bit greyscale PNG."""
    b = np.clip(a, 0.0, 1.0)
    b = np.round(b * 255.0).astype(np.uint8)
    path = os.path.join(OUT_DIR, name)
    Image.fromarray(b, mode="L").save(path, optimize=True)
    print(f"  {name:34s} {b.shape[1]}x{b.shape[0]}  "
          f"{os.path.getsize(path) / 1024.0:7.1f} KiB")


def _draw(n: int, fn) -> np.ndarray:
    """Rasterise vector strokes with ImageDraw and return them as float32 [0, 1]."""
    img = Image.new("L", (n, n), 0)
    fn(ImageDraw.Draw(img))
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Aperture mattes
# --------------------------------------------------------------------------- #

def _iris(n: int, blades: int, fill: float, curvature: float,
          rotation: float = 0.0) -> np.ndarray:
    """Signed-ish blade iris: the intersection of `blades` offset discs.

    `curvature` is the blade-arc radius in aperture radii.  Large values give a
    straight-edged polygon; values near 1 give an almost circular opening.
    """
    x, y = _coords(n)
    inside = np.ones((n, n), dtype=bool)
    for k in range(blades):
        th = rotation + 2.0 * np.pi * k / blades
        # Disc of radius `curvature` whose edge is tangent to the blade line at
        # distance `fill` from centre.
        cx = (curvature - fill) * np.cos(th)
        cy = (curvature - fill) * np.sin(th)
        inside &= ((x - cx) ** 2 + (y - cy) ** 2) <= curvature ** 2
    return inside.astype(np.float32)


def mask_iris_6blade(rng: np.random.Generator) -> np.ndarray:
    n = MASK_PX * MASK_SS
    # Rotated a half-blade so the opening is mirror-symmetric top/bottom.
    m = _iris(n, blades=6, fill=0.82, curvature=1.9, rotation=np.pi / 6.0)
    return _downsample(m, MASK_SS)


def mask_iris_9blade_stopped(rng: np.random.Generator) -> np.ndarray:
    """Stopped well down with near-straight blades -> a hard 18-point star."""
    n = MASK_PX * MASK_SS
    m = _iris(n, blades=9, fill=0.52, curvature=6.0, rotation=np.pi / 2.0)
    return _downsample(m, MASK_SS)


def mask_iris_14blade_chipped(rng: np.random.Generator) -> np.ndarray:
    """Nearly round 14-blade iris with a chipped blade and a nick in the rim."""
    n = MASK_PX * MASK_SS
    m = _iris(n, blades=14, fill=0.84, curvature=1.15, rotation=np.pi / 14.0)
    x, y = _coords(n)

    # One blade sits proud of the others -- a flat chord clipping the rim.
    chord = (y * np.cos(0.18) + x * np.sin(0.18)) > 0.775
    m[chord] = 0.0

    # A small nick out of the lower-left rim.
    nick = ((x + 0.52) ** 2 + (y - 0.58) ** 2) < 0.050 ** 2
    m[nick] = 0.0
    return _downsample(m, MASK_SS)


def mask_spider_vanes(rng: np.random.Generator) -> np.ndarray:
    """Four spider vanes across a round aperture -- mirror-lens style spikes.

    Deliberately *not* an annulus.  A central obstruction leaves no clear axial
    ray, and the entrance-pupil solve then returns zero, so the lens fails
    calibration with "degenerate first-order optics" before anything renders.
    The vanes carry the catadioptric spike signature on their own; they stop
    short of the axis (where the secondary they would carry has been left out)
    so a clear axial ray survives.
    """
    n = MASK_PX * MASK_SS
    x, y = _coords(n)
    r = np.hypot(x, y)
    m = (r <= 0.90).astype(np.float32)

    half, r_hub = 0.028, 0.17
    for th in (np.pi / 4.0, 3.0 * np.pi / 4.0):
        d = np.abs(x * np.sin(th) - y * np.cos(th))
        m[(d <= half) & (r >= r_hub)] = 0.0
    return _downsample(m, MASK_SS)


def mask_star_5(rng: np.random.Generator) -> np.ndarray:
    """Five-point star cutout -- a classic novelty bokeh gobo.

    A straight-edged star polygon, not a polar rose: real gobos are cut, so the
    points need to come to a hard vertex rather than a rounded lobe.
    """
    n = MASK_PX * MASK_SS
    c = n * 0.5
    r_out, r_in = c * 0.94, c * 0.40
    pts = []
    for k in range(10):
        # k=0 points straight up; screen y grows downward, hence the minus.
        th = -np.pi / 2.0 + k * np.pi / 5.0
        rad = r_out if k % 2 == 0 else r_in
        pts.append((c + rad * np.cos(th), c + rad * np.sin(th)))
    return _downsample(_draw(n, lambda d: d.polygon(pts, fill=255)), MASK_SS)


def mask_heart(rng: np.random.Generator) -> np.ndarray:
    """Heart cutout.  Deliberately asymmetric -- see the orientation note in
    README.md; this is the asset to load when you want to confirm which way up
    the renderer maps a matte."""
    n = MASK_PX * MASK_SS
    x, y = _coords(n)
    hx = x * 1.25
    hy = -y * 1.25 + 0.25          # flip to screen-up, lift to centre the mass
    implicit = (hx ** 2 + hy ** 2 - 0.36) ** 3 - hx ** 2 * hy ** 3 * 1.05
    return _downsample((implicit <= 0.0).astype(np.float32), MASK_SS)


def mask_anamorphic_oval(rng: np.random.Generator) -> np.ndarray:
    """2:1 oval slot with softly clipped ends -- stretches the flare horizontally."""
    n = MASK_PX * MASK_SS
    x, y = _coords(n)
    m = ((x / 0.90) ** 2 + (y / 0.45) ** 2 <= 1.0)
    m &= np.abs(x) <= 0.84         # flat-cut the extreme ends, as a real oval mask is
    return _downsample(m.astype(np.float32), MASK_SS)


# --------------------------------------------------------------------------- #
# Dirt / transmission maps
# --------------------------------------------------------------------------- #

def _specks(rng: np.random.Generator, n: int, count: int,
            r_lo: float, r_hi: float, depth: float) -> np.ndarray:
    """Additive occlusion field of soft round dust motes (0 = clean)."""
    occ = np.zeros((n, n), dtype=np.float32)
    xs = rng.random(count) * n
    ys = rng.random(count) * n
    # Heavier tail on the small end: most dust is barely visible.
    rs = r_lo + (r_hi - r_lo) * rng.random(count) ** 3.0
    ds = depth * (0.25 + 0.75 * rng.random(count))
    for cx, cy, rad, d in zip(xs, ys, rs, ds):
        pad = int(np.ceil(rad * 2.5)) + 1
        x0, x1 = max(0, int(cx) - pad), min(n, int(cx) + pad)
        y0, y1 = max(0, int(cy) - pad), min(n, int(cy) + pad)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = np.arange(x0, x1, dtype=np.float32) - cx
        gy = np.arange(y0, y1, dtype=np.float32) - cy
        rr = np.hypot(gx[None, :], gy[:, None]) / rad
        # Flat core with a soft shoulder -- reads as a mote, not a gaussian blob.
        blob = d * np.clip(1.6 - 1.6 * rr ** 2.2, 0.0, 1.0)
        occ[y0:y1, x0:x1] = np.maximum(occ[y0:y1, x0:x1], blob)
    return occ


def _fibres(rng: np.random.Generator, n: int, count: int,
            depth: float, width: int = 3) -> np.ndarray:
    """Occlusion field of curved lint fibres."""
    def paint(d):
        for _ in range(count):
            x, y = rng.random() * n, rng.random() * n
            th = rng.random() * 2.0 * np.pi
            pts = [(x, y)]
            for _ in range(28):
                th += rng.normal(0.0, 0.26)
                x += np.cos(th) * n * 0.012
                y += np.sin(th) * n * 0.012
                pts.append((x, y))
            d.line(pts, fill=255, width=width, joint="curve")
    return _draw(n, paint) * depth


def dirt_dust_light(rng: np.random.Generator) -> np.ndarray:
    """A lens someone actually looks after: a scatter of motes and two fibres."""
    n = DIRT_PX
    occ = _specks(rng, n, count=420, r_lo=1.2, r_hi=9.0, depth=0.55)
    occ = np.maximum(occ, _blur(_fibres(rng, n, 2, 0.30, width=2), 1.2))
    occ = _blur(occ, 0.8)
    return 1.0 - np.clip(occ, 0.0, 0.9)


def dirt_dust_heavy(rng: np.random.Generator) -> np.ndarray:
    """Neglected front element: dense dust, lint, and a film of grime."""
    n = DIRT_PX
    occ = _specks(rng, n, count=2600, r_lo=1.2, r_hi=16.0, depth=0.75)
    occ = np.maximum(occ, _blur(_fibres(rng, n, 7, 0.42, width=3), 1.5))
    # Broad greasy film on top, strongest away from centre.
    x, y = _coords(n)
    film = _fbm(rng, n, cells=6, octaves=4)
    occ += 0.16 * film * (0.35 + 0.65 * np.hypot(x, y))
    occ = _blur(occ, 1.0)
    return 1.0 - np.clip(occ, 0.0, 0.92)


def dirt_fingerprint(rng: np.random.Generator) -> np.ndarray:
    """Thumb print off to one side, plus the dust that always comes with it."""
    n = DIRT_PX
    x, y = _coords(n)
    # Whorl centred low-right of the optical axis.
    px, py = x - 0.24, y + 0.16
    rr = np.hypot(px * 1.0, py * 1.25)
    th = np.arctan2(py, px)
    # Ridges spiral outwards; the theta term is what makes it a whorl not a target.
    # A slow noise term warps the spacing so it doesn't read as a printed target.
    warp = _fbm(rng, n, cells=5, octaves=3)
    ridges = 0.5 + 0.5 * np.sin(rr * 52.0 + th * 2.2 + 3.0 * rr ** 2 + 6.0 * warp)
    # Ridge contact is patchy -- modulate by noise and gate on a soft ellipse.
    patch = _fbm(rng, n, cells=8, octaves=3)
    envelope = np.clip(1.0 - (rr / 0.70) ** 2, 0.0, 1.0) ** 0.8
    occ = 0.30 * ridges * envelope * (0.20 + 0.80 * patch)
    # Smeared halo where the finger dragged.
    occ += 0.09 * _blur(envelope, n * 0.02) * patch
    occ = np.maximum(occ, _specks(rng, n, count=300, r_lo=1.2, r_hi=7.0, depth=0.45))
    occ = _blur(occ, 1.1)
    return 1.0 - np.clip(occ, 0.0, 0.85)


def dirt_scratches_polish(rng: np.random.Generator) -> np.ndarray:
    """Swirl marks from careless cleaning, plus three deep scratches."""
    n = DIRT_PX
    c = n * 0.5

    def paint(d):
        # Fine circular polish swirls: short arcs at random radii.
        for _ in range(700):
            rad = (0.08 + 0.92 * rng.random() ** 0.7) * c
            a0 = rng.random() * 360.0
            span = 6.0 + rng.random() * 40.0
            box = (c - rad, c - rad, c + rad, c + rad)
            d.arc(box, a0, a0 + span, fill=150, width=1 + int(rng.random() * 2))
        # A few longer, deeper gouges straight across the glass.
        for _ in range(3):
            a = rng.random() * 2.0 * np.pi
            off = (rng.random() - 0.5) * n * 0.55
            dx, dy = np.cos(a), np.sin(a)
            nx, ny = -dy, dx
            length = n * (0.35 + 0.5 * rng.random())
            x0 = c + nx * off - dx * length * 0.5
            y0 = c + ny * off - dy * length * 0.5
            d.line([(x0, y0), (x0 + dx * length, y0 + dy * length)],
                   fill=210, width=2 + int(rng.random() * 2))

    occ = _blur(_draw(n, paint), 1.0) * 0.75
    occ = np.maximum(occ, _specks(rng, n, count=200, r_lo=1.2, r_hi=6.0, depth=0.35))
    return 1.0 - np.clip(occ, 0.0, 0.85)


def dirt_water_droplets(rng: np.random.Generator) -> np.ndarray:
    """Rain on the front element: dark discs with bright refracting rims."""
    n = DIRT_PX
    field = np.ones((n, n), dtype=np.float32)
    for _ in range(150):
        cx, cy = rng.random() * n, rng.random() * n
        rad = n * (0.004 + 0.030 * rng.random() ** 2.2)
        squash = 1.0 + 0.35 * rng.random()          # droplets sag under gravity
        pad = int(rad * 2.0) + 3
        x0, x1 = max(0, int(cx) - pad), min(n, int(cx) + pad)
        y0, y1 = max(0, int(cy) - pad), min(n, int(cy) + pad)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = (np.arange(x0, x1, dtype=np.float32) - cx) / rad
        gy = (np.arange(y0, y1, dtype=np.float32) - cy) / (rad * squash)
        rr = np.hypot(gx[None, :], gy[:, None])
        # The body scatters light out of the beam; the meniscus at the rim is
        # thin enough to stay clear, so it reads as a bright ring.  Transmission
        # is capped at 1.0 -- a dirt map may attenuate the pupil, never amplify it.
        body = np.clip(1.0 - rr, 0.0, 1.0)
        rim = np.exp(-((rr - 0.92) / 0.13) ** 2)
        tile = np.clip(1.0 - 0.60 * body ** 0.5 + 0.50 * rim, 0.0, 1.0)
        patch = field[y0:y1, x0:x1]
        field[y0:y1, x0:x1] = np.where(rr <= 1.15, np.minimum(patch, tile), patch)
    field = _blur(field, 1.2)
    # Residual mist between the drops.
    field *= 1.0 - 0.10 * _fbm(rng, n, cells=10, octaves=3)
    return np.clip(field, 0.0, 1.0)


def dirt_grime_haze(rng: np.random.Generator) -> np.ndarray:
    """Soft edge-heavy haze -- the map to reach for when you want veil, not sparkle."""
    n = DIRT_PX
    x, y = _coords(n)
    r = np.hypot(x, y)
    base = _fbm(rng, n, cells=4, octaves=5)
    # Grime creeps in from the rim where the barrel traps it, so the centre stays
    # near-clean; the whole map is gentle -- this one is for veil, not sparkle.
    occ = 0.22 * base * np.clip((r - 0.25) / 0.75, 0.0, 1.0) ** 1.8
    occ += 0.03 * base
    occ = np.maximum(occ, _specks(rng, n, count=120, r_lo=1.5, r_hi=10.0, depth=0.30))
    return 1.0 - np.clip(_blur(occ, 2.0), 0.0, 0.8)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

# (filename, builder, seed).  Seeds are fixed so the PNGs are reproducible.
MASKS = [
    ("aperture_iris_6blade.png",        mask_iris_6blade,          1),
    ("aperture_iris_9blade_stopped.png", mask_iris_9blade_stopped, 2),
    ("aperture_iris_14blade_chipped.png", mask_iris_14blade_chipped, 3),
    ("aperture_spider_vanes.png",        mask_spider_vanes,         4),
    ("aperture_star_5.png",             mask_star_5,               5),
    ("aperture_heart.png",              mask_heart,                6),
    ("aperture_anamorphic_oval.png",    mask_anamorphic_oval,      7),
]

DIRT = [
    ("dirt_dust_light.png",       dirt_dust_light,       11),
    ("dirt_dust_heavy.png",       dirt_dust_heavy,       12),
    ("dirt_fingerprint.png",      dirt_fingerprint,      13),
    ("dirt_scratches_polish.png", dirt_scratches_polish, 14),
    ("dirt_water_droplets.png",   dirt_water_droplets,   15),
    ("dirt_grime_haze.png",       dirt_grime_haze,       16),
]


def main() -> None:
    print("aperture mattes:")
    for name, fn, seed in MASKS:
        a = fn(np.random.default_rng(seed))
        # Guarantee the black margin the GPU border-clamp needs.
        margin = max(2, MASK_PX // 128)
        a[:margin, :] = 0.0
        a[-margin:, :] = 0.0
        a[:, :margin] = 0.0
        a[:, -margin:] = 0.0
        _save(a, name)

    print("dirt / transmission maps:")
    for name, fn, seed in DIRT:
        _save(fn(np.random.default_rng(seed)), name)


if __name__ == "__main__":
    main()
