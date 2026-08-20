"""Reference geometry for the bladed aperture profile.

The C++ ``ApertureProfile`` must match this implementation's ``radius_at()``
to float32 tolerance. Each half-edge is a conic arc focused at the aperture
centre and linear in ``u = 1/r``:

    u(phi) = A*cos(phi - phi_w) + C

Pinning u(phi_w) = 1/r_w and u(tip) = 1 gives A and C in closed form. This
    * reduces EXACTLY to a/cos(phi) -- the regular polygon -- at curvature 0
    * reduces EXACTLY to u = 1 -- the circle -- at curvature +1
    * is monotone from waist to tip, so it is star-convex by construction, with
      no feasibility limit and no root selection
    * is C1 at the waist (both halves have zero slope there)
    * costs one cos and one divide, no sqrt

Tip facets are straight chords, which are also linear in u, so the whole
profile is a max over same-shaped terms: u = max(edge, facet), r = 1/u.

SECTOR ORIENTATION: this file measures its sector-local angle from a blade midpoint at
theta = 0, so an unrotated profile here is flat-up. Ghostlight's polygon
convention is vertex-up (the file format documents it, and both check_aperture()
and aperture_edge_distance() implement it), so the C++ port drops the half-sector
offset -- ApertureProfile::sector_angle() omits the "+ half" this file's
radius_at() applies. To compare the two, construct this profile with
``rotation_deg + 180 / blades``.

Run: ``python aperture_profile_oracle.py``
"""
from __future__ import annotations

import math
from typing import List, Tuple

# --- constants ---------------------------------------------------------------
# Negative curvature scoops the waist by a fraction OF THE APOTHEM, not by the
# absolute tip-to-apothem gap. The gap shrinks as blades rise (0.5 at 3 blades,
# 0.04 at 11), so scooping by the gap makes -100% curvature pinch the pupil shut
# at 3 blades while doing almost nothing at 11. Scooping by the apothem gives the
# same visual depth at every blade count and can never reach zero.
NEG_DEPTH    = 0.55    # waist drop at curvature -100%, as a fraction of the apothem
MIN_SPAN     = math.radians(4.0)  # narrowest half-edge; bounds the ramp slope at |twist| = 1
# Notch depth is authored in degrees but APPLIED as a fraction of the half-sector,
# for the same reason negative curvature is measured against the apothem: an
# absolute angle means wildly different things across blade counts. A 30deg bite
# is a third of a 3-blade sector and the WHOLE of a 12-blade one. Scaling makes
# +/-45deg mean "the deepest sensible tooth at this blade count" everywhere, so
# the slider responds smoothly instead of saturating above 6 blades.
# The stored degrees are therefore NOMINAL, not literal geometry.
NOTCH_FULL   = math.radians(45.0)   # slider extent that maps to NOTCH_MAX
NOTCH_MAX    = 0.45    # deepest bite, as a fraction of the half-sector
NOTCH_SECTOR = 0.90    # hard ceiling on either side of a skewed bite
SCAN_N       = 512     # setup scan for the area integral


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ApertureProfile:
    """Normalised aperture boundary: r(theta) in (0, 1], 1.0 = blade-tip radius."""

    def __init__(self, blades: int, rotation_deg: float = 0.0,
                 curvature: float = 0.0, twist: float = 0.0,
                 notch_deg: float = 0.0, notch_angle_deg: float = 0.0):
        # ---- authored, hard-clamped ----------------------------------------
        self.blades = max(3, int(blades))
        self.rotation = math.radians(rotation_deg)
        curvature = _clamp(curvature, -1.0, 1.0)
        twist     = _clamp(twist, -1.0, 1.0)
        notch     = _clamp(math.radians(notch_deg), -math.pi / 4, math.pi / 4)
        rake      = _clamp(math.radians(notch_angle_deg), 0.0, math.pi / 4)

        # ---- derived block --------------------------------------------------
        self.sigma = 2.0 * math.pi / self.blades
        self.half  = 0.5 * self.sigma
        a = math.cos(self.half)                    # plain polygon apothem
        self.a = a

        # Curvature: positive reaches the circumscribed circle, negative scoops a
        # fixed fraction of the apothem (see NEG_DEPTH).
        if curvature >= 0.0:
            self.r_w = a + curvature * (1.0 - a)
        else:
            self.r_w = a * (1.0 + curvature * NEG_DEPTH)

        # Twist: the waist stops MIN_SPAN short of a tip, so neither half-edge
        # ever collapses to a near-vertical wall that the pupil raster can't
        # resolve (which would alias and flicker on scrub).
        lim = max(0.0, self.half - MIN_SPAN)
        self.phi_w = _clamp(twist * self.half, -lim, lim)

        uw = 1.0 / self.r_w                        # u at the waist
        dR = self.half - self.phi_w                # angular span, waist -> +tip
        dL = self.half + self.phi_w                # angular span, waist -> -tip
        # NOTE for the C++ float32 port: evaluate the 1 - cos(d) denominators as
        # 2*sin(d/2)^2 -- at the narrowest span the cancellation costs 1.1e-5
        # relative error in float32 vs 1.1e-7 for the stable form.
        self.aR = (uw - 1.0) / (1.0 - math.cos(dR))
        self.aL = (uw - 1.0) / (1.0 - math.cos(dL))
        self.cR = uw - self.aR
        self.cL = uw - self.aL

        # ---- tip facets ------------------------------------------------------
        # The facet is the CHORD between two points that stay ON the blade edge,
        # one either side of the tip. Skew moves those two points in opposite
        # directions rather than pivoting a free line: an unanchored line raked
        # off the tip direction sweeps in toward the centre and ends up slicing
        # the whole aperture instead of trimming a corner.
        e = (abs(notch) / NOTCH_FULL) * NOTCH_MAX * self.half
        s = _clamp(rake / (math.pi / 4.0), 0.0, 1.0)     # skew, 0 = symmetric bite
        eA = min(e * (1.0 + s), NOTCH_SECTOR * self.half)
        eB = e * (1.0 - s)
        if notch < 0.0:
            eA, eB = eB, eA                              # sign picks the leaning side
        self.has_facets = e > 1e-9 and self.r_w < 1.0 and (eA + eB) > 1e-9
        if self.has_facets:
            pa = self.half - eA                          # on this sector's edge
            ra = 1.0 / self._u_edge(pa)
            pb = self.half + eB                          # on the neighbour's edge,
            rb = 1.0 / self._u_edge(-self.half + eB)     # in its own sector coords
            ax, ay = ra * math.cos(pa), ra * math.sin(pa)
            bx, by = rb * math.cos(pb), rb * math.sin(pb)
            dx, dy = bx - ax, by - ay
            dlen = math.hypot(dx, dy)
            if dlen < 1e-9:
                self.has_facets = False
                self.beta = self.p = 0.0
            else:
                nx, ny = dy / dlen, -dx / dlen           # chord normal
                if ax * nx + ay * ny < 0.0:              # orient outward
                    nx, ny = -nx, -ny
                self.p = ax * nx + ay * ny
                self.beta = math.atan2(ny, nx)
                self.eA, self.eB = eA, eB          # the facet's own angular span
                if self.p <= 1e-6:
                    self.has_facets = False
                    self.beta = self.p = 0.0
        if not self.has_facets:
            self.beta = 0.0
            self.p = 0.0
            self.eA = self.eB = 0.0

        # ---- normalisation + area --------------------------------------------
        # NO renormalisation. A facet REMOVES material from a blade tip, so the
        # aperture legitimately gets smaller; rescaling it back out to radius 1
        # would inflate the whole silhouette and flatten the scallops the
        # curvature control just made. The invariant is therefore max r <= 1,
        # with equality when notch == 0 -- which is all the bounding-ellipse fast
        # reject needs (it only wants a conservative upper bound), and the true
        # bounding radius is measured by the calibration azimuth sweep anyway.
        self.area_fraction = self._area_fraction()

    # -- edge -----------------------------------------------------------------
    def _u_edge(self, phi: float) -> float:
        psi = phi - self.phi_w
        if psi >= 0.0:
            return self.aR * math.cos(psi) + self.cR
        return self.aL * math.cos(psi) + self.cL

    def _fold(self, ang: float) -> float:
        """Fold an angle into [-sigma/2, +sigma/2)."""
        a = math.fmod(ang + self.half, self.sigma)
        if a < 0.0:
            a += self.sigma
        return a - self.half

    def _u_raw(self, phi: float) -> float:
        u = self._u_edge(phi)
        if self.has_facets:
            # One facet per tip, so the family is sigma-periodic; folding against
            # the TIP direction picks the relevant one. The facet is a SEGMENT,
            # not a half-plane -- it is only applied across its own angular span.
            # A skewed chord's infinite line would otherwise cut regions the real
            # facet never reaches, collapsing the silhouette. Both endpoints lie
            # ON the edge, so the constraint agrees with the edge exactly at the
            # span boundaries and the profile stays continuous.
            w = self._fold(phi - self.half)
            if -self.eA <= w <= self.eB:
                uf = math.cos(w + self.half - self.beta) / self.p
                if uf > u:
                    u = uf
        return u

    # -- public ---------------------------------------------------------------
    def radius_at(self, theta: float) -> float:
        """Boundary radius at absolute angle theta (rotation applied)."""
        phi = math.fmod(theta - self.rotation + self.half, self.sigma)
        if phi < 0.0:
            phi += self.sigma
        phi -= self.half
        return 1.0 / self._u_raw(phi)

    def _area_fraction(self) -> float:
        """Enclosed area as a fraction of the unit disk. Simpson over one sector."""
        n = SCAN_N if SCAN_N % 2 == 0 else SCAN_N + 1
        h = self.sigma / n
        total = 0.0
        for i in range(n + 1):
            phi = -self.half + i * h
            r = 1.0 / self._u_raw(phi)
            w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            total += w * r * r
        return self.blades * (total * h / 3.0) / (2.0 * math.pi)

    def dr_dtheta(self, theta: float) -> float:
        """Analytic slope, for the HURB first-order edge distance."""
        phi = math.fmod(theta - self.rotation + self.half, self.sigma)
        if phi < 0.0:
            phi += self.sigma
        phi -= self.half
        u = self._u_raw(phi)
        # du/dphi of whichever term currently wins
        ue = self._u_edge(phi)
        best, du = ue, (-(self.aR if phi >= self.phi_w else self.aL)
                        * math.sin(phi - self.phi_w))
        if self.has_facets:
            w = self._fold(phi - self.half)
            if -self.eA <= w <= self.eB:
                uf = math.cos(w + self.half - self.beta) / self.p
                if uf > best:
                    best, du = uf, -math.sin(w + self.half - self.beta) / self.p
        return -du / (u * u)

    def points(self, n: int = 2048) -> List[Tuple[float, float]]:
        out = []
        for i in range(n):
            t = 2.0 * math.pi * i / n
            r = self.radius_at(t)
            out.append((r * math.cos(t), r * math.sin(t)))
        return out


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------
def _all_test_profiles(flag=False):
    out = []
    for blades in (3, 5, 6, 8):
        for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for t in (-1.0, 0.0, 0.6):
                for nd, na in ((0.0, 0.0), (25.0, 0.0), (-35.8, 45.0), (45.0, 20.0)):
                    p = ApertureProfile(blades, 0.0, c, t, nd, na)
                    lab = f"b{blades} c{c} t{t} n{nd}/{na}"
                    out.append((p, lab, p.has_facets) if flag else (p, lab))
    return out


def self_checks() -> List[Tuple[str, bool, str]]:
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    worst = 0.0
    for blades in (3, 5, 6, 8, 11):
        p = ApertureProfile(blades)
        a = math.cos(math.pi / blades)
        for i in range(1000):
            th = 2.0 * math.pi * i / 1000
            off = math.fmod(th + math.pi / blades, 2 * math.pi / blades)
            if off < 0:
                off += 2 * math.pi / blades
            off -= math.pi / blades
            worst = max(worst, abs(p.radius_at(th) - a / math.cos(off)))
    chk("defaults == regular polygon", worst < 1e-12, f"max err {worst:.2e}")

    w = max(abs(ApertureProfile(5, curvature=1.0).radius_at(2 * math.pi * i / 500) - 1.0)
            for i in range(500))
    chk("curvature +1 == circle", w < 1e-12, f"max err {w:.2e}")

    # curvature +1 must win regardless of twist / notch
    w2 = max(abs(ApertureProfile(7, curvature=1.0, twist=-0.8, notch_deg=40,
                                 notch_angle_deg=45).radius_at(2 * math.pi * i / 500) - 1.0)
             for i in range(500))
    chk("curvature +1 circle under twist/notch", w2 < 1e-12, f"max err {w2:.2e}")

    bad = []
    for prof, label, notched in _all_test_profiles(flag=True):
        m = max(prof.radius_at(2 * math.pi * i / 4000) for i in range(4000))
        if m > 1.0 + 1e-9 or (not notched and abs(m - 1.0) > 1e-9):
            bad.append(f"{label}={m:.4f}")
    chk("max radius <= 1, == 1 unnotched", not bad, ", ".join(bad[:3]))

    p0 = ApertureProfile(5, curvature=-0.6)
    sym = max(abs(p0.radius_at(t) - p0.radius_at(-t))
              for t in [2 * math.pi * i / 500 for i in range(500)])
    p1 = ApertureProfile(5, curvature=-0.6, twist=0.7)
    asym = max(abs(p1.radius_at(t) - p1.radius_at(-t))
               for t in [2 * math.pi * i / 500 for i in range(500)])
    chk("twist 0 symmetric / twist!=0 chiral", sym < 1e-12 and asym > 1e-2,
        f"sym {sym:.1e}, chiral {asym:.3f}")

    pa = ApertureProfile(5, curvature=-0.6, twist=0.7)
    pb = ApertureProfile(5, curvature=-0.6, twist=-0.7)
    mir = max(abs(pa.radius_at(t) - pb.radius_at(-t))
              for t in [2 * math.pi * (i + 0.37) / 500 for i in range(500)])
    chk("twist +/-t are mirror images", mir < 1e-12, f"max err {mir:.1e}")

    qa = ApertureProfile(6, curvature=-0.4, notch_deg=30, notch_angle_deg=35)
    qb = ApertureProfile(6, curvature=-0.4, notch_deg=-30, notch_angle_deg=35)
    nmir = max(abs(qa.radius_at(t) - qb.radius_at(-t))
               for t in [2 * math.pi * (i + 0.37) / 500 for i in range(500)])
    chk("notch +/- are mirror images", nmir < 1e-12, f"max err {nmir:.1e}")

    q0 = ApertureProfile(6, curvature=0.4, twist=0.3, notch_deg=0.0, notch_angle_deg=45.0)
    q1 = ApertureProfile(6, curvature=0.4, twist=0.3)
    noop = max(abs(q0.radius_at(t) - q1.radius_at(t))
               for t in [2 * math.pi * i / 500 for i in range(500)])
    chk("notch 0 == no-op", noop == 0.0, f"max err {noop:.1e}")

    hexa = ApertureProfile(6).area_fraction
    want = (6 / (2 * math.pi)) * math.sin(math.pi / 3)
    chk("hexagon area analytic", abs(hexa - want) < 1e-6, f"{hexa:.6f} vs {want:.6f}")

    areas = [ApertureProfile(5, curvature=c / 10).area_fraction for c in range(-10, 11)]
    chk("area monotonic in curvature", all(b >= a - 1e-9 for a, b in zip(areas, areas[1:])),
        f"{areas[0]:.3f} -> {areas[-1]:.3f}")

    bad = []
    for prof, label in _all_test_profiles():
        rs = [prof.radius_at(2 * math.pi * i / 2000) for i in range(2000)]
        if min(rs) <= 0.0 or not all(math.isfinite(r) for r in rs):
            bad.append(label)
    chk("r > 0 and finite", not bad, ", ".join(bad[:3]))

    # Continuity, not steepness: a genuine discontinuity keeps the same jump as
    # the sampling refines, whereas a merely steep ramp halves it. Measure the
    # ratio between 4k and 8k samples -- continuous => ~0.5, jump => ~1.0.
    bad, worst_slope = [], 0.0
    for prof, label in _all_test_profiles():
        def max_jump(n):
            rs = [prof.radius_at(2 * math.pi * i / n) for i in range(n)]
            return max(abs(b - a) for a, b in zip(rs, rs[1:] + rs[:1]))
        j1, j2 = max_jump(4000), max_jump(8000)
        worst_slope = max(worst_slope, j2 * 8000 / (2 * math.pi))
        if j1 > 1e-6 and j2 / j1 > 0.75:
            bad.append(f"{label} ratio={j2 / j1:.2f}")
    chk("continuous (jump halves as sampling doubles)", not bad,
        ", ".join(bad[:3]) or f"steepest |dr/dtheta| ~ {worst_slope:.0f}/rad")

    # analytic slope vs finite difference
    worst = 0.0
    for prof, _ in _all_test_profiles()[:60]:
        for i in range(200):
            t = 2 * math.pi * (i + 0.31) / 200
            h = 1e-6
            fd = (prof.radius_at(t + h) - prof.radius_at(t - h)) / (2 * h)
            worst = max(worst, abs(fd - prof.dr_dtheta(t)))
    chk("analytic dr/dtheta vs finite diff", worst < 1e-3, f"max err {worst:.2e}")

    # area never collapses -- the degenerate-pupil guard
    amin = min(p.area_fraction for p, _ in _all_test_profiles())
    chk("open area never collapses", amin > 0.05, f"min area fraction {amin:.4f}")

    return res


if __name__ == "__main__":
    ok_all = True
    for name, ok, detail in self_checks():
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    print()
    ref = ApertureProfile(5, rotation_deg=29.0, curvature=-0.90, twist=-0.80,
                          notch_deg=35.8, notch_angle_deg=45.0)
    rs = [ref.radius_at(2 * math.pi * i / 4000) for i in range(4000)]
    print(f"reference combo: area {ref.area_fraction:.4f}  r_w {ref.r_w:.4f}  "
          f"r range {min(rs):.4f}-{max(rs):.4f}  p {ref.p:.4f}")
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
