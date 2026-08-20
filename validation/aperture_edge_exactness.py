"""How wrong is the HURB edge distance, and is closing it worth it?

`ApertureProfile::edge_gap()` returns a FIRST-ORDER distance: the query point's
distance to the tangent of the one blade edge that owns its sector. That is
exact on the boundary and one-sided at a blade tip, where two blades meet and
the nearer one is not always the one owning the sector.

This script measures the gap against the exact answer so the decision not to
close it rests on numbers. Run it:

    python aperture_edge_exactness.py

The exact distance comes from the normal-condition quartic. Each half-edge is
the focal conic 1/r = A*cos(psi) + C, which in the psi frame is

    alpha*X^2 + beta*X + gamma*Y^2 - 1 = 0    alpha = C^2-A^2, beta = 2A, gamma = C^2

The foot of the normal from Q satisfies (Q - X) parallel to grad F. That is
linear in Y, so eliminating Y leaves a quartic in X:

    (alpha*X^2 + beta*X - 1) * D(X)^2 + gamma * N(X)^2 = 0
    N = qy*(2*alpha*X + beta)        D = -2*A^2*X + 2*C^2*qx + 2*A

Three things the quartic alone does not give, each of which cost a debugging
round and each of which a C++ port would also need:

  * A -> 0 is a circle and C -> 0 is a straight line (the plain polygon, and
    everything near it). Both degeneracies have to be branched out by hand.
  * A focal conic has a conjugate branch that is not the blade edge. The quartic
    returns feet on it, so every root is checked against the polar equation.
  * Minima land on arc ENDS as often as inside them -- a tip or a waist -- so the
    endpoints are evaluated separately.

Verification: where a 120k-point brute force over the closed silhouette can
resolve the answer (depths of 2% and beyond), the quartic agrees with it to
0.000000%. Below that the brute force runs out of angular resolution before the
quartic does, so the quartic is the reference here and the brute force is not.
"""
import math

from aperture_profile_oracle import ApertureProfile

# The kick scale is sigma = lambda*K/d, so d does not have a cutoff so much as a
# weighting: HURB stays live out to d = lambda*K/hurb_min_sigma_rad, which at the
# defaults (587.56 nm, Lorentzian K = 1/2pi, min sigma 1e-5 rad) is 9.35 mm --
# essentially the whole of a 10 mm semi-aperture. What narrows the interesting
# region is not the cutoff but the 1/d: each decade of kick strength comes from a
# decade of d, so an error at large d perturbs a kick that was already invisible.
#
# Sampling by sigma rather than by depth keeps that honest. LAMBDA_K/sigma is the
# distance in mm at which the kick has scale sigma.
LAMBDA_K_MM = 587.56e-6 * 0.159154943
SEMI_MM = 10.0
SIGMAS = (1.0e-2, 1.0e-3, 1.0e-4)     # rad: strong, moderate, near the floor

PROFILES = [
    ("plain",          dict()),
    ("curvature +0.6", dict(curvature=0.6)),
    ("curvature -0.6", dict(curvature=-0.6)),
    ("twist 0.5",      dict(twist=0.5)),
    ("twist 1.0",      dict(twist=1.0)),
    ("curv+0.6 tw1.0", dict(curvature=0.6, twist=1.0)),
    ("curv-0.6 tw1.0", dict(curvature=-0.6, twist=1.0)),
]


def _quartic_roots(c4, c3, c2, c1, c0):
    """Real roots of a quartic, by deflation through the resolvent cubic."""
    if abs(c4) < 1e-18:
        return _cubic_roots(c3, c2, c1, c0)
    a, b, c, d = c3 / c4, c2 / c4, c1 / c4, c0 / c4
    # Depress: x = y - a/4.
    p = b - 3.0 * a * a / 8.0
    q = c - a * b / 2.0 + a * a * a / 8.0
    r = d - a * c / 4.0 + a * a * b / 16.0 - 3.0 * a ** 4 / 256.0
    shift = -a / 4.0

    if abs(q) < 1e-14:                       # biquadratic
        out = []
        disc = p * p - 4.0 * r
        if disc >= 0.0:
            for z in ((-p + math.sqrt(disc)) / 2.0, (-p - math.sqrt(disc)) / 2.0):
                if z >= 0.0:
                    out += [math.sqrt(z) + shift, -math.sqrt(z) + shift]
        return out

    # Resolvent cubic: z^3 - p*z^2 - 4r*z + (4pr - q^2) = 0. Any real root works.
    z = max(_cubic_roots(1.0, -p, -4.0 * r, 4.0 * p * r - q * q))
    u = z - p
    if u <= 1e-18:
        return []
    su = math.sqrt(u)
    v = z * z / 4.0 - r
    sv = math.sqrt(v) if v > 0.0 else 0.0
    if q > 0.0:
        sv = -sv
    out = []
    for sgn in (1.0, -1.0):
        A2 = sgn * su
        B2 = z / 2.0 + sgn * sv
        disc = A2 * A2 - 4.0 * B2
        if disc >= 0.0:
            sd = math.sqrt(disc)
            out += [(-A2 + sd) / 2.0 + shift, (-A2 - sd) / 2.0 + shift]
    return out


def _cubic_roots(a, b, c, d):
    if abs(a) < 1e-18:
        if abs(b) < 1e-18:
            return [-d / c] if abs(c) > 1e-18 else []
        disc = c * c - 4.0 * b * d
        if disc < 0.0:
            return []
        s = math.sqrt(disc)
        return [(-c + s) / (2.0 * b), (-c - s) / (2.0 * b)]
    b, c, d = b / a, c / a, d / a
    p = c - b * b / 3.0
    q = 2.0 * b ** 3 / 27.0 - b * c / 3.0 + d
    shift = -b / 3.0
    disc = q * q / 4.0 + p ** 3 / 27.0
    if disc > 0.0:
        s = math.sqrt(disc)
        return [math.copysign(abs(-q / 2.0 + s) ** (1 / 3), -q / 2.0 + s)
                + math.copysign(abs(-q / 2.0 - s) ** (1 / 3), -q / 2.0 - s) + shift]
    rad = math.sqrt(-p / 3.0)
    phi = math.acos(max(-1.0, min(1.0, 3.0 * q / (2.0 * p * rad))))
    return [2.0 * rad * math.cos((phi - 2.0 * math.pi * k) / 3.0) + shift
            for k in range(3)]


def _feet_on_conic(A, C, qx, qy):
    """Feet of the normal from Q onto 1/r = A*cos(psi) + C, in the psi frame."""
    if abs(A) < 1e-12:                                   # circle, radius 1/C
        rq = math.hypot(qx, qy)
        if rq < 1e-12:
            return [(1.0 / C, 0.0)]
        s = (1.0 / C) / rq
        return [(qx * s, qy * s)]
    if abs(C) < 1e-12:                                   # straight edge X = 1/A
        return [(1.0 / A, qy)]

    al, be, ga = C * C - A * A, 2.0 * A, C * C
    n0, n1 = be * qy, 2.0 * al * qy                      # N = n0 + n1*X
    d0, d1 = 2.0 * C * C * qx + 2.0 * A, -2.0 * A * A    # D = d0 + d1*X
    # (al*X^2 + be*X - 1) * D^2 + ga * N^2
    c4 = al * d1 * d1
    c3 = al * 2.0 * d0 * d1 + be * d1 * d1
    c2 = al * d0 * d0 + be * 2.0 * d0 * d1 - d1 * d1 + ga * n1 * n1
    c1 = be * d0 * d0 - 2.0 * d0 * d1 + ga * 2.0 * n0 * n1
    c0 = -d0 * d0 + ga * n0 * n0

    out = []
    for X in _quartic_roots(c4, c3, c2, c1, c0):
        den = d0 + d1 * X
        if abs(den) < 1e-14:
            continue
        out.append((X, qy * (2.0 * al * X + be) / den))
    return out


def exact_distance(P, px, py):
    """Distance from (px, py) to the nearest point of the silhouette."""
    tp = math.atan2(py, px)
    k0 = math.floor((tp - P.rotation + P.half) / P.sigma)
    best = float("inf")
    for k in (k0 - 1, k0, k0 + 1):
        base = P.rotation + k * P.sigma
        for A, C, lo, hi in ((P.aR, P.cR, P.phi_w, P.half),
                             (P.aL, P.cL, -P.half, P.phi_w)):
            ang = base + P.phi_w
            ca, sa = math.cos(ang), math.sin(ang)
            qx, qy = px * ca + py * sa, -px * sa + py * ca
            for X, Y in _feet_on_conic(A, C, qx, qy):
                psi = math.atan2(Y, X)
                if not (lo - P.phi_w - 1e-12 <= psi <= hi - P.phi_w + 1e-12):
                    continue
                # Reject feet on the conic's conjugate branch.
                if abs(math.hypot(X, Y) * (A * math.cos(psi) + C) - 1.0) > 1e-7:
                    continue
                best = min(best, math.hypot(X - qx, Y - qy))
            for phi in (lo, hi):                          # constrained minima
                t = base + phi
                r = 1.0 / (A * math.cos(phi - P.phi_w) + C)
                best = min(best, math.hypot(r * math.cos(t) - px,
                                            r * math.sin(t) - py))
    return best


def first_order_distance(P, px, py):
    """What edge_gap() computes, in the oracle's own terms."""
    t = math.atan2(py, px)
    r = math.hypot(px, py)
    k = P.dr_dtheta(t) / r
    return (P.radius_at(t) - r) / math.sqrt(1.0 + k * k)


def main():
    n = 3600
    print(f"kick lives out to d = {LAMBDA_K_MM / 1.0e-5:.2f} mm of a "
          f"{SEMI_MM:.0f} mm semi-aperture; sampled where it is worth having\n")
    print(f"{'profile':17} {'sigma':>8} {'depth':>7} | {'median':>9} "
          f"{'over-state':>11} {'under-state':>12} | {'>10%':>6}")
    print("-" * 84)
    for name, kw in PROFILES:
        P = ApertureProfile(blades=6, **kw)
        for sigma in SIGMAS:
            depth = (LAMBDA_K_MM / sigma) / SEMI_MM
            errs, lo, hi = [], 0.0, 0.0
            for i in range(n):
                t = 2.0 * math.pi * i / n
                r = P.radius_at(t) * (1.0 - depth)
                px, py = r * math.cos(t), r * math.sin(t)
                truth = exact_distance(P, px, py)
                rel = (first_order_distance(P, px, py) - truth) / truth
                errs.append(abs(rel))
                lo, hi = min(lo, rel), max(hi, rel)
            errs.sort()
            over = sum(1 for e in errs if e > 0.10) / n
            print(f"{name:17} {sigma:8.0e} {depth * 100:6.3f}% | "
                  f"{errs[n // 2] * 100:8.4f}% {hi * 100:10.2f}% "
                  f"{-lo * 100:11.2f}% | {over * 100:5.2f}%")
    print("\nover-state = a WEAK kick (safe). under-state = a strong one.")


if __name__ == "__main__":
    main()
