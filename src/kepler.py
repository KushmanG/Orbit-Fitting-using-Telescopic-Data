"""
Two-body (Kepler) orbit machinery: solving Kepler's equation, converting between
orbital elements and position/velocity, and propagating an orbit in time.

Conventions
-----------
Elements are heliocentric, referred to the ecliptic and equinox of J2000 --
the same convention JPL uses, so we can compare numbers directly:
    a      semi-major axis [au]
    e      eccentricity
    i      inclination [rad]
    Om     longitude of the ascending node [rad]
    w      argument of perihelion [rad]
    M0     mean anomaly at the epoch t0 [rad]
Positions coming out of propagate() are in the EQUATORIAL (ICRF) frame, because
that is the frame RA/Dec live in.
"""
import numpy as np

from .constants import EPS_J2000, GM_SUN

_ce, _se = np.cos(EPS_J2000), np.sin(EPS_J2000)
# rotation that takes ecliptic J2000 vectors to equatorial (ICRF) vectors
ECL_TO_EQ = np.array([[1.0, 0.0, 0.0],
                      [0.0, _ce, -_se],
                      [0.0, _se, _ce]])
EQ_TO_ECL = ECL_TO_EQ.T


def solve_kepler(M, e, tol=1e-13, max_iter=60):
    """Solve E - e*sin(E) = M for E (vectorised Newton iteration)."""
    M = np.asarray(M, dtype=float)
    M = np.remainder(M + np.pi, 2 * np.pi) - np.pi      # keep M in [-pi, pi)
    E = M + 0.85 * e * np.sign(np.sin(M))               # Danby's starting guess, safe for any e < 1
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        step = f / (1.0 - e * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < tol:
            break
    return E


def perifocal_vectors(i, Om, w):
    """Unit vectors P (towards perihelion) and Q (90 deg ahead, in the orbit plane), ecliptic frame."""
    cO, sO, cw, sw, ci, si = np.cos(Om), np.sin(Om), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    P = np.array([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
    Q = np.array([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si])
    return P, Q


def propagate(el, t0, t, mu=GM_SUN, frame="eq"):
    """
    Heliocentric position(s) at time(s) t for an orbit with elements el at epoch t0.

    el = (a, e, i, Om, w, M0); t0, t in days (TDB Julian dates). Returns an (N, 3)
    array in au, in the equatorial frame by default (frame="ecl" for ecliptic).
    """
    a, e, i, Om, w, M0 = el
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = np.sqrt(mu / a**3)                    # mean motion [rad/day]
    E = solve_kepler(M0 + n * (t - t0), e)
    P, Q = perifocal_vectors(i, Om, w)
    x = a * (np.cos(E) - e)                   # coordinates inside the orbit plane
    y = a * np.sqrt(1.0 - e * e) * np.sin(E)
    r_ecl = np.outer(x, P) + np.outer(y, Q)
    return r_ecl @ ECL_TO_EQ.T if frame == "eq" else r_ecl


def elements_to_state(el, mu=GM_SUN):
    """(a, e, i, Om, w, M) -> position and velocity at that instant, ecliptic frame."""
    a, e, i, Om, w, M = el
    E = solve_kepler(np.array([M]), e)[0]
    P, Q = perifocal_vectors(i, Om, w)
    b = a * np.sqrt(1 - e * e)
    r = a * (np.cos(E) - e) * P + b * np.sin(E) * Q
    Edot = np.sqrt(mu / a**3) / (1 - e * np.cos(E))
    v = -a * np.sin(E) * Edot * P + b * np.cos(E) * Edot * Q
    return r, v


def state_to_elements(r, v, mu=GM_SUN):
    """Position/velocity (ecliptic frame, au and au/day) -> (a, e, i, Om, w, M)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, v2 = np.linalg.norm(r), v @ v
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    i = np.arccos(h[2] / hn)
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(node)
    evec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - v2 / mu)            # vis-viva
    Om = np.arctan2(node[1], node[0]) if nn > 1e-14 else 0.0
    if nn > 1e-14 and e > 1e-12:
        w = np.arctan2(np.dot(np.cross(node, evec), h) / hn, np.dot(node, evec))
    else:
        w = 0.0
    if e < 1.0:
        # eccentric anomaly from r.v and r, then mean anomaly
        cosE = (1 - rn / a) / e
        sinE = (r @ v) / (e * np.sqrt(mu * a))
        E = np.arctan2(sinE, cosE)
        M = E - e * np.sin(E)
    else:
        M = np.nan                             # unbound orbit; not something Eros does
    wrap = lambda x: np.remainder(x, 2 * np.pi)
    return np.array([a, e, i, wrap(Om), wrap(w), wrap(M)])


def specific_energy(r, v, mu=GM_SUN):
    """v^2/2 - mu/r. Negative means a bound (elliptical) orbit."""
    return 0.5 * (np.asarray(v) @ np.asarray(v)) - mu / np.linalg.norm(r)
