"""
Where the telescope is, and which way it is looking.

Everything here is in the equatorial ICRF frame (the frame RA/Dec are measured in),
heliocentric, in au. The heavy lifting (Earth's orbit, Earth's rotation, precession
and nutation) is done by ERFA, the open-source version of the IAU's SOFA library,
which astropy ships with.
"""
import warnings

import erfa
import numpy as np
from astropy.time import Time

from .constants import AU_KM, C_AU_DAY, R_EARTH_KM

warnings.filterwarnings("ignore", module="erfa")     # "dubious year" chatter for future dates


def utc_to_tt_tdb(jd_utc):
    """UTC Julian dates -> (TT, TDB) Julian dates. TDB is the time our orbit model runs on."""
    t = Time(np.asarray(jd_utc, float), format="jd", scale="utc")
    return t.tt.jd, t.tdb.jd


def earth_heliocentric(jd_tdb):
    """Heliocentric position of Earth's centre [au], shape (N, 3), ICRF axes (ERFA epv00)."""
    jd = np.atleast_1d(np.asarray(jd_tdb, float))
    pvh, _ = erfa.epv00(np.floor(jd), jd - np.floor(jd))
    return pvh["p"]          # pyerfa hands back a structured array with fields p, v


def site_geocentric(jd_utc, jd_tt, lon_deg, rho_cos, rho_sin):
    """
    Geocentric position of an observatory [au] in ICRF axes.

    lon_deg, rho_cos, rho_sin are the MPC parallax constants (longitude east of
    Greenwich, and rho*cos(phi'), rho*sin(phi') in Earth radii). We rotate the
    Earth-fixed vector into the sky frame with the IAU 2006/2000A matrices.
    UT1 is taken equal to UTC: the difference (< 0.9 s) moves a site by < 0.5 km,
    which is a few milli-arcseconds from 0.3 au away.
    """
    jd_utc = np.atleast_1d(np.asarray(jd_utc, float))
    jd_tt = np.atleast_1d(np.asarray(jd_tt, float))
    lon = np.radians(np.asarray(lon_deg, float))
    r_terr = np.stack([rho_cos * np.cos(lon), rho_cos * np.sin(lon), rho_sin * np.ones_like(lon)], -1)
    r_terr = r_terr * (R_EARTH_KM / AU_KM)
    rc2t = erfa.c2t06a(np.floor(jd_tt), jd_tt - np.floor(jd_tt),
                       np.floor(jd_utc), jd_utc - np.floor(jd_utc), 0.0, 0.0)
    # celestial = rc2t^T @ terrestrial
    return np.einsum("nji,nj->ni", rc2t, r_terr)


def radec_to_unit(ra, dec):
    """RA/Dec [rad] -> unit vectors, shape (N, 3)."""
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], -1)


def unit_to_radec(u):
    """Vectors (any length) -> RA in [0, 2pi) and Dec, both radians."""
    u = np.atleast_2d(u)
    ra = np.remainder(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    dec = np.arctan2(u[:, 2], np.hypot(u[:, 0], u[:, 1]))
    return ra, dec


def sky_basis(ra, dec):
    """
    Two unit vectors perpendicular to the line of sight: east (direction of
    increasing RA) and north (increasing Dec). Shape (N, 2, 3).
    """
    sa, ca, sd, cd = np.sin(ra), np.cos(ra), np.sin(dec), np.cos(dec)
    east = np.stack([-sa, ca, np.zeros_like(ra)], -1)
    north = np.stack([-sd * ca, -sd * sa, cd], -1)
    return np.stack([east, north], 1)


def light_time_positions(pos_fn, t_obs, R_obs, n_iter=3):
    """
    Light from the asteroid takes a few minutes to reach us, so the direction we
    measure at t_obs points to where it WAS at t_obs - tau. pos_fn(t) gives the
    heliocentric position; we iterate tau = |r(t - tau) - R| / c a few times
    (it converges to well below a millisecond in 2-3 steps).
    Returns the topocentric vectors rho (N, 3) and tau [days].
    """
    tau = np.zeros_like(t_obs)
    for _ in range(n_iter):
        rho = pos_fn(t_obs - tau) - R_obs
        tau = np.linalg.norm(rho, axis=1) / C_AU_DAY
    return rho, tau
