"""
Everything we do AFTER the two fits: compare with JPL, test predictions, probe
the non-convexity, and measure what each piece of realism buys us.

JPL appears here and only here. It is never fed into either fit.
"""
import time

import numpy as np

from . import easy, hard
from .constants import ARCSEC, AU_KM, DEG, GM_SUN
from .geometry import earth_heliocentric, light_time_positions, radec_to_unit, unit_to_radec, utc_to_tt_tdb
from .kepler import ECL_TO_EQ, EQ_TO_ECL, elements_to_state, propagate, state_to_elements

ELEMENT_NAMES = ["a", "e", "i", "Om", "w", "M0"]
ELEMENT_UNITS = ["au", "", "deg", "deg", "deg", "deg"]


def wrap_pi(x):
    return np.remainder(x + np.pi, 2 * np.pi) - np.pi


def separation_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle angle between two sets of directions, arcsec."""
    u1, u2 = radec_to_unit(ra1, dec1), radec_to_unit(ra2, dec2)
    return np.arctan2(np.linalg.norm(np.cross(u1, u2), axis=1), np.sum(u1 * u2, axis=1)) / ARCSEC


# ---------------------------------------------------------------------------------
# JPL reference values (Horizons tables)
# ---------------------------------------------------------------------------------

def jpl_elements_at(h_el, t0):
    row = h_el.iloc[int(np.argmin(np.abs(h_el.JDTDB.values - t0)))]
    assert abs(row.JDTDB - t0) < 1e-6, "pick an epoch that is a Horizons grid point (0h TDB)"
    return np.array([row.A, row.EC, row.IN * DEG, row.OM * DEG, row.W * DEG, row.MA * DEG])


def jpl_state_at(h_vec, t0):
    row = h_vec.iloc[int(np.argmin(np.abs(h_vec.JDTDB.values - t0)))]
    return row[["X", "Y", "Z"]].values.astype(float), row[["VX", "VY", "VZ"]].values.astype(float)


def element_table(fits, jpl_el, sigmas=None):
    """Rows = elements; columns = each fit, JPL, and differences (angles in degrees)."""
    rows = []
    for k, (name, unit) in enumerate(zip(ELEMENT_NAMES, ELEMENT_UNITS)):
        conv = (lambda v: np.degrees(v)) if unit == "deg" else (lambda v: v)
        row = dict(element=name, unit=unit, jpl=float(conv(jpl_el[k])))
        for label, el in fits.items():
            d = el[k] - jpl_el[k]
            if unit == "deg":
                d = wrap_pi(d)
            row[label] = float(conv(el[k]))
            row[f"{label}_minus_jpl"] = float(conv(d))
            if sigmas is not None and label in sigmas:
                s = sigmas[label][k]
                row[f"{label}_sigma"] = float(conv(s))
                row[f"{label}_diff_in_sigma"] = float(d / s) if s > 0 else np.nan
        rows.append(row)
    return rows


def position_error_km(el, t0, r_jpl_eq):
    """Distance between where an orbit puts Eros at t0 and where JPL does."""
    return float(np.linalg.norm(propagate(el, t0, [t0])[0] - r_jpl_eq) * AU_KM)


# ---------------------------------------------------------------------------------
# Predictions (the "so what": where do we point the telescope next?)
# ---------------------------------------------------------------------------------

def geocentric_radec(el, t0, jd_utc):
    """Predicted astrometric RA/Dec seen from Earth's centre (same thing Horizons lists)."""
    _, tdb = utc_to_tt_tdb(jd_utc)
    R = earth_heliocentric(tdb)
    rho, _ = light_time_positions(lambda t: propagate(el, t0, t), tdb, R)
    return unit_to_radec(rho)


def prediction_error_curve(el, t0, h_radec, jd_utc_grid):
    """Angle between our predicted position and JPL's, at each date in the grid [arcsec]."""
    ra_p, dec_p = geocentric_radec(el, t0, jd_utc_grid)
    jd = h_radec.jd_utc.values
    idx = np.array([int(np.argmin(np.abs(jd - x))) for x in jd_utc_grid])
    ra_j, dec_j = h_radec.ra_deg.values[idx] * DEG, h_radec.dec_deg.values[idx] * DEG
    return separation_arcsec(ra_p, dec_p, ra_j, dec_j)


def out_of_sample(el, t0, t, ra, dec, R):
    """Residuals of LATER, real observations that the fit never saw."""
    res = hard.angular_residuals(el, t0, t, ra, dec, R)
    return res


# ---------------------------------------------------------------------------------
# Non-convexity: multi-start and slices
# ---------------------------------------------------------------------------------

def random_starts(n, rng):
    """Broad, physically-reasonable box of starting orbits (a near-Earth asteroid could be anywhere here)."""
    return np.column_stack([
        rng.uniform(0.8, 3.0, n),           # a  [au]
        rng.uniform(0.0, 0.6, n),           # e
        rng.uniform(0.0, 40.0, n) * DEG,    # i
        rng.uniform(0, 360, n) * DEG,       # Om
        rng.uniform(0, 360, n) * DEG,       # w
        rng.uniform(0, 360, n) * DEG,       # M0
    ])


def multistart(t0, t, ra, dec, R, sigma, used, best_el, n=60, seed=7, max_nfev=200):
    """
    Run the hard solver from n random starting orbits (outliers already removed,
    weights fixed) and record where each one ends up. If the problem were convex,
    every run would end at the same answer.
    """
    rng = np.random.default_rng(seed)
    starts = random_starts(n, rng)
    out = []
    for k, th0 in enumerate(starts):
        tic = time.perf_counter()
        try:
            s = hard.solve(th0, t0, t[used], ra[used], dec[used], R[used], sigma[used],
                           reject=False, max_nfev=max_nfev)
            el, rms, nfev = s["elements"], s["rms_arcsec"], s["nfev"]
        except Exception:                                  # a start so bad the model blows up
            el, rms, nfev = np.full(6, np.nan), np.inf, max_nfev
        d = el - best_el
        d[2:] = wrap_pi(d[2:])
        out.append(dict(start=k, start_elements=th0.tolist(), final_elements=el.tolist(),
                        rms_arcsec=float(rms), nfev=int(nfev), seconds=time.perf_counter() - tic,
                        da=float(d[0]), max_angle_diff_deg=float(np.degrees(np.nanmax(np.abs(d[2:]))))))
    return out


def objective_slice(theta, k, values, t0, t, ra, dec, R, sigma, used):
    """chi^2 of the hard problem as ONE element sweeps through `values`, the rest held at theta."""
    out = []
    for v in values:
        th = np.array(theta, float)
        th[k] = v
        out.append(hard.chi2_at(th, t0, t, ra, dec, R, sigma, used))
    return np.array(out)


# ---------------------------------------------------------------------------------
# Price of realism: switch individual pieces of physics off
# ---------------------------------------------------------------------------------

def ablation(theta_seed, t0, sigma_station, t, ra, dec, R, keep):
    """
    The full hard model, then the same fit with ONE ingredient removed at a time.
    Every variant is fitted to the same kept observations (keep = the set the full
    model kept), except "no outlier rejection", which fits everything. Each variant
    is then judged the same way: RMS of the TRUE (full-model) misses on the kept set.
    """
    R_geo = earth_heliocentric(t)      # Earth's centre instead of the actual telescope
    k = keep
    fits = {
        "full hard model": hard.solve(theta_seed, t0, t[k], ra[k], dec[k], R[k], sigma_station[k], reject=False),
        "no light-time correction": hard.solve(theta_seed, t0, t[k], ra[k], dec[k], R[k], sigma_station[k],
                                               light_time=False, reject=False),
        "telescope at Earth's centre": hard.solve(theta_seed, t0, t[k], ra[k], dec[k], R_geo[k], sigma_station[k],
                                                  reject=False),
        "equal weights for all telescopes": hard.solve(theta_seed, t0, t[k], ra[k], dec[k], R[k], None, reject=False),
        "no outlier rejection": hard.solve(theta_seed, t0, t, ra, dec, R, sigma_station, reject=False),
    }
    rows = []
    for name, s in fits.items():
        r = hard.angular_residuals(s["elements"], t0, t[k], ra[k], dec[k], R[k])
        rows.append(dict(variant=name, rms_arcsec=float(np.sqrt(np.mean(r ** 2))), elements=s["elements"].tolist(),
                         n_used_in_fit=s["n_used"], nfev=s["nfev"]))
    return rows


# ---------------------------------------------------------------------------------
# How much sky do we need? (arc-length study)
# ---------------------------------------------------------------------------------

def arc_length_study(days_list, t0_full, t, ra, dec, R, sigma, jpl_r_at, jd_tdb_start):
    """
    Fit only the first L days of observations, for several L. Report how far each
    answer is from JPL and how big its own error bar claims to be.
    """
    rows = []
    for L in days_list:
        m = t < jd_tdb_start + L
        if m.sum() < 6:
            continue
        tt = t[m]
        t0 = np.round(np.median(tt)) - 0.5
        r_jpl = jpl_r_at(t0)
        # A sub-arc of only a few days can't pin an orbit: the easy solver may hand back a
        # hyperbolic (a < 0) junk seed, and the hard solve may not converge. We record inf
        # for those, which is the honest result -- that is what "too short an arc" looks like.
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                e = easy.solve(tt, ra[m], dec[m], R[m], t0)
                easy_err = position_error_km(e["elements"], t0, r_jpl)
            except Exception:
                e, easy_err = dict(elements=np.array([1.5, 0.2, 0.2, 0.0, 0.0, 0.0])), np.inf
            try:
                h = hard.solve(e["elements"], t0, tt, ra[m], dec[m], R[m], sigma[m])
                hard_err, hard_a = position_error_km(h["elements"], t0, r_jpl), float(h["elements"][0])
                hard_sa, hard_rms = float(h["sigma_elements"][0]), h["rms_arcsec"]
            except Exception:
                hard_err, hard_a, hard_sa, hard_rms = np.inf, np.nan, np.nan, np.inf
        rows.append(dict(days=float(L), n_obs=int(m.sum()), t0=float(t0),
                         easy_pos_err_km=easy_err, hard_pos_err_km=hard_err,
                         hard_a=hard_a, hard_sigma_a=hard_sa, hard_rms=hard_rms))
    return rows


# ---------------------------------------------------------------------------------
# Model floor: how well can a two-body orbit match JPL's full n-body orbit at all?
# ---------------------------------------------------------------------------------

def two_body_floor(h_vec, t_lo, t_hi, t0):
    """
    Fit a two-body orbit straight to JPL's own positions over our window (no telescope,
    no noise). Whatever mismatch is left is the price of ignoring the planets.
    """
    from scipy.optimize import least_squares
    m = (h_vec.JDTDB.values >= t_lo) & (h_vec.JDTDB.values <= t_hi)
    tj = h_vec.JDTDB.values[m]
    rj = h_vec[["X", "Y", "Z"]].values[m]
    r0, v0 = jpl_state_at(h_vec, t0)
    el0 = state_to_elements(EQ_TO_ECL @ r0, EQ_TO_ECL @ v0)
    fun = lambda th: ((propagate(th, t0, tj) - rj) * AU_KM).ravel()
    sol = least_squares(fun, el0, method="lm", x_scale="jac")
    resid_km = np.linalg.norm((propagate(sol.x, t0, tj) - rj) * AU_KM, axis=1)
    return dict(elements=sol.x.tolist(), max_km=float(resid_km.max()), rms_km=float(np.sqrt(np.mean(resid_km**2))),
                osculating_elements=el0.tolist())
