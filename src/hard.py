"""
The HARD problem: the real, non-convex orbit fit.

Unknowns: the six orbital elements theta = (a, e, i, Om, w, M0) at epoch t0.

Model for observation i (all exact, no series):
  1. propagate the orbit with Kepler's equation   E - e sin E = M0 + n (t - t0)
  2. step back by the light-travel time:  the light we see at t_i left Eros at t_i - tau_i
  3. look from the actual telescope on the spinning Earth (topocentric), not Earth's centre
  4. turn the vector into predicted (RA, Dec)

Objective: weighted sum of squared ANGULAR misses, in arcsec,
        chi2(theta) = sum_i [ (dRA_i cos Dec_i)^2 + dDec_i^2 ] / sigma_i^2
Constraints:  a > 0,  0 <= e < 1  (a bound, elliptical orbit),  0 <= i <= 180 deg.

Why it is non-convex: Kepler's equation and the trig in steps 1-4 make the
predicted angles a curved, wrapping function of theta. The sum of squares of a
non-linear function has no reason to be a single bowl -- and we check that
numerically with multi-start runs (analysis.multistart).

Solver: scipy's trust-region-reflective least squares ('trf'), which handles the
bounds and is a damped Gauss-Newton method under the hood.
"""
import time

import numpy as np
from scipy.optimize import least_squares

from .constants import ARCSEC, DEG
from .geometry import light_time_positions, unit_to_radec
from .kepler import propagate

LOWER = np.array([0.05, 0.0, 0.0, -np.inf, -np.inf, -np.inf])
UPPER = np.array([50.0, 0.999, np.pi, np.inf, np.inf, np.inf])


def predict_radec(theta, t0, t, R, light_time=True):
    pos = lambda tt: propagate(theta, t0, tt)
    if light_time:
        rho, _ = light_time_positions(pos, t, R)
    else:
        rho = pos(t) - R
    return unit_to_radec(rho)


def angular_residuals(theta, t0, t, ra, dec, R, light_time=True):
    """Observed minus predicted, in arcsec: (dRA*cos(Dec), dDec), shape (N, 2)."""
    ra_p, dec_p = predict_radec(theta, t0, t, R, light_time)
    dra = np.remainder(ra - ra_p + np.pi, 2 * np.pi) - np.pi       # wrap to [-pi, pi)
    return np.stack([dra * np.cos(dec), dec - dec_p], 1) / ARCSEC


def _fit_once(theta0, t0, t, ra, dec, R, sigma, light_time, max_nfev):
    def fun(th):
        r = angular_residuals(th, t0, t, ra, dec, R, light_time) / sigma[:, None]
        return r.ravel()
    # a short-arc seed can be hyperbolic or NaN; make it finite and in-bounds first
    x0 = np.where(np.isfinite(theta0), theta0, [1.5, 0.2, 0.2, 0.0, 0.0, 0.0])
    x0 = np.clip(x0, LOWER + 1e-9, UPPER - 1e-9)
    return least_squares(fun, x0, bounds=(LOWER, UPPER), method="trf", jac="3-point",
                         x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_nfev)


def solve(theta0, t0, t, ra, dec, R, sigma=None, light_time=True, reject=True,
          clip=3.0, max_rounds=10, max_nfev=400):
    """
    Fit the six elements. If reject=True we do the standard astrometric clean-up:
    fit, flag observations whose residual is more than `clip` sigma in RA or Dec,
    refit without them, and repeat until the set of flagged points stops changing
    (a flagged point can come back if the new orbit fits it).
    """
    n = len(t)
    sigma = np.ones(n) if sigma is None else np.asarray(sigma, float)
    use = np.ones(n, bool)
    tic = time.perf_counter()
    theta, nfev, rounds = np.asarray(theta0, float), 0, []
    for k in range(max_rounds if reject else 1):
        sol = _fit_once(theta, t0, t[use], ra[use], dec[use], R[use], sigma[use], light_time, max_nfev)
        theta, nfev, fitted = sol.x, nfev + sol.nfev, use.copy()
        resid = angular_residuals(theta, t0, t, ra, dec, R, light_time)
        z = np.abs(resid) / sigma[:, None]
        new_use = np.all(z < clip, axis=1)
        rounds.append(dict(round=k, n_used=int(use.sum()), chi2=float(2 * sol.cost), nfev=int(sol.nfev)))
        if not reject or np.array_equal(new_use, use):
            break
        if new_use.sum() < max(12, n // 2):
            # Rejecting more than half the data means the MODEL is wrong, not the data.
            # Stop clipping and keep what we had (this is what a misspecified model looks like).
            rounds[-1]["stopped"] = "would reject more than half the data"
            break
        use = new_use
    elapsed = time.perf_counter() - tic

    use = fitted                                  # the set the final fit actually used
    resid = angular_residuals(theta, t0, t, ra, dec, R, light_time)
    J = sol.jac                                   # Jacobian of the normalised residuals
    r_used = (resid[use] / sigma[use, None]).ravel()
    chi2 = float(r_used @ r_used)
    dof = r_used.size - 6
    JTJ = J.T @ J
    with np.errstate(invalid="ignore"):           # a diverged multi-start can give a singular JTJ
        cov = np.linalg.pinv(JTJ) * (chi2 / dof)  # scale by reduced chi^2 (weights are relative)
    grad = J.T @ r_used
    # unit-free first-order check: cosine between the residual vector and each column of J.
    # At a minimum the residuals are orthogonal to every direction we can move in -> all ~0.
    grad_cos = np.abs(grad) / (np.linalg.norm(J, axis=0) * np.linalg.norm(r_used))
    return dict(
        elements=theta, t0=t0, success=bool(sol.success), status=int(sol.status), message=sol.message,
        chi2=chi2, dof=dof, reduced_chi2=chi2 / dof,
        rms_arcsec=float(np.sqrt(np.mean(resid[use] ** 2))),
        rms_ra=float(np.sqrt(np.mean(resid[use, 0] ** 2))),
        rms_dec=float(np.sqrt(np.mean(resid[use, 1] ** 2))),
        residuals=resid, used=use, n_used=int(use.sum()), n_rejected=int((~use).sum()),
        cov=cov, sigma_elements=np.sqrt(np.abs(np.diag(cov))),
        corr=(cov / np.outer(np.sqrt(np.abs(np.diag(cov))), np.sqrt(np.abs(np.diag(cov))) + 1e-300)),
        grad_norm=float(np.linalg.norm(grad)), grad_cos_max=float(grad_cos.max()),
        jtj_eigenvalues=np.linalg.eigvalsh(JTJ).tolist(),
        nfev=int(nfev), rounds=rounds, seconds=elapsed, sigma=sigma)


def station_sigmas(stations, resid, used, floor=0.15, min_count=5):
    """
    Not every telescope is equally good. Give each station a sigma equal to the
    RMS of its own residuals from a first, unweighted fit (with a floor so no
    station gets absurd weight). Stations with very few points get the median.
    """
    stations = np.asarray(stations)
    rms = {}
    for s in np.unique(stations):
        m = (stations == s) & used
        if m.sum() >= min_count:
            rms[s] = max(float(np.sqrt(np.mean(resid[m] ** 2))), floor)
    med = float(np.median(list(rms.values()))) if rms else 1.0
    return np.array([rms.get(s, med) for s in stations]), rms


def chi2_at(theta, t0, t, ra, dec, R, sigma, used=None, light_time=True):
    """The hard objective evaluated at any theta (used for slices and comparisons)."""
    r = angular_residuals(theta, t0, t, ra, dec, R, light_time) / sigma[:, None]
    if used is not None:
        r = r[used]
    return float(np.sum(r * r))
