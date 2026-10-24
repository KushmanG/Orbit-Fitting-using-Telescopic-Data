"""
The EASY problem: a convex (linear least-squares) orbit fit.

Idea. The telescope gives a direction u_i. The asteroid must sit somewhere on that
line of sight:   r(t_i) = R_i + rho_i * u_i,   with the distance rho_i unknown.
Equivalently, the part of (r(t_i) - R_i) that is PERPENDICULAR to u_i must be zero.
Take two unit vectors perpendicular to the line of sight (east and north, the rows
of B_i); then
        e_i = B_i (r(t_i) - R_i)          (2 numbers, in au)
is how far the orbit passes beside the line of sight. We want all e_i ~ 0.

To make the problem linear we approximate the motion with the first terms of the
Lagrange f-g series around the epoch t0 (dt = t - t0):
        r(t) ~ f(dt) r0 + g(dt) v0
        f = 1 - mu dt^2 / (2 rbar^3),      g = dt - mu dt^3 / (6 rbar^3)
where rbar is the Sun-asteroid distance, FROZEN to a number. With rbar fixed,
r(t) is linear in the six unknowns x = (r0, v0), so every e_i is linear in x:

        minimise_x  F(x) = sum_i || B_i (f_i r0 + g_i v0 - R_i) ||^2  =  || A x - b ||^2

That is an unconstrained convex quadratic (its Hessian 2 A^T A is positive
semidefinite), so the global minimum comes from one linear solve -- no starting
guess, no iterations inside the solve.

The only loose end is rbar: the answer has to be self-consistent, |r0| = rbar.
We scan rbar over 0.3-5 au (one convex solve per value) and find where the
mismatch |r0(rbar)| - rbar crosses zero. Typically there are THREE such roots --
the classic ambiguity of Laplace/Gauss orbit determination. One of them is the
"observer root": an orbit hugging the Earth (a few hundredths of an au away). It
has a tiny algebraic miss precisely because a miss measured in au shrinks when the
object is close to us. We keep the root whose misses are smallest as ANGLES (miss
divided by that candidate's own distance), which is what a telescope measures.

What the easy model ignores (on purpose): the exact Kepler motion (the series is
truncated), light-travel time, and it scores misses in au instead of arcsec.
"""
import time

import numpy as np
from scipy.optimize import brentq

from .constants import ARCSEC, AU_KM, GM_SUN
from .geometry import sky_basis
from .kepler import EQ_TO_ECL, state_to_elements

RBAR_GRID = np.arange(0.30, 5.0001, 0.01)


def build_system(t, ra, dec, R, t0, rbar, mu=GM_SUN):
    """Stack the 2x6 blocks A_i and the 2-vectors b_i into A (2N x 6) and b (2N)."""
    dt = t - t0
    k = mu / rbar**3
    f = 1.0 - 0.5 * k * dt**2
    g = dt - k * dt**3 / 6.0
    B = sky_basis(ra, dec)                               # (N, 2, 3): east and north rows
    A = np.concatenate([f[:, None, None] * B, g[:, None, None] * B], axis=2)   # (N, 2, 6)
    b = np.einsum("nkj,nj->nk", B, R)                    # (N, 2)
    return A.reshape(-1, 6), b.reshape(-1)


def convex_solve(t, ra, dec, R, t0, rbar, mu=GM_SUN):
    """The convex step: for one fixed rbar, the global minimiser of ||Ax - b||^2."""
    A, b = build_system(t, ra, dec, R, t0, rbar, mu)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, A, b


def model_ranges(x, t, R, t0, rbar, mu=GM_SUN):
    """Distance from each telescope to where the easy model puts the asteroid [au]."""
    dt = t - t0
    k = mu / rbar**3
    r = np.outer(1 - 0.5 * k * dt**2, x[:3]) + np.outer(dt - k * dt**3 / 6, x[3:])
    return np.linalg.norm(r - R, axis=1)


def angular_misses(x, A, b, t, R, t0, rbar):
    """Algebraic misses (au) divided by the candidate's own distance -> arcsec, shape (N, 2)."""
    res = (A @ x - b).reshape(-1, 2)
    return res / model_ranges(x, t, R, t0, rbar)[:, None] / ARCSEC


def solve(t, ra, dec, R, t0, grid=RBAR_GRID, mu=GM_SUN):
    """
    Solve the easy problem. Returns the chosen candidate (state, elements,
    objective), all self-consistent candidates, and the rbar scan for plotting.
    """
    tic = time.perf_counter()
    mismatch, F = np.empty(grid.size), np.empty(grid.size)
    for k, rb in enumerate(grid):
        x, A, b = convex_solve(t, ra, dec, R, t0, rb, mu)
        mismatch[k] = np.linalg.norm(x[:3]) - rb
        F[k] = np.sum((A @ x - b) ** 2)

    g = lambda rb: np.linalg.norm(convex_solve(t, ra, dec, R, t0, rb, mu)[0][:3]) - rb
    roots = [brentq(g, grid[k], grid[k + 1], xtol=1e-13)
             for k in range(grid.size - 1) if np.sign(mismatch[k]) != np.sign(mismatch[k + 1])]
    if not roots:                                         # no crossing: take the closest approach
        roots = [float(grid[np.argmin(np.abs(mismatch))])]

    cands = []
    for rb in roots:
        x, A, b = convex_solve(t, ra, dec, R, t0, rb, mu)
        ang = angular_misses(x, A, b, t, R, t0, rb)
        cands.append(dict(rbar=rb, x=x, objective_au2=float(np.sum((A @ x - b) ** 2)),
                          angular_rms=float(np.sqrt(np.mean(ang ** 2))),
                          geocentric_au=float(np.median(model_ranges(x, t, R, t0, rb)))))
    best = min(cands, key=lambda c: c["angular_rms"])
    x, A, b = convex_solve(t, ra, dec, R, t0, best["rbar"], mu)
    elapsed = time.perf_counter() - tic

    r0, v0 = x[:3], x[3:]
    el = state_to_elements(EQ_TO_ECL @ r0, EQ_TO_ECL @ v0, mu)
    res = A @ x - b
    return dict(x=x, r0=r0, v0=v0, elements=el, rbar=best["rbar"], t0=t0,
                objective_au2=float(res @ res), rms_miss_km=float(np.sqrt(np.mean(res ** 2)) * AU_KM),
                candidates=cands, scan=dict(rbar=grid, mismatch=mismatch, objective_au2=F),
                n_convex_solves=int(grid.size + 40 * len(roots)), seconds=elapsed, A=A, b=b)


def optimality_report(A, b, x):
    """
    Numbers that PROVE x is the global minimum of ||Ax - b||^2 (for its rbar):
      * gradient 2 A^T (A x - b) should be ~0 (first-order condition),
      * all eigenvalues of the Hessian 2 A^T A should be > 0 (strictly convex,
        so the stationary point is the unique global minimum).
    Columns are scaled to unit length first so the numbers are unit-free.
    """
    scale = np.linalg.norm(A, axis=0)
    As = A / scale
    xs = x * scale
    grad = 2 * As.T @ (As @ xs - b)
    H = 2 * As.T @ As
    eig = np.linalg.eigvalsh(H)
    return dict(grad_norm=float(np.linalg.norm(grad)),
                grad_rel=float(np.linalg.norm(grad) / (np.linalg.norm(2 * As.T @ b) + 1e-300)),
                hessian_eigenvalues=eig.tolist(), condition_number=float(eig[-1] / eig[0]),
                min_eigenvalue=float(eig[0]))
