#!/usr/bin/env python3
"""
run_all.py -- the whole project, start to finish.

    python3 run_all.py            # everything (a few minutes, mostly the multi-start test)
    python3 run_all.py --quick    # skip the slow non-convexity experiments

Needs the raw files from fetch_data.py in data/raw/. Writes:
    data/processed/observations_fit.csv     the cleaned observations we fit
    results/results.json                     every number quoted in the slides/report
    results/*.csv                            the tables
    figures/*_slide.png, figures/*_paper.png the plots (two colour themes)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from astropy.time import Time

from src import analysis, data, easy, hard, plots
from src.constants import ARCSEC, AU_KM, DEG
from src.geometry import earth_heliocentric, unit_to_radec
from src.kepler import EQ_TO_ECL, propagate, state_to_elements

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PROCESSED = os.path.join(HERE, "data", "processed")

# ---- the choices we made (see the report for why) --------------------------------
FIT_START, FIT_END = "2025-10-15", "2025-12-15"   # 61 nights around closest approach (0.40 au)
LATER_END = "2026-07-01"                           # later observations: only used to TEST predictions
THEMES = ("slide", "paper")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main(quick=False):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    out = {"config": dict(fit_start=FIT_START, fit_end=FIT_END, later_end=LATER_END)}
    tic_all = time.perf_counter()

    # ---------------------------------------------------------------- data
    print("loading MPC observations ...")
    allobs = data.load_mpc_observations()
    codes = data.load_obscodes()
    out["dataset"] = dict(
        total_observations=int(len(allobs)),
        first=Time(allobs.jd_utc.min(), format="jd").iso[:10],
        last=Time(allobs.jd_utc.max(), format="jd").iso[:10],
        by_type={str(k): int(v) for k, v in allobs.note2.value_counts().items()},
        n_obscodes=int(len(codes)))
    in_window = allobs[(allobs.jd_utc >= Time(FIT_START).jd) & (allobs.jd_utc < Time(FIT_END).jd)]
    fit = data.attach_geometry(data.clean_observations(allobs, FIT_START, FIT_END, codes), codes)
    later = data.attach_geometry(data.clean_observations(allobs, FIT_END, LATER_END, codes), codes)
    fit.to_csv(os.path.join(PROCESSED, "observations_fit.csv"), index=False)
    t, ra, dec, u, R = data.arrays(fit)
    t0 = float(np.round(np.median(t)) - 0.5)          # 0h TDB nearest the middle of the arc
    days = t - t.min()
    out["window"] = dict(
        in_window_any_type=int(len(in_window)), used=int(len(fit)), later_used=int(len(later)),
        nights=int(np.unique(np.floor(fit.jd_utc - 0.5)).size), stations=int(fit.stn.nunique()),
        epoch_tdb=t0, epoch_iso=Time(t0, format="jd", scale="tdb").iso[:16],
        with_ades_rms=int(np.isfinite(fit.rmsRA).sum()))
    print(f"  {len(fit)} observations, {fit.stn.nunique()} stations, epoch {out['window']['epoch_iso']} TDB")

    # ---------------------------------------------------------------- EASY
    print("easy problem (convex) ...")
    E = easy.solve(t, ra, dec, R, t0)
    E_opt = easy.optimality_report(E["A"], E["b"], E["x"])
    easy_res = hard.angular_residuals(E["elements"], t0, t, ra, dec, R)   # judged by the real yardstick
    out["easy"] = dict(
        elements=E["elements"].tolist(), rbar=E["rbar"], objective_au2=E["objective_au2"],
        rms_miss_km=E["rms_miss_km"], seconds=E["seconds"], n_convex_solves=E["n_convex_solves"],
        candidates=[{k: v for k, v in c.items() if k != "x"} for c in E["candidates"]],
        optimality=E_opt, angular_rms_arcsec=rms(easy_res))
    print(f"  rbar = {E['rbar']:.4f} au, angular RMS of easy orbit = {rms(easy_res):.1f}\"")

    # ---------------------------------------------------------------- HARD
    print("hard problem (non-convex), pass 1: equal weights ...")
    H1 = hard.solve(E["elements"], t0, t, ra, dec, R, sigma=None)
    sig, station_rms = hard.station_sigmas(fit.stn.values, H1["residuals"], H1["used"])
    print("hard problem, pass 2: per-telescope weights ...")
    H = hard.solve(H1["elements"], t0, t, ra, dec, R, sigma=sig)
    out["hard"] = dict(
        elements=H["elements"].tolist(), sigma_elements=H["sigma_elements"].tolist(),
        chi2=H["chi2"], dof=H["dof"], reduced_chi2=H["reduced_chi2"],
        rms_arcsec=H["rms_arcsec"], rms_ra=H["rms_ra"], rms_dec=H["rms_dec"],
        mean_ra=float(np.mean(H["residuals"][H["used"], 0])), mean_dec=float(np.mean(H["residuals"][H["used"], 1])),
        n_used=H["n_used"], n_rejected=H["n_rejected"], nfev=H["nfev"] + H1["nfev"],
        seconds=H["seconds"] + H1["seconds"], grad_norm=H["grad_norm"], grad_cos_max=H["grad_cos_max"],
        jtj_eigenvalues=H["jtj_eigenvalues"], correlation=H["corr"].tolist(),
        rounds_pass1=H1["rounds"], rounds_pass2=H["rounds"], pass1_rms=H1["rms_arcsec"],
        station_sigma=station_rms, success=H["success"], message=str(H["message"]))
    print(f"  RMS {H['rms_arcsec']:.3f}\", {H['n_rejected']} rejected, reduced chi2 {H['reduced_chi2']:.2f}")

    # elements -> state for the record
    out["hard"]["chi2_at_easy"] = hard.chi2_at(E["elements"], t0, t, ra, dec, R, sig, H["used"])

    # ---------------------------------------------------------------- JPL (checking only)
    print("comparing with JPL ...")
    h_el, h_vec = data.load_horizons_elements(), data.load_horizons_vectors()
    h_radec, h_earth = data.load_horizons_radec(), data.load_horizons_vectors("horizons_earth_vectors.txt")
    jpl_el = analysis.jpl_elements_at(h_el, t0)
    r_jpl, v_jpl = analysis.jpl_state_at(h_vec, t0)
    table = analysis.element_table({"easy": E["elements"], "hard": H["elements"]}, jpl_el,
                                   sigmas={"hard": H["sigma_elements"]})
    pd.DataFrame(table).to_csv(os.path.join(RESULTS, "elements_vs_jpl.csv"), index=False)
    out["jpl"] = dict(elements=jpl_el.tolist(), state=[r_jpl.tolist(), v_jpl.tolist()], table=table,
                      easy_pos_err_km=analysis.position_error_km(E["elements"], t0, r_jpl),
                      hard_pos_err_km=analysis.position_error_km(H["elements"], t0, r_jpl),
                      jpl_chi2=hard.chi2_at(jpl_el, t0, t, ra, dec, R, sig, H["used"]),
                      jpl_rms_on_our_data=rms(hard.angular_residuals(jpl_el, t0, t, ra, dec, R)[H["used"]]))

    # sanity checks of our own geometry against JPL
    e_ours = earth_heliocentric(h_earth.JDTDB.values)
    out["checks"] = dict(earth_position_max_km=float(np.max(np.linalg.norm(
        e_ours - h_earth[["X", "Y", "Z"]].values, axis=1)) * AU_KM))
    jpl_from_state = state_to_elements(EQ_TO_ECL @ r_jpl, EQ_TO_ECL @ v_jpl)
    grid_chk = Time(FIT_START).jd + np.arange(0, 62, 1.0)
    ra_c, dec_c = analysis.geocentric_radec(jpl_from_state, t0, grid_chk)
    jd = h_radec.jd_utc.values
    idx = [int(np.argmin(np.abs(jd - x))) for x in grid_chk]
    sep = analysis.separation_arcsec(ra_c, dec_c, h_radec.ra_deg.values[idx] * DEG, h_radec.dec_deg.values[idx] * DEG)
    out["checks"]["forward_model_vs_horizons_max_arcsec"] = float(sep.max())
    out["checks"]["forward_model_vs_horizons_rms_arcsec"] = rms(sep)
    floor = analysis.two_body_floor(h_vec, t.min(), t.max(), t0)
    out["checks"]["two_body_floor_km"] = dict(max=floor["max_km"], rms=floor["rms_km"])

    # ---------------------------------------------------------------- predictions
    print("predictions ...")
    end_tdb = t.max()
    grid = Time(FIT_END).jd + np.arange(0, 366, 5.0)
    err_e = analysis.prediction_error_curve(E["elements"], t0, h_radec, grid)
    err_h = analysis.prediction_error_curve(H["elements"], t0, h_radec, grid)
    days_after = grid - Time(FIT_END).jd
    out["prediction"] = dict(days_after=days_after.tolist(), easy_arcsec=err_e.tolist(), hard_arcsec=err_h.tolist())
    oos = None
    if len(later):
        tl, ral, decl, ul, Rl = data.arrays(later)
        res_lh = analysis.out_of_sample(H["elements"], t0, tl, ral, decl, Rl)
        res_le = analysis.out_of_sample(E["elements"], t0, tl, ral, decl, Rl)
        month = np.floor((tl - Time(FIT_END).jd) / 30.0)
        rows = []
        for m in np.unique(month):
            k = month == m
            # robust: median absolute miss so a few bad measurements don't dominate
            rows.append(dict(month=int(m), days_mid=float(np.median(tl[k] - Time(FIT_END).jd)), n=int(k.sum()),
                             hard_median_miss=float(np.median(np.hypot(*res_lh[k].T))),
                             easy_median_miss=float(np.median(np.hypot(*res_le[k].T)))))
        out["out_of_sample"] = rows
        oos = (np.array([r["days_mid"] for r in rows]), np.array([r["easy_median_miss"] for r in rows]),
               np.array([r["hard_median_miss"] for r in rows]))

    # ---------------------------------------------------------------- non-convexity
    used = H["used"]
    if not quick:
        print("multi-start (60 random starting orbits) ...")
        ms = analysis.multistart(t0, t, ra, dec, R, sig, used, H["elements"], n=60)
        best = H["rms_arcsec"]
        reached = [abs(m["da"]) < 1e-5 and m["max_angle_diff_deg"] < 1e-3 for m in ms]
        out["multistart"] = dict(n=len(ms), reached=int(sum(reached)), runs=ms)
        print("objective landscape over (a, M0) ...")
        a_grid = np.linspace(H["elements"][0] - 0.35, H["elements"][0] + 0.35, 90)
        M_grid = np.radians(np.linspace(0, 360, 181))
        L = np.empty((a_grid.size, M_grid.size))
        for i_, a_ in enumerate(a_grid):
            for j_, M_ in enumerate(M_grid):
                th = H["elements"].copy()
                th[0], th[5] = a_, M_
                L[i_, j_] = hard.chi2_at(th, t0, t, ra, dec, R, sig, used)
        np.savez(os.path.join(RESULTS, "landscape.npz"), a=a_grid, M=M_grid, chi2=L)
        Mslice = np.radians(np.linspace(0, 360, 721))
        chiM = analysis.objective_slice(H["elements"], 5, Mslice, t0, t, ra, dec, R, sig, used)
        out["slice_M0"] = dict(M_deg=np.degrees(Mslice).tolist(), chi2=chiM.tolist())

    # ---------------------------------------------------------------- price of realism
    print("ablation (switch pieces of physics off one at a time) ...")
    abl = analysis.ablation(H["elements"], t0, sig, t, ra, dec, R, H["used"])
    out["ablation"] = [{k: v for k, v in r.items() if k != "elements"} for r in abl]
    print("arc-length study ...")
    jpl_r_at = lambda tt: analysis.jpl_state_at(h_vec, tt)[0]
    arc = analysis.arc_length_study([3, 5, 7, 10, 15, 20, 30, 45, 61], t0, t, ra, dec, R, sig, jpl_r_at,
                                    np.floor(t.min() - 0.5) + 0.5)
    out["arc_length"] = arc

    out["runtime_seconds"] = time.perf_counter() - tic_all

    # ---------------------------------------------------------------- figures
    print("figures ...")
    curve_t = np.linspace(t.min(), Time(LATER_END).jd, 600)
    rho_c, _ = __import__("src.geometry", fromlist=["x"]).light_time_positions(
        lambda tt: propagate(H["elements"], t0, tt), curve_t, earth_heliocentric(curve_t))
    ra_curve, dec_curve = unit_to_radec(rho_c)
    for th in THEMES:
        plots.sky_track(np.degrees(ra), np.degrees(dec),
                        np.degrees(later.ra.values) if len(later) else None,
                        np.degrees(later.dec.values) if len(later) else None,
                        np.degrees(ra_curve), np.degrees(dec_curve), theme=th)
        plots.obs_per_night(fit.jd_utc.values, theme=th)
        top = fit.stn.value_counts().head(10)
        names = [c if codes.loc[c, "name"] in ("", c) else f"{c}  {codes.loc[c, 'name'][:24]}"
                 for c in top.index]
        plots.station_bars(names, top.values.tolist(), theme=th)
        plots.rbar_scan(E["scan"]["rbar"], E["scan"]["mismatch"], E["candidates"], E["rbar"], theme=th)
        plots.residuals_vs_time(days, H["residuals"], used, theme=th)
        z = (H["residuals"][used] / sig[used, None]).ravel()
        z = z / np.sqrt(H["reduced_chi2"])
        plots.residual_histogram(z, theme=th)
        plots.easy_vs_hard_misses(days, np.hypot(*easy_res.T), np.hypot(*H["residuals"].T), theme=th)
        plots.prediction_curves(days_after, err_e, err_h, oos, theme=th)
        plots.ablation_bars([r["variant"] for r in abl], [r["rms_arcsec"] for r in abl], theme=th)
        plots.arc_length_plot([r["days"] for r in arc], [r["easy_pos_err_km"] for r in arc],
                              [r["hard_pos_err_km"] for r in arc], theme=th)
        # top-down geometry
        ecl = lambda v: (v @ EQ_TO_ECL.T)[:, :2]
        tt = np.linspace(t.min(), t.max(), 120)
        full_t = t0 + np.linspace(0, 2 * np.pi / np.sqrt(2.9591220828559115e-4 / H["elements"][0] ** 3), 400)
        earth_orbit = ecl(earth_heliocentric(np.linspace(t0, t0 + 365.25, 400)))
        los = []
        for tk in np.linspace(t.min(), t.max(), 5):
            e_k = ecl(earth_heliocentric([tk]))[0]
            a_k = ecl(propagate(H["elements"], t0, [tk]))[0]
            los.append((e_k, a_k))
        plots.geometry_view(ecl(earth_heliocentric(tt)), ecl(propagate(H["elements"], t0, tt)),
                            ecl(propagate(H["elements"], t0, full_t)), earth_orbit, los, theme=th)
        if not quick:
            plots.multistart_plot([m["rms_arcsec"] for m in ms], reached, H["rms_arcsec"], theme=th)
            plots.landscape(a_grid, np.degrees(M_grid), np.log10(L), H["elements"], theme=th)
            best_M = np.degrees(H["elements"][5]) % 360
            plots.slice_plot(np.degrees(Mslice), chiM, best_M,
                             r"mean anomaly at epoch $M_0$ [deg]  (other five held at the optimum)",
                             "slice_M0", chord_from_deg=best_M - 110 if best_M > 180 else best_M + 110, theme=th)

    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"done in {time.perf_counter() - tic_all:.0f} s -> results/results.json, figures/")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip multi-start and the landscape scan")
    main(quick=ap.parse_args().quick)
