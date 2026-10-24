"""
All the figures. Each function takes the numbers it needs and a theme:

    theme="slide"  transparent background, chalk-coloured ink -- meant to sit
                   directly on the charcoal slides of our deck
    theme="paper"  white background, dark ink -- for the written guide/report

Colours have fixed jobs and never swap between figures:
    hard fit  -> blue      easy fit -> orange      JPL -> aqua
    observations -> chalk/neutral ink (they are the measurements, not a model)
(The three series colours were checked for colour-blind separation on both
the charcoal slide and on white.)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

THEMES = {
    "slide": dict(bg="none", ink="#ECE8DC", ink2="#BDB9AE", muted="#8F8B82", grid="#4B4F51",
                  hard="#3987E5", easy="#D95926", jpl="#199E70", obs="#ECE8DC", surface="#35393A"),
    "paper": dict(bg="#FFFFFF", ink="#1B1B1A", ink2="#52514E", muted="#8A8984", grid="#E4E3DF",
                  hard="#2A78D6", easy="#EB6834", jpl="#1BAF7A", obs="#3D3C39", surface="#FFFFFF"),
}


def _style(theme):
    T = THEMES[theme]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.facecolor": T["bg"], "figure.facecolor": T["bg"], "savefig.facecolor": T["bg"],
        "axes.edgecolor": T["grid"], "axes.labelcolor": T["ink2"], "axes.titlecolor": T["ink"],
        "xtick.color": T["ink2"], "ytick.color": T["ink2"], "text.color": T["ink"],
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.labelcolor": T["ink2"],
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })
    return T


def _save(fig, name, theme):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}_{theme}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12,
                transparent=(theme == "slide"))
    plt.close(fig)
    return path


def _dot(ax, x, y, color, T, size=26, alpha=1.0, **kw):
    """Scatter with the surface-coloured ring the style guide asks for."""
    ring = T["surface"] if T["surface"] != "none" else "white"
    return ax.scatter(x, y, s=size, color=color, edgecolors=ring, linewidths=0.6, alpha=alpha, **kw)


# --------------------------------------------------------------------------------------
def sky_track(ra_fit, dec_fit, ra_later, dec_later, ra_curve, dec_curve, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(ra_curve, dec_curve, color=T["hard"], lw=2, zorder=2, label="hard fit (predicted track)")
    _dot(ax, ra_fit, dec_fit, T["obs"], T, size=14, zorder=3, label="observations used in the fit")
    if ra_later is not None and len(ra_later):
        _dot(ax, ra_later, dec_later, T["muted"], T, size=10, alpha=0.8, zorder=3,
             label="later observations (never shown to the fit)")
    ax.invert_xaxis()                      # sky convention: RA grows to the left
    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Declination [deg]")
    ax.legend(loc="best", fontsize=11, markerscale=1.6)
    return _save(fig, "sky_track", theme)


def obs_per_night(jd_utc, theme="slide"):
    T = _style(theme)
    nights = np.floor(jd_utc - 0.5)
    vals, counts = np.unique(nights, return_counts=True)
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.bar(vals - vals.min(), counts, width=0.8, color=T["obs"], alpha=0.85)
    ax.set_xlabel("night of the fit window (day number)")
    ax.set_ylabel("observations")
    ax.grid(axis="x", visible=False)
    return _save(fig, "obs_per_night", theme)


def station_bars(names, counts, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 0.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, counts, height=0.62, color=T["obs"], alpha=0.85)
    ax.set_yticks(y, names, fontsize=11)
    for yy, c in zip(y, counts):
        ax.text(c, yy, f"  {c}", va="center", fontsize=10.5, color=T["ink2"])
    ax.set_xlabel("observations in the fit window")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(counts) * 1.18)
    return _save(fig, "stations", theme)


def geometry_view(earth_xy, eros_xy_arc, eros_orbit_xy, earth_orbit_xy, los_pairs, theme="slide"):
    """Top-down view of the ecliptic: Sun, both orbits, positions during the arc, a few lines of sight."""
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot(*earth_orbit_xy.T, color=T["ink2"], lw=1.2, alpha=0.7)
    ax.plot(*eros_orbit_xy.T, color=T["hard"], lw=1.6, alpha=0.9)
    for (e, a) in los_pairs:
        ax.plot([e[0], a[0]], [e[1], a[1]], color=T["muted"], lw=1.0, alpha=0.8)
    ax.plot(*earth_xy.T, color=T["ink"], lw=3.0)
    ax.plot(*eros_xy_arc.T, color=T["hard"], lw=3.4)
    ax.scatter([0], [0], s=260, marker="*", color="#F2C14E", zorder=5)
    ax.text(0.07, -0.12, "Sun", fontsize=12, color=T["ink2"])
    em, am = earth_xy[len(earth_xy) // 2], eros_xy_arc[len(eros_xy_arc) // 2]
    ax.text(*(em * 0.80), "Earth", fontsize=12, color=T["ink"], ha="center", va="center")
    ax.text(*(am * 1.13), "Eros", fontsize=12, color=T["hard"], ha="center", va="center", fontweight="bold")
    ax.text(*(-em * 0.80), "Earth's orbit", fontsize=10, color=T["ink2"], ha="center", va="center")
    ax.text(*(-am * 1.10), "Eros's fitted orbit", fontsize=10, color=T["hard"], ha="center", va="center")
    ax.set_aspect("equal")
    lim = 1.15 * np.max(np.abs(eros_orbit_xy))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return _save(fig, "geometry_view", theme)


def rbar_scan(grid, mismatch, cands, chosen, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    m = (grid >= 0.5) & (grid <= 3.0)
    ax.axhline(0, color=T["ink2"], lw=1.0)
    ax.plot(grid[m], mismatch[m], color=T["easy"], lw=2)
    for c in cands:
        if c["rbar"] < 0.5:
            continue
        is_chosen = abs(c["rbar"] - chosen) < 1e-9
        _dot(ax, [c["rbar"]], [0], T["hard"] if is_chosen else T["ink"], T, size=110 if is_chosen else 60, zorder=4)
        if is_chosen:
            lab = f"kept: {c['rbar']:.2f} au\n{c['geocentric_au']:.2f} au from Earth"
        elif c["geocentric_au"] < 0.1:
            lab = f"observer root\n{c['geocentric_au']:.2f} au from Earth"
        else:
            lab = "spurious root"
        if is_chosen:
            off, ha = (0, -46), "center"
        elif c["geocentric_au"] < 0.1:
            off, ha = (18, 38), "left"
        else:
            off, ha = (-14, -44), "right"
        ax.annotate(lab, (c["rbar"], 0), xytext=off, textcoords="offset points",
                    ha=ha, fontsize=10.5, color=T["ink"] if is_chosen else T["ink2"],
                    arrowprops=dict(arrowstyle="-", color=T["muted"], lw=0.8))
    ax.set_xlabel(r"assumed Sun–Eros distance  $\bar r$  [au]")
    ax.set_ylabel(r"$|r_0| - \bar r$  [au]")
    span = np.max(np.abs(mismatch[m & (grid > 0.65)]))
    ax.set_ylim(-min(span, 1.6) * 1.1, min(span, 1.6) * 1.1)
    ax.set_xlim(0.5, 3.0)
    return _save(fig, "rbar_scan", theme)


def residuals_vs_time(days, res, used, theme="slide"):
    T = _style(theme)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=True)
    for k, (ax, lab) in enumerate(zip(axes, [r"$\Delta\alpha\cos\delta$  [″]", r"$\Delta\delta$  [″]"])):
        ax.axhline(0, color=T["ink2"], lw=1.0)
        _dot(ax, days[used], res[used, k], T["hard"], T, size=10, alpha=0.75)
        if (~used).any():
            ax.scatter(days[~used], np.clip(res[~used, k], -6, 6), s=22, marker="x",
                       color=T["easy"], linewidths=1.2, label="rejected (> 3σ)")
        ax.set_ylabel(lab)
        ax.set_ylim(-6, 6)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[1].set_xlabel("days since start of the fit window")
    return _save(fig, "residuals_time", theme)


def residual_histogram(z, theme="slide"):
    """Normalised residuals (residual / sigma) vs a standard normal curve."""
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.linspace(-4, 4, 41)
    ax.hist(z, bins=bins, density=True, color=T["hard"], alpha=0.85, rwidth=0.9)
    x = np.linspace(-4, 4, 400)
    ax.plot(x, np.exp(-x * x / 2) / np.sqrt(2 * np.pi), color=T["ink"], lw=2, label="standard normal")
    ax.set_xlabel("residual / σ  (both RA and Dec)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=11)
    return _save(fig, "residual_hist", theme)


def easy_vs_hard_misses(days, mag_easy, mag_hard, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _dot(ax, days, mag_easy, T["easy"], T, size=10, alpha=0.7, label="easy solution")
    _dot(ax, days, mag_hard, T["hard"], T, size=10, alpha=0.7, label="hard solution")
    ax.set_yscale("log")
    ax.set_xlabel("days since start of the fit window")
    ax.set_ylabel("angular miss per observation  [″]")
    ax.legend(loc="lower right", fontsize=11, markerscale=2)
    return _save(fig, "easy_vs_hard_misses", theme)


def prediction_curves(days_after, err_easy, err_hard, oos=None, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(days_after, err_easy, color=T["easy"], lw=2, label="easy orbit vs JPL")
    ax.plot(days_after, err_hard, color=T["hard"], lw=2, label="hard orbit vs JPL")
    if oos is not None:
        d, rms_e, rms_h = oos
        _dot(ax, d, rms_h, T["hard"], T, size=46, zorder=4, label="hard orbit vs later REAL observations")
        _dot(ax, d, rms_e, T["easy"], T, size=46, zorder=4, label="easy orbit vs later REAL observations")
    ax.set_yscale("log")
    ax.set_xlabel("days after the last observation we fitted")
    ax.set_ylabel("prediction error on the sky  [″]")
    for yv, lab in [(60, "1 arcmin"), (3600, "1 degree")]:
        ax.axhline(yv, color=T["grid"], lw=1.0)
        ax.text(days_after.max(), yv * 1.12, lab, ha="right", fontsize=10, color=T["muted"])
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.30), fontsize=10.5)
    return _save(fig, "prediction_error", theme)


def multistart_plot(final_rms, reached, best_rms, theme="slide"):
    T = _style(theme)
    order = np.argsort(final_rms)
    v = np.asarray(final_rms)[order]
    ok = np.asarray(reached)[order]
    x = np.arange(len(v))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    _dot(ax, x[~ok], np.clip(v[~ok], 1e-1, 1e7), T["muted"], T, size=34, label="stuck in a wrong valley / diverged")
    _dot(ax, x[ok], v[ok], T["hard"], T, size=40, label="reached the best orbit")
    ax.axhline(best_rms, color=T["hard"], lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("random starting orbits, sorted by where they ended")
    ax.set_ylabel("final RMS miss  [″]")
    ax.legend(loc="upper left", fontsize=11)
    return _save(fig, "multistart", theme)


def landscape(a_grid, M_grid_deg, log_chi2, best, theme="slide"):
    """log10 chi^2 over (a, M0) with the other four elements held at the optimum."""
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.grid(False)
    cmap = "Blues_r" if theme == "slide" else "Blues"
    im = ax.pcolormesh(M_grid_deg, a_grid, log_chi2, cmap=cmap, shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(r"log$_{10}\,\chi^2$", color=T["ink2"])
    cb.ax.tick_params(colors=T["ink2"])
    cb.outline.set_edgecolor(T["grid"])
    ax.scatter([np.degrees(best[5]) % 360], [best[0]], s=120, marker="*", color=T["easy"],
               edgecolors=T["ink"], linewidths=0.8, zorder=5)
    ax.set_xlabel(r"mean anomaly at epoch  $M_0$  [deg]")
    ax.set_ylabel(r"semi-major axis  $a$  [au]")
    return _save(fig, "landscape", theme)


def slice_plot(values_deg, chi2, best_deg, label, name, chord_from_deg=None, theme="slide"):
    """
    chi^2 along one element. Drawn on a LINEAR axis with one chord: if the curve
    rises above a chord anywhere, the function is not convex (that is the definition).
    """
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.asarray(chi2) / np.max(chi2)
    ax.plot(values_deg, y, color=T["hard"], lw=2, zorder=3)
    if chord_from_deg is not None:
        # put the chord across the hump (the biggest value): a convex curve can't bulge above it
        xp = values_deg[int(np.argmax(y))]
        x1 = max(values_deg.min(), xp - 55)
        x2 = min(values_deg.max(), xp + 95)
        y1 = np.interp(x1, values_deg, y)
        y2 = np.interp(x2, values_deg, y)
        ax.plot([x1, x2], [y1, y2], color=T["easy"], lw=2, zorder=4)
        _dot(ax, [x1, x2], [y1, y2], T["easy"], T, size=60, zorder=5)
        lo, hi = sorted([x1, x2])
        xs = np.linspace(lo, hi, 200)
        ys = np.interp(xs, values_deg, y)
        ch = y1 + (y2 - y1) * (xs - x1) / (x2 - x1)
        ax.fill_between(xs, ch, ys, where=ys > ch, color=T["easy"], alpha=0.18, lw=0)
    ax.set_xlabel(label)
    ax.set_ylabel(r"$\chi^2$ / max")
    ax.set_xlim(values_deg.min(), values_deg.max())
    ax.set_ylim(0, 1.08)
    return _save(fig, name, theme)


def ablation_bars(names, rms, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(names) + 0.8))
    y = np.arange(len(names))[::-1]
    colors = [T["hard"] if k == 0 else T["obs"] for k in range(len(names))]
    ax.barh(y, rms, height=0.6, color=colors, alpha=0.9)
    ax.set_yticks(y, names, fontsize=11.5)
    for yy, v in zip(y, rms):
        ax.text(v, yy, f"  {v:.2f}″", va="center", fontsize=11, color=T["ink2"])
    ax.set_xlabel("RMS miss of the kept observations  [″]")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(rms) * 1.25)
    return _save(fig, "ablation", theme)


def arc_length_plot(days, err_easy, err_hard, theme="slide"):
    T = _style(theme)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(days, err_easy, color=T["easy"], lw=2, marker="o", ms=6, label="easy")
    ax.plot(days, err_hard, color=T["hard"], lw=2, marker="o", ms=6, label="hard")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(days, [f"{d:g}" for d in days])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.minorticks_off()
    ax.set_xlabel("length of the observed arc  [days]")
    ax.set_ylabel("position error vs JPL at epoch  [km]")
    ax.legend(loc="lower left", fontsize=11)
    return _save(fig, "arc_length", theme)
