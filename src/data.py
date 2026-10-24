"""
Reading the raw files: MPC observations, MPC observatory codes, and the JPL
Horizons tables we use for checking.

The observation pipeline is:
    load_mpc_observations()  ->  every Eros observation the MPC has (pandas table)
    clean_observations()     ->  keep ground-based CCD measurements in our window
    attach_geometry()        ->  add observer positions and unit vectors
"""
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from astropy.time import Time

from .constants import DEG
from .geometry import (earth_heliocentric, radec_to_unit, site_geocentric,
                       utc_to_tt_tdb)

RAW = os.environ.get("ORBITFIT_RAW") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


# ----------------------------------------------------------------------------------
# MPC observations
# ----------------------------------------------------------------------------------

def _mpc_reply():
    with open(os.path.join(RAW, "mpc_obs_433.json")) as f:
        reply = json.load(f)
    return reply[0] if isinstance(reply, list) else reply


def parse_obs80(text):
    """
    Parse the classic MPC 80-column format. Column positions (1-indexed):
      15 note2 (C = CCD, S = satellite, R = radar, ...), 16-32 date (UTC),
      33-44 RA (h m s), 45-56 Dec (d m s), 66-70 mag, 71 band, 72 catalogue,
      78-80 observatory code.
    """
    rows = []
    for line in text.splitlines():
        if len(line) < 80:
            continue
        try:
            y, mo, d = int(line[15:19]), int(line[20:22]), float(line[23:32])
            h, m, s = (float(x) for x in line[32:44].split())
            sign = -1.0 if line[44] == "-" else 1.0
            dd, dm, ds = (float(x) for x in line[45:56].split())
        except ValueError:
            continue                      # old formats / malformed lines; not in our window anyway
        rows.append(dict(
            note2=line[14], year=y, month=mo, day=d,
            ra_deg=15.0 * (h + m / 60 + s / 3600),
            dec_deg=sign * (dd + dm / 60 + ds / 3600),
            mag=line[65:70].strip(), band=line[70].strip(), cat=line[71].strip(),
            stn=line[77:80].strip(), rmsRA=np.nan, rmsDec=np.nan, raw=line))
    df = pd.DataFrame(rows)
    # calendar date with fractional day -> Julian date (UTC)
    base = Time({"year": df.year.values, "month": df.month.values, "day": np.ones(len(df), int)},
                format="ymdhms", scale="utc").jd
    df["jd_utc"] = base + (df.day.values - 1.0)
    df["mode"] = np.where(df.note2 == "C", "CCD", df.note2)
    return df.drop(columns=["year", "month", "day"])


def parse_ades_xml(text):
    """ADES XML (the modern format). Gives decimal-degree RA/Dec and, often, uncertainties."""
    root = ET.fromstring(text)
    rows = []
    for o in root.iter("optical"):
        g = {c.tag: (c.text or "").strip() for c in o}
        try:
            t = g["obsTime"].rstrip("Z")
            rows.append(dict(obsTime=t, ra_deg=float(g["ra"]), dec_deg=float(g["dec"]),
                             stn=g.get("stn", ""), mode=g.get("mode", ""),
                             rmsRA=float(g["rmsRA"]) if g.get("rmsRA") else np.nan,
                             rmsDec=float(g["rmsDec"]) if g.get("rmsDec") else np.nan,
                             cat=g.get("astCat", ""), mag=g.get("mag", ""), band=g.get("band", "")))
        except (KeyError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if len(df):
        df["jd_utc"] = Time(list(df.obsTime), format="isot", scale="utc").jd
    return df


def load_mpc_observations(prefer="obs80"):
    """
    All MPC observations of Eros. OBS80 is the default because every observation
    exists in it; if the reply also has ADES XML we borrow the uncertainties
    (rmsRA/rmsDec) from it, matching on station + time.
    """
    reply = _mpc_reply()
    obs80 = parse_obs80(reply.get("OBS80", ""))
    xml = reply.get("XML") or ""
    if xml.strip():
        try:
            ades = parse_ades_xml(xml)
        except ET.ParseError:
            ades = pd.DataFrame()
        if len(ades) and prefer == "ades":
            return ades
        if len(ades):
            # match each OBS80 row to an ADES row at the same station within ~1 s
            key80 = obs80.stn + ":" + (np.round(obs80.jd_utc * 86400)).astype("int64").astype(str)
            keyad = ades.stn + ":" + (np.round(ades.jd_utc * 86400)).astype("int64").astype(str)
            lut = dict(zip(keyad, zip(ades.rmsRA, ades.rmsDec)))
            got = [lut.get(k, (np.nan, np.nan)) for k in key80]
            obs80["rmsRA"] = [g[0] for g in got]
            obs80["rmsDec"] = [g[1] for g in got]
    return obs80


# ----------------------------------------------------------------------------------
# Observatory codes
# ----------------------------------------------------------------------------------

def load_obscodes():
    """MPC observatory codes -> DataFrame indexed by code, with lon, rho_cos, rho_sin."""
    html_path = os.path.join(RAW, "ObsCodes.html")
    rows = []
    if os.path.exists(html_path):
        text = open(html_path, encoding="utf-8", errors="replace").read()
        text = re.sub(r"<[^>]+>", "", text)
        for line in text.splitlines():
            if len(line) < 30 or not re.match(r"^[0-9A-Z]{3}\s", line):
                continue
            try:
                rows.append(dict(code=line[0:3], lon=float(line[4:13]), rho_cos=float(line[13:21]),
                                 rho_sin=float(line[21:30]), name=line[30:].strip()))
            except ValueError:
                pass                       # space telescopes etc. have no parallax constants
    else:
        alt = [p for p in ("mpc_obscodes.json", "station_coords_horizons.json")
               if os.path.exists(os.path.join(RAW, p))]
        with open(os.path.join(RAW, alt[0])) as f:
            j = json.load(f)
        items = j.items() if isinstance(j, dict) else [(x.get("obscode"), x) for x in j]
        for code, v in items:
            try:
                rows.append(dict(code=code, lon=float(v["longitude"]), rho_cos=float(v["rhocosphi"]),
                                 rho_sin=float(v["rhosinphi"]), name=v.get("name", "")))
            except (KeyError, TypeError, ValueError):
                pass
    return pd.DataFrame(rows).drop_duplicates("code").set_index("code")


# ----------------------------------------------------------------------------------
# Cleaning + geometry
# ----------------------------------------------------------------------------------

def clean_observations(obs, start_utc, end_utc, codes):
    """
    Keep what a two-body model can honestly be fitted to:
      * inside our time window,
      * CCD measurements (note2 == 'C') -- the modern, accurate kind,
      * from a ground station whose position on Earth we know.
    Satellite (S), radar (R), roving (V) and deleted (X) lines are dropped.
    """
    t1, t2 = Time(start_utc, scale="utc").jd, Time(end_utc, scale="utc").jd
    keep = (obs.jd_utc >= t1) & (obs.jd_utc < t2)
    keep &= obs["mode"].isin(["CCD", "C"])
    keep &= obs.stn.isin(codes.index)
    out = obs[keep].sort_values("jd_utc").reset_index(drop=True)
    return out


def attach_geometry(obs, codes):
    """Add TT/TDB times, observer heliocentric positions R and observed unit vectors u."""
    obs = obs.copy()
    obs["jd_tt"], obs["jd_tdb"] = utc_to_tt_tdb(obs.jd_utc.values)
    c = codes.loc[obs.stn]
    site = site_geocentric(obs.jd_utc.values, obs.jd_tt.values,
                           c.lon.values, c.rho_cos.values, c.rho_sin.values)
    earth = earth_heliocentric(obs.jd_tdb.values)
    R = earth + site
    for k, ax in enumerate("xyz"):
        obs[f"R{ax}"] = R[:, k]
        obs[f"site_{ax}"] = site[:, k]
    obs["ra"] = obs.ra_deg * DEG
    obs["dec"] = obs.dec_deg * DEG
    return obs


def arrays(obs):
    """Pull the numpy arrays the solvers need out of a cleaned table."""
    R = obs[["Rx", "Ry", "Rz"]].values
    u = radec_to_unit(obs.ra.values, obs.dec.values)
    return obs.jd_tdb.values, obs.ra.values, obs.dec.values, u, R


# ----------------------------------------------------------------------------------
# JPL Horizons tables (validation only)
# ----------------------------------------------------------------------------------

def _horizons_table(name):
    text = open(os.path.join(RAW, name), encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    i0 = lines.index("$$SOE")
    i1 = lines.index("$$EOE")
    header = None
    for l in reversed(lines[:i0]):
        if "," in l and not l.startswith("*"):
            header = [h.strip() for h in l.split(",")]
            break
    body = [[x.strip() for x in l.split(",")] for l in lines[i0 + 1:i1]]
    width = max(len(r) for r in body)
    header = (header + [f"c{k}" for k in range(len(header), width)])[:width]
    return pd.DataFrame(body, columns=header)


def load_horizons_elements():
    df = _horizons_table("horizons_eros_elements.txt")
    num = ["JDTDB", "EC", "QR", "IN", "OM", "W", "Tp", "N", "MA", "TA", "A", "AD", "PR"]
    return df[num].astype(float)


def load_horizons_vectors(name="horizons_eros_vectors.txt"):
    df = _horizons_table(name)
    return df[["JDTDB", "X", "Y", "Z", "VX", "VY", "VZ"]].astype(float)


def load_horizons_radec():
    df = _horizons_table("horizons_eros_radec_geocentric.txt")
    col = lambda key: next(c for c in df.columns if key in c)
    out = pd.DataFrame({
        "jd_utc": df[col("JDUT")].astype(float),
        "ra_deg": df[col("R.A.")].astype(float),
        "dec_deg": df[col("DEC")].astype(float),
        "delta": df[col("delta")].astype(float),
    })
    return out
