#!/usr/bin/env python3
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
"""
- This file downloads the raw telescopic data for the Eros orbit-fitting project.
- Everything is stored in ./data/raw/ 
- No need to pip install anything
- Run it using: python3 fetch_data.py

What we download and why:

  MPC_Observations_433Eros.json     Every published telescopic observation of 433 Eros, from
                                    the Minor Planet Center (MPC) get-obs dataset.
                                    API: https://data.minorplanetcenter.net/api/get-obs

  ObservatoryCodes.html             MPC list of observatory codes with their parallax constants
                                    This tells us the position of each telescope used for observation
                                    in this dataset
                                    API: https://www.minorplanetcenter.net/iau/lists/ObsCodes.html

  JPL_*.txt                         Tables from JPL Horizons Dataset (NASA), used as the benchmark dataset
                                    for verification
                                    API: https://ssd.jpl.nasa.gov/horizons/
                        

  JPL_SBdb_433Eros.json             JPL Small-Body Database record 
                                    API: https://ssd-api.jpl.nasa.gov/sbdb.api
"""

def _ssl_context():
    # python.org installers on macOS ship without CA certificates; certifi fixes that
    # if it happens to be installed. Otherwise fall back to the system default.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
    
CTX = _ssl_context()

here = os.path.dirname(os.path.abspath(__file__)) # Current Directory
raw = os.path.join(here, "data", "raw")           # Storage Directory


# JPL_Horizons has predicted data beyond 2026 aswell, and obviously has the current accurate measurements
# Thus, we can compare the current tables with our optimized data and predicted data of JPL with 
# predicted data using our optimization technique
H_START, H_STOP = "2024-01-01", "2028-01-01"

def fetch(url, json_body=None, timeout=300, tries=3):
    """GET a URL (optionally with a JSON body, which the MPC API wants) and return bytes."""
    data = None
    headers = {"User-Agent": "orbitfit-student-project (IIIT Bangalore, AIT 203)"}
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="GET")
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read()
        except urllib.error.URLError as e:
            last = e
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit(
                    "\nSSL certificate error. On a Mac with python.org Python, run\n"
                    "  /Applications/Python\\ 3.*/Install\\ Certificates.command\n"
                    "(or: pip3 install certifi) and then run this script again.\n")
            print(f"    attempt {k + 1} failed: {e}; retrying...")
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def save(name, blob):
    path = os.path.join(raw, name)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"    saved {name}  ({len(blob) / 1e6:.2f} MB)")
    return path


def horizons(name, **params):
    # Horizons wants string values wrapped in single quotes.
    q = {"format": "text", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "CSV_FORMAT": "YES"}
    q.update({k: f"'{v}'" for k, v in params.items()})
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(
        q, quote_via=urllib.parse.quote)
    blob = fetch(url)
    text = blob.decode("utf-8", "replace")
    if "$$SOE" not in text:
        print("    !! Horizons did not return a table. First lines of the reply:")
        print("       " + "\n       ".join(text.splitlines()[:15]))
    save(name, blob)


def stations_in_use(obs_json_path, years=("2025", "2026")):
    """Observatory codes that appear in the OBS80 lines from the years we care about."""
    reply = json.load(open(obs_json_path))
    obs80 = reply[0].get("OBS80", "") if isinstance(reply, list) else reply.get("OBS80", "")
    codes = {l[77:80].strip() for l in obs80.splitlines() if len(l) >= 80 and l[15:19] in years}
    return sorted(c for c in codes if c)


def station_coords_from_horizons(codes):
    """
    Back-up plan for the observatory list: Horizons prints each site's cylindrical
    coordinates (east longitude, distance from Earth's axis, height above the
    equator) in the header of any ephemeris requested FROM that site.
    """
    out = {}
    for c in codes:
        q = {"format": "text", "COMMAND": "'10'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES",
             "EPHEM_TYPE": "'OBSERVER'", "CENTER": f"'{c}@399'", "TLIST": "'2460990.5'",
             "QUANTITIES": "'1'"}
        url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
        try:
            text = fetch(url, timeout=60).decode("utf-8", "replace")
        except Exception as e:
            print(f"    {c}: failed ({e})")
            continue
        for line in text.splitlines():
            if line.startswith("Center cylindric"):
                lon, dxy, dz = (float(x) for x in line.split(":")[1].split("{")[0].split(","))
                out[c] = {"longitude": lon, "rhocosphi": dxy / 6378.137, "rhosinphi": dz / 6378.137,
                          "name": c, "source": "JPL Horizons site header"}
        time.sleep(0.3)
    return out


def main():
    os.makedirs(raw, exist_ok=True)
    ok, failed = [], []

    def step(name, fn):
        try:
            fn()
            ok.append(name)
        except Exception as e:
            print(f"    !! {name} failed: {e}")
            failed.append(name)

    # 1. MPC observations -------------------------------------------------------
    def mpc_obs():
        url = "https://data.minorplanetcenter.net/api/get-obs"
        try:
            blob = fetch(url, {"desigs": ["433"], "output_format": ["OBS80", "XML"]})
        except Exception as e:
            print(f"    OBS80+XML request failed ({e}); trying OBS80 only")
            blob = fetch(url, {"desigs": ["433"], "output_format": ["OBS80"]})
        save("mpc_obs_433.json", blob)
        reply = json.loads(blob)
        obs80 = reply[0].get("OBS80", "") if isinstance(reply, list) else ""
        print(f"    {sum(len(l) >= 80 for l in obs80.splitlines())} observation lines in OBS80 format")

    print("[1/4] MPC observations of 433 Eros (this one is big, give it a minute)")
    step("MPC observations", mpc_obs)

    # 2. Observatory codes ------------------------------------------------------
    def obscodes():
        try:
            save("ObsCodes.html", fetch("https://www.minorplanetcenter.net/iau/lists/ObsCodes.html", timeout=90))
            return
        except Exception as e:
            print(f"    ObsCodes.html failed ({e}); asking JPL Horizons for each telescope instead")
        codes = stations_in_use(os.path.join(raw, "mpc_obs_433.json"))
        coords = station_coords_from_horizons(codes)
        save("station_coords_horizons.json", json.dumps(coords, indent=1).encode())
        print(f"    got coordinates for {len(coords)} of {len(codes)} telescopes")

    print("[2/4] Observatory codes")
    step("observatory codes", obscodes)

    # 3. JPL Horizons (validation only) -----------------------------------------
    print("[3/4] JPL Horizons tables (only used to CHECK our fit)")
    span = dict(START_TIME=H_START, STOP_TIME=H_STOP, STEP_SIZE="1 d")
    step("Horizons elements", lambda: horizons(
        "horizons_eros_elements.txt", COMMAND="433;", EPHEM_TYPE="ELEMENTS",
        CENTER="500@10", REF_PLANE="ECLIPTIC", REF_SYSTEM="ICRF",
        OUT_UNITS="AU-D", TP_TYPE="ABSOLUTE", **span))
    step("Horizons Eros vectors", lambda: horizons(
        "horizons_eros_vectors.txt", COMMAND="433;", EPHEM_TYPE="VECTORS",
        CENTER="500@10", REF_PLANE="FRAME", REF_SYSTEM="ICRF",
        OUT_UNITS="AU-D", VEC_TABLE="2", VEC_CORR="NONE", **span))
    step("Horizons RA/Dec", lambda: horizons(
        "horizons_eros_radec_geocentric.txt", COMMAND="433;", EPHEM_TYPE="OBSERVER",
        CENTER="500@399", QUANTITIES="1,20", ANG_FORMAT="DEG", EXTRA_PREC="YES",
        CAL_FORMAT="BOTH", REF_SYSTEM="ICRF", **span))
    step("Horizons Earth vectors", lambda: horizons(
        "horizons_earth_vectors.txt", COMMAND="399", EPHEM_TYPE="VECTORS",
        CENTER="500@10", REF_PLANE="FRAME", REF_SYSTEM="ICRF",
        OUT_UNITS="AU-D", VEC_TABLE="2", VEC_CORR="NONE", **span))

    # 4. SBDB record --------------------------------------------------------------
    print("[4/4] JPL Small-Body Database record")
    step("SBDB record", lambda: save("sbdb_433.json", fetch(
        "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=433&full-prec=true&cov=mat&phys-par=true")))

    print("\nsucceeded:", ", ".join(ok) or "-")
    if failed:
        print("FAILED:   ", ", ".join(failed), "\n(the zip is still written -- send it anyway, plus this output)")

    # manifest + zip ------------------------------------------------------------------
    manifest = {}
    for name in sorted(os.listdir(raw)):
        p = os.path.join(raw, name)
        if os.path.isfile(p) and not name.startswith("."):
            manifest[name] = {"bytes": os.path.getsize(p),
                              "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    manifest["_downloaded_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(os.path.join(raw, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    zpath = os.path.join(here, "orbitfit_raw_data.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(raw)):
            z.write(os.path.join(raw, name), arcname=os.path.join("data", "raw", name))
    print(f"\nDone. raw files are in data/raw/ and zipped into {os.path.basename(zpath)} "
          f"({os.path.getsize(zpath) / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
