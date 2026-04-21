"""
noah_philsa_gfd_consensus.py
============================
Extended NOAH vs PhilSA + GFD comparison.
Adds two improvements over noah_philsa_consensus.py:

1. Combined flood observations
   • PhilSA SAR/optical  (2022–2026, data/philsa_satellite_flood/)
   • GFD / Cloud to Street MODIS  (2000–2018, data/gfd/)
   Together: ~20 yr of flood evidence per city.

2. GHS-SMOD urbanisation stratification
   • Classifies each grid cell as Rural / Peri-urban / Urban
   • Cross-tabs consensus category × urbanisation → tests whether
     "modelled only" cells concentrate in urban areas (drainage effect)
   • Downloaded from JRC GHSL if not already present.

Figures
-------
  output/noah_philsa_gfd_01_maps.png        5-city × 5-panel maps
  output/noah_philsa_gfd_02_timeline.png    event timeline (PhilSA + GFD)
  output/noah_philsa_gfd_03_urban.png       urbanisation stratification
"""

import glob
import io
import os
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
import osmnx as ox

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "output", "noah_validation", "philsa")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
GFD_DIR    = os.path.join(ROOT, "data", "gfd")
NOAH_BASE  = os.path.join(ROOT, "data", "noah", "5yr")
GHSL_DIR   = os.path.join(ROOT, "data", "ghsl")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GFD_DIR,  exist_ok=True)
os.makedirs(GHSL_DIR, exist_ok=True)

UTM          = 32651
GRID_M       = 250
SAT_PRIORITY = ["s1", "s2", "rcm", "alos2", "k5", "gf3", "iceye", "nv1", "l9", "gfd"]
NOAH_ACTIVE  = 2       # Var ≥ 2 → "NOAH active"

CITIES = [
    {"name": "Tuguegarao",    "slug": "tuguegarao",    "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan",             "region": "Cagayan Valley"},
    {"name": "Dagupan",       "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan",          "region": "Ilocos"},
    {"name": "Manila",        "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Cagayan de Oro","slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental",    "region": "Mindanao"},
    {"name": "Cotabato",      "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao",         "region": "BARMM"},
]

# Colours
NOAH_COLORS  = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR  = "#2CBBB4"
DIFF_COLORS  = {
    "noah_much_higher": "#54278F", "noah_higher": "#9467BD",
    "match": "#2E7D32",
    "philsa_higher": "#FDAE61",    "philsa_much_higher": "#B2182B",
}
CONSENSUS_COLORS = {
    "confirmed": "#B71C1C", "modelled": "#1565C0",
    "empirical": "#E65100", "low": "#E8E8E8", "water": WATER_COLOR,
}
URBAN_COLORS = {
    "rural":      "#A5D6A7",
    "peri-urban": "#FFF176",
    "urban":      "#EF9A9A",
    "water":      WATER_COLOR,
    "unknown":    "#EEEEEE",
}
SOURCE_COLORS = {"philsa": "#1976D2", "gfd": "#E65100"}
CITY_COLORS   = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]


# ===========================================================================
# Helpers
# ===========================================================================

def _sat_rank(sat):
    try:
        return SAT_PRIORITY.index(sat.lower())
    except ValueError:
        return len(SAT_PRIORITY)


# ---------- PhilSA loader ---------------------------------------------------
def _load_philsa():
    PAT = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    inventory = {}
    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str, _, sensor, _ = m.groups()
        inventory.setdefault(date_str, {}).setdefault(sensor, []).append(
            os.path.join(PHILSA_DIR, fname)
        )
    parts = []
    for date_str, sensor_map in sorted(inventory.items()):
        event_date    = pd.to_datetime(date_str, format="%Y%m%d")
        chosen_sensor = min(sensor_map.keys(), key=_sat_rank)
        event_gdfs    = []
        for fpath in sensor_map[chosen_sensor]:
            try:
                with zipfile.ZipFile(fpath) as zf:
                    shps = [n for n in zf.namelist() if n.lower().endswith(".shp")]
                    if not shps:
                        continue
                    gdf = gpd.read_file(f"zip://{fpath}!{shps[0]}")
                if gdf is None or len(gdf) == 0:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
                event_gdfs.append(gdf[["geometry"]])
            except Exception as e:
                print(f"    WARN {os.path.basename(fpath)}: {e}")
        if not event_gdfs:
            continue
        event = pd.concat(event_gdfs, ignore_index=True)
        event = gpd.GeoDataFrame(event, geometry="geometry", crs=4326)
        event["event_date"] = event_date
        event["sensor"]     = chosen_sensor
        event["source"]     = "philsa"
        event["event_key"]  = f"{date_str}_{chosen_sensor}"
        parts.append(event)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "event_date", "sensor",
                                         "source", "event_key"], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            geometry="geometry", crs=4326)


# ---------- GFD loader ------------------------------------------------------
def _load_gfd():
    PAT = re.compile(r"^(\d{8})_(.+)_fld_gfd_shp\.zip$")
    parts = []
    if not os.path.isdir(GFD_DIR):
        return gpd.GeoDataFrame(columns=["geometry", "event_date", "sensor",
                                         "source", "event_key"], crs=4326)
    for fname in sorted(os.listdir(GFD_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str, evt_id = m.groups()
        fpath = os.path.join(GFD_DIR, fname)
        try:
            with zipfile.ZipFile(fpath) as zf:
                shps = [n for n in zf.namelist() if n.lower().endswith(".shp")]
                if not shps:
                    continue
                gdf = gpd.read_file(f"zip://{fpath}!{shps[0]}")
            if gdf is None or len(gdf) == 0:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
            event_date = pd.to_datetime(date_str, format="%Y%m%d")
            g = gdf[["geometry"]].copy()
            g["event_date"] = event_date
            g["sensor"]     = "gfd"
            g["source"]     = "gfd"
            g["event_key"]  = f"{date_str}_gfd_{evt_id}"
            parts.append(g)
        except Exception as e:
            print(f"    WARN GFD {fname}: {e}")
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "event_date", "sensor",
                                         "source", "event_key"], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            geometry="geometry", crs=4326)


# ---------- Combined loader -------------------------------------------------
def load_all_observations():
    philsa = _load_philsa()
    gfd    = _load_gfd()
    all_obs = pd.concat([philsa, gfd], ignore_index=True)
    return gpd.GeoDataFrame(all_obs, geometry="geometry", crs=4326)


# ---------- NOAH / grid helpers (same as before) ----------------------------
def _find_noah_shp(province):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(NOAH_BASE, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


def _build_grid(city):
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    r  = city["radius_m"]
    xs = np.arange(centre.x - r, centre.x + r + GRID_M, GRID_M)
    ys = np.arange(centre.y - r, centre.y + r + GRID_M, GRID_M)
    xx, yy = np.meshgrid(xs, ys)
    inside = ((xx - centre.x) ** 2 + (yy - centre.y) ** 2) <= r ** 2
    yi, xi = np.where(inside)
    pts = gpd.GeoDataFrame(
        {"xi": xi, "yi": yi},
        geometry=[Point(xx[y, x], yy[y, x]) for y, x in zip(yi, xi)],
        crs=UTM,
    )
    return pts, centre.buffer(r)


def _load_water_mask(city, buf):
    import signal
    buf_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(epsg=4326).iloc[0]
    def _timeout(signum, frame): raise TimeoutError
    frames = []
    for tags in [{"natural": "water"}, {"landuse": "reservoir"}]:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(30)
        try:
            gdf = ox.features_from_polygon(buf_wgs, tags=tags)
            signal.alarm(0)
            polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not polys.empty:
                frames.append(polys[["geometry"]])
        except Exception:
            signal.alarm(0)
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], crs=UTM)
    water = pd.concat(frames, ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=4326).to_crs(epsg=UTM)
    water = gpd.clip(water[["geometry"]], buf)
    return water[~water.is_empty].copy()


def _apply_water_mask(points, water):
    if water.empty:
        return np.zeros(len(points), dtype=bool)
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], water[["geometry"]],
                       how="left", predicate="within")
    is_water = set(joined.dropna(subset=["index_right"]).index)
    return np.array([i in is_water for i in range(len(points))], dtype=bool)


def _sample_noah(points, noah):
    out = np.zeros(len(points), dtype=int)
    if noah.empty:
        return out
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]],
                       noah[["Var", "geometry"]], how="left", predicate="intersects")
    max_var = joined.groupby(["xi", "yi"])["Var"].max().reset_index()
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in max_var.iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map and not np.isnan(row["Var"]):
            out[idx_map[k]] = int(row["Var"])
    return out


def _rasterize_polygons(points, polys):
    counts = np.zeros(len(points), dtype=float)
    if polys.empty:
        return counts
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], polys[["geometry"]],
                       how="left", predicate="within")
    hit = (joined.dropna(subset=["index_right"])
               .groupby(["xi", "yi"]).size().reset_index(name="n"))
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in hit.iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map:
            counts[idx_map[k]] = float(row["n"])
    return counts


def _classify_tertile(score, water_mask):
    out = np.full(len(score), -1, dtype=int)
    valid = ~water_mask
    s   = score[valid]
    cls = np.zeros(len(s), dtype=int)
    flooded = s > 0
    if flooded.sum() > 0:
        t33, t67 = np.percentile(s[flooded], [33, 67])
        cls[flooded & (s <= t33)] = 1
        cls[flooded & (s > t33) & (s <= t67)] = 2
        cls[flooded & (s > t67)] = 3
    out[valid] = cls
    return out


def _freq_thresh(philsa_freq, water_mask):
    valid   = philsa_freq[~water_mask]
    flooded = valid[valid > 0]
    return float(np.percentile(flooded, 67)) if len(flooded) > 0 else 0.01


def _diff_bucket(diff):
    if diff <= -2: return "noah_much_higher"
    if diff == -1: return "noah_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "philsa_higher"
    return "philsa_much_higher"


def _consensus(noah_cls, freq, water, thresh):
    if water:
        return "water"
    hi_noah   = noah_cls >= NOAH_ACTIVE
    hi_philsa = freq     >= thresh
    if hi_noah and hi_philsa:   return "confirmed"
    if hi_noah:                 return "modelled"
    if hi_philsa:               return "empirical"
    return "low"


# ---------- GHS-SMOD urbanisation -------------------------------------------
# GHSL SMOD 2020 at 1km resolution, WGS84.
# Tile for Philippines: R10_C45 (approx) from GHS_SMOD_E2020_GLOBE_R2023A.
# Classes: 10=rural, 21=suburb/peri-urban, 22=semi-dense, 23=dense, 30=urban
GHSL_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/"
    "V1-0/tiles/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R8_C26.zip"
)

def _get_ghsl_smod():
    """Download and cache GHSL SMOD tile covering Philippines."""
    local_zip = os.path.join(GHSL_DIR, "ghsl_smod_phl.zip")
    if not os.path.exists(local_zip):
        print("  Downloading GHSL SMOD tile (Philippines)…", flush=True)
        try:
            r = requests.get(GHSL_URL, timeout=120, stream=True)
            r.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            print(f"  Downloaded → {local_zip}", flush=True)
        except Exception as ex:
            print(f"  WARN: GHSL download failed ({ex}) — urbanisation skipped")
            return None

    # Extract TIF from zip
    tif_path = os.path.join(GHSL_DIR, "ghsl_smod_phl.tif")
    if not os.path.exists(tif_path):
        try:
            with zipfile.ZipFile(local_zip) as zf:
                tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                if not tifs:
                    return None
                zf.extract(tifs[0], GHSL_DIR)
                extracted = os.path.join(GHSL_DIR, tifs[0])
                os.rename(extracted, tif_path)
        except Exception as ex:
            print(f"  WARN: GHSL extract failed ({ex})")
            return None
    return tif_path


def _sample_urban_class(pts_wgs, ghsl_path):
    """Sample GHSL SMOD class for each grid point. Returns str array."""
    result = np.array(["unknown"] * len(pts_wgs), dtype=object)
    if ghsl_path is None or not os.path.exists(ghsl_path):
        return result
    try:
        from PIL import Image as PILImage
        img = PILImage.open(ghsl_path)
        arr = np.array(img)
        # GHSL SMOD is in Mollweide (EPSG:54009); reproject points
        pts_moll = pts_wgs.to_crs("ESRI:54009")
        # Read geotransform from TIFF tags
        tag = img.tag_v2
        px_scale = tag.get(33550, (1000.0, 1000.0, 0))
        tiepoint = tag.get(33922, (0, 0, 0, -20037508.34, 9009964.0, 0))
        x0 = float(tiepoint[3])
        y0 = float(tiepoint[4])
        dx = float(px_scale[0])
        dy = float(px_scale[1])

        for i, geom in enumerate(pts_moll.geometry):
            col = int((geom.x - x0) / dx)
            row = int((y0 - geom.y) / dy)
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                val = int(arr[row, col])
                if val == 10:
                    result[i] = "water"
                elif val in (11, 12, 13):
                    result[i] = "rural"
                elif val == 21:
                    result[i] = "peri-urban"
                elif val in (22, 23, 30):
                    result[i] = "urban"
    except Exception as ex:
        print(f"  WARN urban sampling: {ex}")
    return result


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 72, flush=True)
print("NOAH vs PhilSA + GFD — Extended Consensus Analysis", flush=True)
print("=" * 72, flush=True)

print("\n[1/5] Loading observations (PhilSA + GFD)…", flush=True)
all_obs = load_all_observations()
all_utm = all_obs.to_crs(epsg=UTM)

n_philsa = all_obs[all_obs["source"] == "philsa"]["event_key"].nunique()
n_gfd    = all_obs[all_obs["source"] == "gfd"]["event_key"].nunique()
n_total  = all_obs["event_key"].nunique()
date_min = all_obs["event_date"].min().date()
date_max = all_obs["event_date"].max().date()
print(f"  PhilSA: {n_philsa} dates | GFD: {n_gfd} dates | "
      f"Total: {n_total} | {date_min} → {date_max}", flush=True)

print("\n[2/5] OSM water masks…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} polygons", flush=True)

print("\n[3/5] GHS-SMOD urbanisation tile…", flush=True)
ghsl_path = _get_ghsl_smod()

print("\n[4/5] Building grids…", flush=True)
city_data = {}

for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, buf   = _build_grid(city)
    water      = water_by_city[city["slug"]]
    water_mask = _apply_water_mask(pts, water)

    # NOAH
    noah_shp = _find_noah_shp(city["noah_province"])
    if noah_shp:
        noah = gpd.read_file(noah_shp)
        if noah.crs is None: noah = noah.set_crs(epsg=4326)
        if noah.crs.to_epsg() != UTM: noah = noah.to_crs(epsg=UTM)
        noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).astype(int)
        noah = gpd.clip(noah[["Var", "geometry"]], buf)
        noah = noah[~noah.is_empty].copy()
    else:
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)

    noah_cls = _sample_noah(pts, noah)

    # Combined observations
    obs_city = gpd.clip(all_utm[all_utm.intersects(buf)].copy(), buf)
    obs_city = obs_city[~obs_city.is_empty].copy()

    # Separate PhilSA vs GFD counts for timeline
    philsa_dates_city   = []
    philsa_sensors_city = []
    gfd_dates_city      = []

    obs_count    = np.zeros(len(pts), dtype=float)
    n_obs_events = 0

    for ekey, grp in obs_city.groupby("event_key"):
        c = _rasterize_polygons(pts, grp)
        obs_count += (c > 0).astype(float)
        n_obs_events += 1
        row = grp.iloc[0]
        if row["source"] == "philsa":
            philsa_dates_city.append(row["event_date"])
            philsa_sensors_city.append(row["sensor"])
        else:
            gfd_dates_city.append(row["event_date"])

    obs_freq   = obs_count / max(n_obs_events, 1)
    obs_cls    = _classify_tertile(obs_count, water_mask)
    thresh     = _freq_thresh(obs_freq, water_mask)

    consensus = [
        _consensus(int(noah_cls[i]), float(obs_freq[i]), bool(water_mask[i]), thresh)
        for i in range(len(pts))
    ]
    diff_buckets = [
        "water" if water_mask[i]
        else _diff_bucket(int(max(0, obs_cls[i])) - int(noah_cls[i]))
        for i in range(len(pts))
    ]

    # Urbanisation
    pts_wgs   = pts.to_crs(epsg=4326)
    urban_cls = _sample_urban_class(pts_wgs, ghsl_path)

    plot_df = pd.DataFrame({
        "lon":        pts_wgs.geometry.x.values,
        "lat":        pts_wgs.geometry.y.values,
        "noah_cls":   noah_cls,
        "obs_freq":   obs_freq,
        "obs_cls":    np.where(obs_cls == -1, 0, obs_cls),
        "is_water":   water_mask,
        "consensus":  consensus,
        "diff_bucket": diff_buckets,
        "urban_cls":  urban_cls,
    })

    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    # Stats
    valid   = ~water_mask
    n_cls   = noah_cls[valid]
    p_cls_s = np.where(obs_cls[valid] == -1, 0, obs_cls[valid])
    exact   = float(np.mean(n_cls == p_cls_s))
    within1 = float(np.mean(np.abs(n_cls.astype(int) - p_cls_s.astype(int)) <= 1))
    spearman = pd.Series(n_cls).corr(pd.Series(obs_count[valid]), method="spearman")
    rho_str  = f"{spearman:.3f}" if pd.notna(spearman) else "nan"

    cs = pd.Series(consensus)
    conf_s = float((cs == "confirmed").mean())
    mod_s  = float((cs == "modelled").mean())
    emp_s  = float((cs == "empirical").mean())

    print(f"    obs={n_obs_events} (p={len(philsa_dates_city)},g={len(gfd_dates_city)}) | "
          f"exact={exact:.3f} | ρ={rho_str} | thresh={thresh:.3f} | "
          f"conf={conf_s:.2f} mod={mod_s:.2f} emp={emp_s:.2f}", flush=True)

    city_data[city["slug"]] = {
        "city": city, "plot_df": plot_df, "extent": extent,
        "n_obs": n_obs_events, "n_philsa": len(philsa_dates_city),
        "n_gfd": len(gfd_dates_city),
        "n_noah": int((noah_cls > 0).sum()), "water_cells": int(water_mask.sum()),
        "exact": exact, "within_one": within1,
        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
        "conf_share": conf_s, "mod_share": mod_s, "emp_share": emp_s,
        "freq_thresh": thresh,
        "freq_max": float(obs_freq[valid].max()) if valid.any() else 1.0,
        "philsa_dates": philsa_dates_city, "philsa_sensors": philsa_sensors_city,
        "gfd_dates": gfd_dates_city,
    }


# ===========================================================================
# Figure 1 — Maps
# ===========================================================================
print("\n[5/5] Rendering figures…", flush=True)

freq_cmap = plt.cm.YlOrRd

PANEL_TITLES = [
    "",
    "NOAH 5-yr\n(modelled hazard)",
    f"Flood frequency\n(PhilSA+GFD, frac. of events)",
    "Consensus risk\n(NOAH + observations)",
    "Difference\n(obs tertile − NOAH class)",
]

fig1, axes1 = plt.subplots(
    len(CITIES), 5,
    figsize=(20.0, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.60, 1.0, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
n_philsa_all = sum(d["n_philsa"] for d in city_data.values())
n_gfd_all    = sum(d["n_gfd"]    for d in city_data.values())
fig1.suptitle(
    f"NOAH 5-yr Hazard vs Flood Observations  "
    f"({date_min} → {date_max})  —  "
    f"PhilSA SAR {n_philsa} dates  +  GFD/MODIS {n_gfd} dates",
    fontsize=12, fontweight="bold", y=0.999,
)
fig1.text(
    0.5, 0.974,
    f"Consensus: NOAH ≥ Medium  AND/OR  obs frequent (top tertile, per city)  |  "
    f"Water (OSM) = teal  |  NOAH-higher = purple",
    ha="center", fontsize=8, color="#444",
)
for ci, title in enumerate(PANEL_TITLES):
    axes1[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=5)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d    = city_data[slug]
    plot_df = d["plot_df"]
    rho_str = f"{d['spearman']:.2f}" if not np.isnan(d["spearman"]) else "nan"
    water_pts = plot_df[plot_df["is_water"]]

    city_freq_norm = mcolors.Normalize(vmin=0, vmax=max(d["freq_max"], 0.01))

    def _ax(ci):
        ax = axes1[ri, ci]
        ax.set_facecolor("#F1EDE5")
        ax.plot(city["lng"], city["lat"], "r*", markersize=4, zorder=5)
        ax.set_xlim(d["extent"][0], d["extent"][1])
        ax.set_ylim(d["extent"][2], d["extent"][3])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=5.5)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.grid(alpha=0.18)
        return ax

    # Col 0 – label
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(0.5, 0.5,
             f"{city['name']}\n{city['region']}\n"
             f"PhilSA={d['n_philsa']} dates\n"
             f"GFD={d['n_gfd']} dates\n"
             f"exact={d['exact']:.2f}\nρ={rho_str}\n\n"
             f"confirmed={d['conf_share']:.2f}\n"
             f"modelled={d['mod_share']:.2f}\n"
             f"empirical={d['emp_share']:.2f}",
             ha="center", va="center", fontsize=7, family="monospace",
             bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.92,
                       boxstyle="round,pad=0.35"))

    # Col 1 – NOAH
    ax = _ax(1)
    for k in [0, 1, 2, 3]:
        sub = plot_df[plot_df["noah_cls"] == k]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                       c=NOAH_COLORS[k], linewidths=0,
                       alpha=0.22 if k == 0 else 0.88)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)

    # Col 2 – flood frequency
    ax = _ax(2)
    zero   = plot_df[~plot_df["is_water"] & (plot_df["obs_freq"] == 0)]
    active = plot_df[~plot_df["is_water"] & (plot_df["obs_freq"] >  0)]
    if not zero.empty:
        ax.scatter(zero["lon"], zero["lat"], s=6, marker="s",
                   c="#F1EDE5", linewidths=0, alpha=0.25)
    if not active.empty:
        ax.scatter(active["lon"], active["lat"], s=6, marker="s",
                   c=freq_cmap(city_freq_norm(active["obs_freq"].values)),
                   linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98,
            f"max={d['freq_max']:.2f}\nthresh={d['freq_thresh']:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.12"))

    # Col 3 – consensus
    ax = _ax(3)
    for bucket in ["low", "modelled", "empirical", "confirmed", "water"]:
        sub = plot_df[plot_df["consensus"] == bucket]
        if sub.empty: continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=CONSENSUS_COLORS[bucket], linewidths=0,
                   alpha=0.20 if bucket == "low" else 0.90)
    ax.text(0.02, 0.98,
            f"conf={d['conf_share']:.2f}\n"
            f"mod={d['mod_share']:.2f}\n"
            f"emp={d['emp_share']:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))

    # Col 4 – difference
    ax = _ax(4)
    for bucket in ["noah_much_higher","noah_higher","match",
                   "philsa_higher","philsa_much_higher"]:
        sub = plot_df[plot_df["diff_bucket"] == bucket]
        if sub.empty: continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98, f"ρ={rho_str}",
            transform=ax.transAxes, va="top", fontsize=6, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))

# Legends
noah_h = [mpatches.Patch(color=NOAH_COLORS[k], label=f"NOAH {l}")
          for k, l in [(1,"Low"),(2,"Medium"),(3,"High")]]
cons_h = [
    mpatches.Patch(color=CONSENSUS_COLORS["confirmed"],  label="Confirmed risk"),
    mpatches.Patch(color=CONSENSUS_COLORS["modelled"],   label="Modelled only (NOAH)"),
    mpatches.Patch(color=CONSENSUS_COLORS["empirical"],  label="Empirical gap (obs)"),
    mpatches.Patch(color=CONSENSUS_COLORS["low"],        label="Low risk"),
]
diff_h = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"],  label="NOAH >> obs"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"],       label="NOAH > obs"),
    mpatches.Patch(color=DIFF_COLORS["match"],             label="Match"),
    mpatches.Patch(color=DIFF_COLORS["philsa_higher"],     label="obs > NOAH"),
    mpatches.Patch(color=DIFF_COLORS["philsa_much_higher"],label="obs >> NOAH"),
    mpatches.Patch(color=WATER_COLOR,                      label="Permanent water"),
]
fig1.legend(handles=noah_h+cons_h+diff_h, loc="lower center", ncol=6,
            fontsize=7, framealpha=0.92, bbox_to_anchor=(0.5, -0.008))
sm = plt.cm.ScalarMappable(cmap=freq_cmap, norm=mcolors.Normalize(0,1))
sm.set_array([])
cbar_ax = fig1.add_axes([0.40, -0.022, 0.10, 0.012])
cb = fig1.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Flood freq. (0→city max)", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
fig1.tight_layout(rect=[0, 0.03, 1, 0.968])
p1 = os.path.join(OUT_DIR, "noah_philsa_gfd_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# Figure 2 — Combined timeline
# ===========================================================================
fig2, axes2 = plt.subplots(len(CITIES), 1, figsize=(14, 1.6*len(CITIES)), sharex=True)
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle("Flood observation timeline — PhilSA (blue) + GFD/MODIS (orange)\n"
              "GFD extends record to 2000; note typhoon-driven clustering in both",
              fontsize=10, fontweight="bold")

all_dt   = pd.to_datetime(all_obs["event_date"].unique())
year_min = all_dt.min().year
year_max = all_dt.max().year

SENSOR_COLORS = {
    "s1":"#1976D2","s2":"#42A5F5","rcm":"#8E24AA","alos2":"#F57F17",
    "tdx":"#5D4037","iceye":"#E53935","saocom":"#FB8C00","gf3":"#546E7A",
    "gfd":"#E65100",
}

for ri, city in enumerate(CITIES):
    ax = axes2[ri]
    d  = city_data[city["slug"]]
    ax.set_facecolor("#FAFAFA")
    ax.set_yticks([])
    ax.set_ylabel(city["name"], fontsize=8, rotation=0, labelpad=55, va="center")

    for yr in range(year_min, year_max+1):
        ax.axvspan(pd.Timestamp(f"{yr}-06-01"),
                   pd.Timestamp(f"{yr}-11-30"),
                   color="#FFF3E0", alpha=0.5, zorder=0)

    for date, sensor in zip(d["philsa_dates"], d["philsa_sensors"]):
        col = SENSOR_COLORS.get(sensor, "#1976D2")
        ax.axvline(pd.Timestamp(date), color=col, lw=1.8, alpha=0.85, zorder=2)
        ax.scatter([pd.Timestamp(date)], [0.7], color=col, s=28, zorder=3, linewidths=0)

    for date in d["gfd_dates"]:
        col = SENSOR_COLORS["gfd"]
        ax.axvline(pd.Timestamp(date), color=col, lw=1.4, alpha=0.70,
                   linestyle="--", zorder=2)
        ax.scatter([pd.Timestamp(date)], [0.3], color=col, s=20,
                   zorder=3, linewidths=0, marker="D")

    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.2)
    ax.text(0.01, 0.88, f"PhilSA n={d['n_philsa']}  GFD n={d['n_gfd']}",
            transform=ax.transAxes, fontsize=6.5, va="top")

axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes2[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.setp(axes2[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

tl_handles = [
    mpatches.Patch(color="#1976D2", label="PhilSA S1"),
    mpatches.Patch(color="#E65100", label="GFD/MODIS"),
    mpatches.Patch(color="#FFF3E0", alpha=0.5, label="Typhoon season Jun–Nov"),
]
fig2.legend(handles=tl_handles, loc="lower center", ncol=4,
            fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5,-0.04))
fig2.tight_layout(rect=[0, 0.04, 1, 0.95])
p2 = os.path.join(OUT_DIR, "noah_philsa_gfd_02_timeline.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)


# ===========================================================================
# Figure 3 — Urbanisation stratification
# ===========================================================================
fig3, axes3 = plt.subplots(len(CITIES), 3,
                           figsize=(12, 3.0 * len(CITIES)),
                           gridspec_kw={"width_ratios": [0.5, 1.2, 1.0]})
fig3.patch.set_facecolor("#F7F7F7")
fig3.suptitle(
    "Consensus risk × Urbanisation (GHS-SMOD 2020)\n"
    "'Modelled only' concentrating in urban areas → drainage suppresses NOAH prediction\n"
    "'Empirical gap' in rural/peri-urban → drainage failures not in NOAH model",
    fontsize=10, fontweight="bold",
)

for ri, city in enumerate(CITIES):
    slug    = city["slug"]
    d       = city_data[slug]
    plot_df = d["plot_df"]

    # Left: urbanisation map
    ax = axes3[ri, 0]
    ax.axis("off")
    ax.text(0.5, 0.5,
            f"{city['name']}\n{city['region']}",
            ha="center", va="center", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.9,
                      boxstyle="round,pad=0.3"))

    # Middle: urban class map
    ax = axes3[ri, 1]
    ax.set_facecolor("#F5F5F5")
    for uc in ["rural", "peri-urban", "urban", "unknown", "water"]:
        mask = (plot_df["urban_cls"] == uc) if uc != "water" \
               else plot_df["is_water"]
        sub = plot_df[mask] if uc != "water" else plot_df[plot_df["is_water"]]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=5, marker="s",
                       c=URBAN_COLORS.get(uc, "#EEE"), linewidths=0, alpha=0.85)
    ax.set_xlim(d["extent"][0], d["extent"][1])
    ax.set_ylim(d["extent"][2], d["extent"][3])
    ax.set_aspect("equal"); ax.tick_params(labelsize=5.5)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(alpha=0.18)
    if ri == 0:
        ax.set_title("Urbanisation class\n(GHS-SMOD 2020)", fontsize=8, fontweight="bold")

    # Right: stacked bar — consensus × urban class
    ax = axes3[ri, 2]
    urban_cats = ["rural", "peri-urban", "urban"]
    cons_cats  = ["confirmed", "empirical", "modelled", "low"]
    cons_cols  = [CONSENSUS_COLORS[c] for c in cons_cats]

    valid_df = plot_df[~plot_df["is_water"]].copy()
    tab = pd.crosstab(valid_df["urban_cls"], valid_df["consensus"], normalize="index")
    tab = tab.reindex(index=urban_cats, columns=cons_cats, fill_value=0)

    bottoms = np.zeros(len(urban_cats))
    for ci2, (cc, col) in enumerate(zip(cons_cats, cons_cols)):
        vals = tab[cc].values if cc in tab.columns else np.zeros(len(urban_cats))
        ax.barh(urban_cats, vals, left=bottoms, color=col, alpha=0.85,
                label=cc if ri == 0 else "_")
        bottoms += vals

    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    if ri == 0:
        ax.set_title("Consensus × Urbanisation", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6.5, loc="lower right")

urban_h = [mpatches.Patch(color=URBAN_COLORS[k], label=k.capitalize())
           for k in ["rural", "peri-urban", "urban"]]
fig3.legend(handles=urban_h, loc="lower center", ncol=3,
            fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig3.tight_layout(rect=[0, 0.02, 1, 0.93])
p3 = os.path.join(OUT_DIR, "noah_philsa_gfd_03_urban.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved → {p3}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
for p in [p1, p2, p3]:
    print(f"  {p}", flush=True)
