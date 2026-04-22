"""
noah_event_validation.py
========================
Event-specific validation: compare NOAH 5-yr hazard zones against SAR-detected
flood extents during specific major Philippine typhoon events.

Unlike noah_ai4g_comparison.py (which compares flood *frequency* over 10 years),
this script asks: during a specific named storm, did flooding occur where NOAH
predicted it? This is a more direct spatial test of NOAH's predictive accuracy.

Sources
-------
GFD    — Global Flood Database (Tellman et al. 2021, Nature), MODIS-based,
          2000–2018, ~250 m, downloaded from public GCS bucket gs://gfd_v1_4
AI4G   — Sentinel-1 SAR flood detections, Oct 2014 – Sep 2024
PhilSA — Multi-sensor SAR flood extents, 2022 – 2026

Events
------
Pre-2014 (GFD / MODIS):
  Typhoon Ondoy  (Ketsana) — 21–28 Sep 2009 — Manila / Central Luzon
  Typhoon Sendong (Washi)  — 16–22 Dec 2011 — Cagayan de Oro
  Typhoon Pablo  (Bopha)   — 02–18 Dec 2012 — Mindanao

Post-2014 (AI4G + GFD / PhilSA):
  Typhoon Lando  (Koppu)   — 18–26 Oct 2015 — Cagayan Valley
  Typhoon Lawin  (Haima)   — 17–24 Oct 2016 — Cagayan Valley
  Typhoon Ompong (Mangkhut)— 13–20 Sep 2018 — Cagayan Valley
  Typhoon Vamco  (Ulysses) — 11–17 Nov 2020 — Cagayan Valley
  Typhoon Odette (Rai)     — 15–22 Dec 2021 — Mindanao
  Typhoon Karding (Noru)   — 24–30 Sep 2022 — Central Luzon
  Typhoon Paeng  (Nalgae)  — 27 Oct – 3 Nov 2022 — Multiple
  Typhoon Egay   (Doksuri) — 23–30 Jul 2023 — Northern Luzon
  Typhoon Carina (Gaemi) + Habagat — 22–26 Jul 2024 — Luzon / NCR
  Tropical Storm Enteng (Yagi)      — 02–12 Sep 2024 — Luzon
  Severe Tropical Storm Kristine (Trami) — 22–29 Oct 2024 — Luzon / Bicol
  Late-2024 typhoon sequence        — 12–20 Nov 2024 — Luzon / Bicol
  TC Crising (Wipha) + Habagat      — 16–23 Jul 2025 — Multiple
  Typhoon Nando (Ragasa)            — 22–24 Sep 2025 — Northern Luzon
  Typhoon Tino/Uwan                 — 04–11 Nov 2025 — Visayas / Luzon

Outputs
-------
  output/noah_validation/events/noah_event_01_distribution.png
  output/noah_validation/events/noah_event_02_heatmap.png
  output/noah_validation/events/noah_event_summary.csv
  output/noah_validation/events/noah_event_selected_events.csv
"""

import glob
import io
import json
import os
import re
import urllib.parse
import urllib.request
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "output", "noah_validation", "events")
AI4G_DIR   = os.path.join(ROOT, "data", "ai4g")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
NOAH_BASE  = os.path.join(ROOT, "data", "noah", "5yr")
GFD_DIR    = os.path.join(ROOT, "data", "gfd")
os.makedirs(GFD_DIR, exist_ok=True)

GFD_BUCKET = "https://storage.googleapis.com/download/storage/v1/b/gfd_v1_4/o"
GFD_LIST   = "https://storage.googleapis.com/storage/v1/b/gfd_v1_4/o"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(AI4G_DIR, "philippines_floods.parquet")

UTM    = 32651
GRID_M = 250

# ---------------------------------------------------------------------------
# Typhoon events
# gfd_ids: Global Flood Database DFO event IDs (pre-2014, MODIS-based)
# start/end: used for AI4G + PhilSA filtering (post-2014 only)
# ---------------------------------------------------------------------------
EVENTS = [
    # ── Pre-2014: GFD / MODIS ────────────────────────────────────────────────
    {
        "name":    "Ondoy\n(Sep 2009)",
        "slug":    "ondoy_2009",
        "start":   "2009-09-21",
        "end":     "2009-10-12",
        "gfd_ids": [3544, 3551],   # Sep 21-28 and Sep 25-Oct 12
        "primary": {"Manila", "San Fernando", "Dagupan"},
    },
    {
        "name":    "Sendong\n(Dec 2011)",
        "slug":    "sendong_2011",
        "start":   "2011-12-16",
        "end":     "2011-12-22",
        "gfd_ids": [3884],
        "primary": {"Cagayan de Oro", "Butuan"},
    },
    {
        "name":    "Pablo\n(Dec 2012)",
        "slug":    "pablo_2012",
        "start":   "2012-11-20",
        "end":     "2012-12-18",
        "gfd_ids": [4006, 4009],   # Dec 2-4 and Nov 20-Dec 18
        "primary": {"Cagayan de Oro", "Butuan", "Cotabato"},
    },
    # ── 2015–2018: AI4G + GFD ────────────────────────────────────────────────
    {
        "name":    "Lando\n(Oct 2015)",
        "slug":    "lando_2015",
        "start":   "2015-10-18",
        "end":     "2015-10-26",
        "gfd_ids": [4300, 4301],
        "primary": {"Tuguegarao", "Ilagan", "Dagupan"},
    },
    {
        "name":    "Lawin\n(Oct 2016)",
        "slug":    "lawin_2016",
        "start":   "2016-10-17",
        "end":     "2016-10-24",
        "gfd_ids": [4412],
        "primary": {"Tuguegarao", "Ilagan"},
    },
    {
        "name":    "Ompong\n(Sep 2018)",
        "slug":    "ompong_2018",
        "start":   "2018-09-13",
        "end":     "2018-09-20",
        "gfd_ids": [4676],
        "primary": {"Tuguegarao", "Ilagan", "Dagupan"},
    },
    # ── Post-2020: AI4G + PhilSA ─────────────────────────────────────────────
    {
        "name":    "Vamco\n(Nov 2020)",
        "slug":    "vamco_2020",
        "start":   "2020-11-11",
        "end":     "2020-11-17",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan"},
    },
    {
        "name":    "Odette\n(Dec 2021)",
        "slug":    "odette_2021",
        "start":   "2021-12-15",
        "end":     "2021-12-22",
        "gfd_ids": [],
        "primary": {"Butuan", "Cagayan de Oro", "Cotabato"},
    },
    {
        "name":    "Karding\n(Sep 2022)",
        "slug":    "karding_2022",
        "start":   "2022-09-24",
        "end":     "2022-09-30",
        "gfd_ids": [],
        "primary": {"San Fernando", "Manila", "Dagupan"},
    },
    {
        "name":    "Paeng\n(Oct 2022)",
        "slug":    "paeng_2022",
        "start":   "2022-10-27",
        "end":     "2022-11-03",
        "gfd_ids": [],
        "primary": {"Manila", "Naga", "Daet", "Tuguegarao", "Dagupan"},
    },
    {
        "name":    "Egay\n(Jul 2023)",
        "slug":    "egay_2023",
        "start":   "2023-07-23",
        "end":     "2023-07-30",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan", "Dagupan"},
    },
    {
        "name":    "Carina\n(Jul 2024)",
        "slug":    "carina_2024",
        "start":   "2024-07-22",
        "end":     "2024-07-26",
        "gfd_ids": [],
        "primary": {"Manila", "San Fernando", "Dagupan"},
    },
    {
        "name":    "Enteng\n(Sep 2024)",
        "slug":    "enteng_2024",
        "start":   "2024-09-02",
        "end":     "2024-09-12",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan", "San Fernando", "Manila"},
    },
    {
        "name":    "Kristine\n(Oct 2024)",
        "slug":    "kristine_2024",
        "start":   "2024-10-22",
        "end":     "2024-10-29",
        "gfd_ids": [],
        "primary": {"Naga", "Daet", "Manila", "Tuguegarao"},
    },
    {
        "name":    "Nov seq.\n(2024)",
        "slug":    "nov_sequence_2024",
        "start":   "2024-11-12",
        "end":     "2024-11-20",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan", "Naga", "Daet"},
    },
    {
        "name":    "Crising\n(Jul 2025)",
        "slug":    "crising_2025",
        "start":   "2025-07-16",
        "end":     "2025-07-23",
        "gfd_ids": [],
        "primary": {"Manila", "San Fernando", "Dagupan", "Cotabato"},
    },
    {
        "name":    "Nando\n(Sep 2025)",
        "slug":    "nando_2025",
        "start":   "2025-09-22",
        "end":     "2025-09-24",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan"},
    },
    {
        "name":    "Tino/Uwan\n(Nov 2025)",
        "slug":    "tino_uwan_2025",
        "start":   "2025-11-04",
        "end":     "2025-11-11",
        "gfd_ids": [],
        "primary": {"Tuguegarao", "Ilagan", "Cagayan de Oro", "Butuan"},
    },
]

# Main-paper plotting subset.
#
# Decision rule, set before looking at NOAH capture rates:
# - keep major named Philippine flood/typhoon events with substantial observed
#   flood support in the city-grid sample;
# - preserve temporal/source coverage across GFD, AI4G, and PhilSA eras;
# - avoid zero-detection, near-duplicate, or sequence-heavy events that make the
#   city panels unreadable.
#
# The complete 18-event table is still written to noah_event_summary.csv.
PLOT_EVENT_SLUGS = [
    "sendong_2011",
    "lando_2015",
    "lawin_2016",
    "ompong_2018",
    "vamco_2020",
    "egay_2023",
    "kristine_2024",
    "crising_2025",
]
PLOT_EVENTS = [event for event in EVENTS if event["slug"] in PLOT_EVENT_SLUGS]

CITIES = [
    {"name": "Tuguegarao",     "slug": "tuguegarao",     "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan",             "region": "Cagayan Valley"},
    {"name": "Ilagan",         "slug": "ilagan",         "lat": 17.1485, "lng": 121.8892,
     "radius_m": 10_000, "noah_province": "Isabela",             "region": "Cagayan Valley"},
    {"name": "Dagupan",        "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan",          "region": "Ilocos"},
    {"name": "San Fernando",   "slug": "san_fernando",   "lat": 15.0286, "lng": 120.6940,
     "radius_m": 12_000, "noah_province": "Pampanga",            "region": "Central Luzon"},
    {"name": "Manila",         "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Naga",           "slug": "naga",           "lat": 13.6218, "lng": 123.1948,
     "radius_m": 10_000, "noah_province": "Camarines Sur",       "region": "Bicol"},
    {"name": "Daet",           "slug": "daet",           "lat": 14.1167, "lng": 122.9500,
     "radius_m":  8_000, "noah_province": "Camarines Norte",     "region": "Bicol"},
    {"name": "Cagayan de Oro", "slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental",    "region": "Northern Mindanao"},
    {"name": "Butuan",         "slug": "butuan",         "lat": 8.9515,  "lng": 125.5277,
     "radius_m": 10_000, "noah_province": None,                  "region": "Caraga"},
    {"name": "Cotabato",       "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao",         "region": "BARMM"},
]

NOAH_COLORS  = {0: "#DCDCDC", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
NOAH_LABELS  = {0: "No hazard", 1: "Low", 2: "Medium", 3: "High"}
EVENT_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A",          # pre-2014 GFD events
    "#E65100", "#B71C1C", "#00796B",          # 2015-2018 AI4G + GFD events
    "#F57C00", "#37474F", "#00838F", "#AD1457", "#5D4037",  # post-2020 events
    "#3949AB", "#00897B", "#D81B60", "#6D4C41", "#7CB342", "#C0CA33", "#5E35B1",
]
CITY_COLORS  = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C",
    "#00796B", "#F57C00", "#37474F", "#880E4F", "#1B5E20",
]
MIN_FLOODED_CELLS = 20   # minimum detections to count an event-city as valid


# ===========================================================================
# Shared helpers (consistent with other scripts in this project)
# ===========================================================================

def _build_grid(city):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM).iloc[0]
    )
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


def _find_noah_shp(province):
    if not province:
        return None
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(NOAH_BASE, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


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


def _binary_from_ai4g(df_event, pts):
    """Binary flood detection per 250m cell from AI4G point detections."""
    flooded = np.zeros(len(pts), dtype=bool)
    if df_event.empty:
        return flooded

    gdf = gpd.GeoDataFrame(
        df_event,
        geometry=gpd.points_from_xy(df_event["lon"], df_event["lat"]),
        crs=4326,
    ).to_crs(epsg=UTM)

    cell_polys = pts.copy()
    cell_polys["geometry"] = cell_polys.geometry.buffer(GRID_M / 2, cap_style=3)
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in pts.reset_index().iterrows()}

    joined = gpd.sjoin(
        cell_polys[["xi", "yi", "geometry"]],
        gdf[["geometry"]],
        how="left", predicate="contains",
    )
    for _, row in joined.dropna(subset=["index_right"]).iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map:
            flooded[idx_map[k]] = True
    return flooded


def _load_gfd_event(dfo_id):
    """
    Download and cache a GFD event zip from the public GCS bucket.
    Returns the 'flooded' band (band 1) as a rasterio DatasetReader-compatible
    MemoryFile, or None if download fails.
    Cache: data/gfd/DFO_{id}.tif
    """
    cache_tif = os.path.join(GFD_DIR, f"DFO_{dfo_id}.tif")
    if os.path.exists(cache_tif):
        return rasterio.open(cache_tif)

    # Find filename in bucket (prefix search)
    list_url = f"{GFD_LIST}?prefix=DFO_{dfo_id}_&maxResults=5"
    try:
        with urllib.request.urlopen(list_url) as r:
            listing = json.loads(r.read())
        items = listing.get("items", [])
        if not items:
            print(f"    WARN: DFO_{dfo_id} not found in bucket")
            return None
        fname = items[0]["name"]
    except Exception as e:
        print(f"    WARN: bucket listing failed for DFO_{dfo_id}: {e}")
        return None

    # Download zip
    dl_url = f"{GFD_BUCKET}/{urllib.parse.quote(fname, safe='')}?alt=media"
    print(f"    Downloading {fname} …", end=" ", flush=True)
    try:
        with urllib.request.urlopen(dl_url) as r:
            zip_bytes = r.read()
        print(f"{len(zip_bytes)/1e6:.1f} MB", flush=True)
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    # Extract flooded band (band 1) and save as single-band GeoTIFF
    tif_name = fname.replace(".zip", ".tif")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tif_bytes = zf.read(tif_name)

    with rasterio.MemoryFile(tif_bytes) as mf:
        with mf.open() as ds:
            flooded = ds.read(1)   # band 1 = flooded
            profile = ds.profile.copy()
            profile.update(count=1, dtype="float32")
            with rasterio.open(cache_tif, "w", **profile) as dst:
                dst.write(flooded.astype("float32"), 1)

    print(f"    Cached → {cache_tif}")
    return rasterio.open(cache_tif)


def _binary_from_gfd(dfo_ids, pts_wgs):
    """
    Sample GFD flooded pixels at grid cell centres (WGS84 points).
    Unions across all supplied DFO event IDs.
    Returns boolean array (True = flooded in at least one event).
    """
    import urllib.parse
    flooded = np.zeros(len(pts_wgs), dtype=bool)
    lons = pts_wgs.geometry.x.values
    lats = pts_wgs.geometry.y.values

    for dfo_id in dfo_ids:
        ds = _load_gfd_event(dfo_id)
        if ds is None:
            continue
        with ds:
            rows, cols = rasterio.transform.rowcol(ds.transform, lons, lats)
            rows = np.array(rows)
            cols = np.array(cols)
            h, w = ds.height, ds.width
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            arr   = ds.read(1)
            for i in np.where(valid)[0]:
                if arr[rows[i], cols[i]] == 1:
                    flooded[i] = True
    return flooded


def _binary_from_philsa(philsa_event_gdf, pts):
    """Binary flood detection per 250m cell from PhilSA polygons."""
    flooded = np.zeros(len(pts), dtype=bool)
    if philsa_event_gdf is None or philsa_event_gdf.empty:
        return flooded

    idx_map = {(int(r.xi), int(r.yi)): i for i, r in pts.reset_index().iterrows()}
    polys   = philsa_event_gdf[["geometry"]].copy()
    joined  = gpd.sjoin(pts[["xi", "yi", "geometry"]], polys,
                        how="left", predicate="within")
    for _, row in joined.dropna(subset=["index_right"]).iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map:
            flooded[idx_map[k]] = True
    return flooded


# ===========================================================================
# Data loading
# ===========================================================================

def load_ai4g():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"AI4G cache not found: {CACHE_PATH}\n"
            "Run noah_ai4g_comparison.py first to download and cache the data."
        )
    df = pd.read_parquet(CACHE_PATH)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df


def load_philsa():
    PAT   = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    parts = []
    if not os.path.isdir(PHILSA_DIR):
        return None
    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str = m.group(1)
        fpath     = os.path.join(PHILSA_DIR, fname)
        event_date = pd.to_datetime(date_str, format="%Y%m%d")
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
        except Exception:
            continue
        ev = gdf[["geometry"]].copy()
        ev["event_date"] = event_date
        parts.append(ev)

    if not parts:
        return None
    all_p = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_p, geometry="geometry", crs=4326).to_crs(epsg=UTM)


# ===========================================================================
# Main analysis
# ===========================================================================

print("=" * 72)
print("NOAH 5-yr — Event-Specific Validation (Typhoon Events)")
print("=" * 72)

print("\n[1/4] Loading AI4G cache…")
ai4g_all = load_ai4g()
print(f"  {len(ai4g_all):,} detections | {ai4g_all['date'].nunique()} dates | "
      f"{ai4g_all['date'].min().date()} → {ai4g_all['date'].max().date()}")

print("\n[2/4] Loading PhilSA shapefiles…")
philsa_all = load_philsa()
if philsa_all is not None:
    print(f"  {len(philsa_all):,} polygons | "
          f"{philsa_all['event_date'].min().date()} → {philsa_all['event_date'].max().date()}")
else:
    print("  PhilSA not available — using AI4G only.")

print("\n[3/4] Computing event × city flood detections vs NOAH…")

# Pre-build grids and NOAH for all cities (reused across events)
city_grids = {}
for city in CITIES:
    pts, buf = _build_grid(city)
    noah_shp  = _find_noah_shp(city["noah_province"])
    if noah_shp:
        noah = gpd.read_file(noah_shp)
        if noah.crs is None:
            noah = noah.set_crs(epsg=4326)
        if noah.crs.to_epsg() != UTM:
            noah = noah.to_crs(epsg=UTM)
        noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).astype(int)
        noah = gpd.clip(noah[["Var", "geometry"]], buf)
        noah = noah[~noah.is_empty].copy()
    else:
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)

    noah_cls = _sample_noah(pts, noah)
    pts_wgs  = pts.to_crs(epsg=4326)
    city_grids[city["slug"]] = {
        "pts":      pts,
        "pts_wgs":  pts_wgs,
        "buf":      buf,
        "noah_cls": noah_cls,
        "has_noah": noah_shp is not None,
    }
    print(f"  {city['name']}: {len(pts):,} cells | "
          f"NOAH High={int((noah_cls==3).sum())} Med={int((noah_cls==2).sum())} "
          f"Low={int((noah_cls==1).sum())} None={int((noah_cls==0).sum())}")

# Results table: rows = event-city combinations
summary_rows = []

# Per-event-city flood binary arrays — for figure
results = {}   # results[event_slug][city_slug] = dict

for event in EVENTS:
    t0  = pd.Timestamp(event["start"])
    t1  = pd.Timestamp(event["end"])
    results[event["slug"]] = {}

    gfd_ids = event.get("gfd_ids", [])
    use_gfd = len(gfd_ids) > 0

    # Filter AI4G to event window (post-2014 events only)
    ai4g_ev = ai4g_all[(ai4g_all["date"] >= t0) & (ai4g_all["date"] <= t1)].copy()
    n_ai4g_dates = int(ai4g_ev["date"].nunique())

    # Filter PhilSA to event window
    if philsa_all is not None:
        philsa_ev = philsa_all[
            (philsa_all["event_date"] >= t0) & (philsa_all["event_date"] <= t1)
        ].copy()
        n_philsa = len(philsa_ev)
    else:
        philsa_ev = None
        n_philsa  = 0

    src_label = f"GFD IDs={gfd_ids}" if use_gfd else \
                f"AI4G dates={n_ai4g_dates} | PhilSA polys={n_philsa}"
    print(f"\n  {event['slug']}: {src_label}")

    for city in CITIES:
        cg        = city_grids[city["slug"]]
        pts       = cg["pts"]
        pts_wgs   = cg["pts_wgs"]
        buf       = cg["buf"]
        noah_cls  = cg["noah_cls"]

        # GFD (pre-2014): sample raster at grid-cell centres
        flood_gfd = np.zeros(len(pts), dtype=bool)
        if use_gfd:
            flood_gfd = _binary_from_gfd(gfd_ids, pts_wgs)

        # AI4G: filter to city bbox then snap to grid
        b_wgs    = gpd.GeoSeries([buf], crs=UTM).to_crs(4326).iloc[0].bounds
        ai4g_city = ai4g_ev[
            (ai4g_ev["lat"] >= b_wgs[1]) & (ai4g_ev["lat"] <= b_wgs[3]) &
            (ai4g_ev["lon"] >= b_wgs[0]) & (ai4g_ev["lon"] <= b_wgs[2])
        ].copy()
        flood_ai4g = _binary_from_ai4g(ai4g_city, pts)

        # PhilSA: intersect with city buffer
        flood_philsa = np.zeros(len(pts), dtype=bool)
        if philsa_ev is not None and not philsa_ev.empty:
            philsa_city = philsa_ev[philsa_ev.intersects(buf)].copy()
            if not philsa_city.empty:
                philsa_city = gpd.clip(philsa_city, buf)
                flood_philsa = _binary_from_philsa(philsa_city, pts)

        # Union of all available sources
        flood_union = flood_gfd | flood_ai4g | flood_philsa
        n_flooded   = int(flood_union.sum())

        # Distribution: how many flooded cells fall in each NOAH class?
        dist = {k: int((flood_union & (noah_cls == k)).sum()) for k in [0, 1, 2, 3]}

        # Metrics vs NOAH (only cities with NOAH data)
        n_city_cells = len(pts)
        n_noah_high = int((noah_cls == 3).sum())
        n_noah_active = int((noah_cls >= 2).sum())   # Medium + High
        n_flooded_in_noah_active = int(dist[2] + dist[3])

        if n_flooded >= MIN_FLOODED_CELLS and cg["has_noah"]:
            # Hit rate: % of NOAH High cells that were flooded
            hit_rate = (
                float((flood_union & (noah_cls == 3)).sum()) / n_noah_high
                if n_noah_high > 0 else float("nan")
            )
            # Active hit rate: % of NOAH Medium+High cells flooded
            active_hit_rate = (
                float((flood_union & (noah_cls >= 2)).sum()) / n_noah_active
                if n_noah_active > 0 else float("nan")
            )
            # Observed-flood recall / capture:
            # % of detected flooded cells that fall inside NOAH Medium+High zones.
            capture_rate = (
                float(n_flooded_in_noah_active) / n_flooded if n_flooded > 0 else float("nan")
            )
            city_region_share_flood_inside_noah_active = (
                float(n_flooded_in_noah_active) / n_city_cells if n_city_cells > 0 else float("nan")
            )
        else:
            hit_rate = active_hit_rate = capture_rate = float("nan")
            city_region_share_flood_inside_noah_active = float("nan")

        results[event["slug"]][city["slug"]] = {
            "n_flooded":       n_flooded,
            "n_ai4g":          int(flood_ai4g.sum()),
            "n_philsa":        int(flood_philsa.sum()),
            "dist":            dist,
            "hit_rate":        hit_rate,
            "active_hit_rate": active_hit_rate,
            "capture_rate":    capture_rate,
            "city_region_share_flood_inside_noah_active": city_region_share_flood_inside_noah_active,
            "has_noah":        cg["has_noah"],
            "is_primary":      city["name"] in event["primary"],
        }

        flag = "*" if city["name"] in event["primary"] else " "
        cap_str = f"{capture_rate:.2f}" if not np.isnan(capture_rate) else "n/a"
        print(f"    {flag}{city['name']:<16}: flooded={n_flooded:>4} "
              f"(GFD={int(flood_gfd.sum()):>4} AI4G={int(flood_ai4g.sum()):>4} "
              f"PhilSA={int(flood_philsa.sum()):>4}) | capture={cap_str}")

        summary_rows.append({
            "event":            event["slug"],
            "event_name":       event["name"].replace("\n", " "),
            "city":             city["name"],
            "region":           city["region"],
            "is_primary":       city["name"] in event["primary"],
            "has_noah":         cg["has_noah"],
            "n_city_cells":     n_city_cells,
            "n_noah_active_cells": n_noah_active,
            "n_flooded_total":  n_flooded,
            "n_flooded_gfd":    int(flood_gfd.sum()),
            "n_flooded_ai4g":   int(flood_ai4g.sum()),
            "n_flooded_philsa": int(flood_philsa.sum()),
            "n_in_noah_none":   dist[0],
            "n_in_noah_low":    dist[1],
            "n_in_noah_med":    dist[2],
            "n_in_noah_high":   dist[3],
            "n_flooded_in_noah_active": n_flooded_in_noah_active,
            "capture_rate":     capture_rate,
            "observed_flood_recall_noah_active": capture_rate,
            "city_region_share_flood_inside_noah_active": city_region_share_flood_inside_noah_active,
            "hit_rate_high":    hit_rate,
            "active_hit_rate":  active_hit_rate,
        })

summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(OUT_DIR, "noah_event_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}")

selected_df = (
    summary_df[summary_df["event"].isin(PLOT_EVENT_SLUGS)]
    .groupby(["event", "event_name"], as_index=False)
    .agg(
        n_flooded_total=("n_flooded_total", "sum"),
        n_flooded_gfd=("n_flooded_gfd", "sum"),
        n_flooded_ai4g=("n_flooded_ai4g", "sum"),
        n_flooded_philsa=("n_flooded_philsa", "sum"),
        n_cities_with_20plus_cells=("n_flooded_total", lambda s: int((s >= MIN_FLOODED_CELLS).sum())),
    )
)
selected_df["plot_order"] = selected_df["event"].map(
    {slug: idx + 1 for idx, slug in enumerate(PLOT_EVENT_SLUGS)}
)
selected_df = selected_df.sort_values("plot_order")
selected_df["selection_rule"] = (
    "Major named event with substantial observed flood support; selected before "
    "NOAH capture-rate interpretation to balance time period, source coverage, "
    "and plot legibility."
)
selected_csv_path = os.path.join(OUT_DIR, "noah_event_selected_events.csv")
selected_df.to_csv(selected_csv_path, index=False)
print(f"  Saved → {selected_csv_path}")


# ===========================================================================
# Figure 1 — Distribution of detected flooding across NOAH classes
# ===========================================================================
print("\n[4/4] Rendering figures…")

fig1, axes1 = plt.subplots(
    2, 5,
    figsize=(22, 9),
    sharey=False,
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    "Where Did Typhoon Flooding Fall Relative to NOAH 5-yr Hazard Classes?\n"
    "Selected major events: distribution of detected flooded cells across NOAH hazard zones",
    fontsize=13, fontweight="bold", y=1.01,
)

event_names_short = [e["name"] for e in PLOT_EVENTS]
bar_width = 0.14
x_offsets = np.linspace(-(len(PLOT_EVENTS) - 1) * bar_width / 2,
                         (len(PLOT_EVENTS) - 1) * bar_width / 2,
                         len(PLOT_EVENTS))

for ci, city in enumerate(CITIES):
    row = ci // 5
    col = ci % 5
    ax  = axes1[row, col]
    ax.set_facecolor("white")

    has_any_data = False
    for ei, event in enumerate(PLOT_EVENTS):
        r = results[event["slug"]][city["slug"]]
        n = r["n_flooded"]
        if n < MIN_FLOODED_CELLS:
            continue

        has_any_data = True
        dist   = r["dist"]
        fracs  = [dist[k] / n for k in [0, 1, 2, 3]]
        x_base = np.arange(4)
        x_pos  = x_base + x_offsets[ei]

        # Draw stacked bar per NOAH class
        bottom = 0.0
        for ki, k in enumerate([0, 1, 2, 3]):
            height = fracs[ki]
            if height > 0:
                ax.bar(
                    ei, height,
                    bottom=bottom,
                    color=NOAH_COLORS[k],
                    edgecolor="white",
                    linewidth=0.5,
                    width=0.65,
                    label=NOAH_LABELS[k] if ci == 0 else "_",
                )
                if height > 0.06:
                    ax.text(
                        ei, bottom + height / 2,
                        f"{height:.0%}",
                        ha="center", va="center",
                        fontsize=6.5, color="white" if k >= 2 else "#333",
                        fontweight="bold",
                    )
            bottom += height

        # Annotate observed-flood recall and sample size at top.
        recall = r["capture_rate"]
        recall_label = f"R={recall:.0%}" if not np.isnan(recall) else "R=n/a"
        ax.text(
            ei, 1.025, f"{recall_label}\nn={n}",
            ha="center", va="bottom",
            fontsize=5.8, color=EVENT_COLORS[ei],
            linespacing=0.9,
        )

    ax.set_title(
        f"{city['name']}\n{city['region']}",
        fontsize=9, fontweight="bold", pad=4,
    )
    ax.set_xticks(range(len(PLOT_EVENTS)))
    ax.set_xticklabels(
        [e["name"] for e in PLOT_EVENTS],
        fontsize=7, rotation=15, ha="right",
    )
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
    ax.yaxis.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if not has_any_data:
        ax.text(0.5, 0.5, "No SAR detections\nduring any event",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#888", style="italic")

    if not city_grids[city["slug"]]["has_noah"]:
        ax.set_facecolor("#FFF8F0")
        ax.text(0.98, 0.98, "No NOAH data",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=7, color="#B71C1C", style="italic")

axes1[0, 0].set_ylabel("Fraction of detected flooded cells", fontsize=8)
axes1[1, 0].set_ylabel("Fraction of detected flooded cells", fontsize=8)

# Legend
legend_handles = [
    mpatches.Patch(color=NOAH_COLORS[3], label="NOAH High (>1.5 m)"),
    mpatches.Patch(color=NOAH_COLORS[2], label="NOAH Medium (0.5–1.5 m)"),
    mpatches.Patch(color=NOAH_COLORS[1], label="NOAH Low (0–0.5 m)"),
    mpatches.Patch(color=NOAH_COLORS[0], label="NOAH No hazard"),
]
fig1.legend(handles=legend_handles, loc="lower center", ncol=4,
            fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, -0.03))
fig1.text(
    0.5, -0.055,
    f"Each bar = one selected typhoon event. Height = fraction of detected flooded 250m cells falling in that NOAH class. "
    f"R = observed-flood recall = flooded cells in NOAH Medium+High divided by all observed flooded cells within the fixed city buffer. "
    f"AI4G (Sentinel-1, 2014–2024) + PhilSA (2022–2026) union. Events with < {MIN_FLOODED_CELLS} detected cells excluded. "
    f"Full 18-event table is retained in CSV; plot subset chosen by pre-defined event/source/legibility rule. "
    f"Primary city for each event marked (*) in data. Permanent water not masked here.",
    ha="center", fontsize=7.5, color="#555", style="italic",
)

fig1.tight_layout(rect=[0, 0.04, 1, 1.0])
p1 = os.path.join(OUT_DIR, "noah_event_01_distribution.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}")


# ===========================================================================
# Figure 2 — Hit rate heatmap (events × cities)
# ===========================================================================

# Two heatmaps side by side:
#   Left:  capture_rate = % of detected flooding in NOAH Medium+High
#   Right: active_hit_rate = % of NOAH Medium+High cells that were flooded

n_ev = len(PLOT_EVENTS)
fig2, axes2 = plt.subplots(1, 2, figsize=(max(18, n_ev * 2.2), 6))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle(
    "Event-Specific NOAH Validation — Spatial Accuracy Heatmaps",
    fontsize=13, fontweight="bold",
)

city_labels  = [f"{c['name']}\n{c['region']}" for c in CITIES]
event_labels = [e["name"] for e in PLOT_EVENTS]

for ax_idx, (metric_key, title, cbar_label) in enumerate([
    ("capture_rate",    "Capture rate\n% of detected flooding inside NOAH Medium+High zones",
     "Fraction of SAR flood in NOAH Med+High"),
    ("active_hit_rate", "Hit rate\n% of NOAH Medium+High cells detected as flooded",
     "Fraction of NOAH Med+High cells detected"),
]):
    ax = axes2[ax_idx]

    mat = np.full((len(PLOT_EVENTS), len(CITIES)), np.nan)
    for ei, event in enumerate(PLOT_EVENTS):
        for ci, city in enumerate(CITIES):
            mat[ei, ci] = results[event["slug"]][city["slug"]][metric_key]

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(CITIES)))
    ax.set_xticklabels(city_labels, fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(PLOT_EVENTS)))
    ax.set_yticklabels(event_labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Annotate cells
    for ei in range(len(PLOT_EVENTS)):
        for ci in range(len(CITIES)):
            val = mat[ei, ci]
            if np.isnan(val):
                txt = "—"
                col = "#888"
            else:
                txt = f"{val:.0%}"
                col = "white" if val > 0.6 or val < 0.25 else "black"
            is_primary = CITIES[ci]["name"] in PLOT_EVENTS[ei]["primary"]
            fw = "bold" if is_primary else "normal"
            ax.text(ci, ei, txt, ha="center", va="center",
                    fontsize=8, color=col, fontweight=fw)

    # Mark NaN cells grey
    for ei in range(len(PLOT_EVENTS)):
        for ci in range(len(CITIES)):
            if np.isnan(mat[ei, ci]):
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ei - 0.5), 1, 1,
                    fill=True, facecolor="#DCDCDC", zorder=0,
                ))

    plt.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)

fig2.text(
    0.5, -0.02,
    "Bold = city is a primary impact zone for that typhoon. "
    "Grey = fewer than 20 detected cells or no NOAH data available. "
    "Source: AI4G Sentinel-1 (2014–2024) + PhilSA SAR (2022–2026) union vs NOAH 5-yr hazard.",
    ha="center", fontsize=8, color="#555", style="italic",
)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "noah_event_02_heatmap.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}")

print("\n" + "=" * 72)
print("DONE")
print(f"  {p1}")
print(f"  {p2}")
print(f"  {csv_path}")
