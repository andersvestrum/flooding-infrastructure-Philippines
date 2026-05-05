"""
noah_event_four_city_maps.py
============================
Four-city map figure linking NOAH 5-year hazard with satellite-observed flood
occurrence across the same major flood event windows used in the event-validation
analysis.

Cities
------
  Manila, San Fernando, Naga, Ilagan

Satellite sources
-----------------
  AI4G   — Sentinel-1 flood detections (Oct 2014 – Sep 2024)
  PhilSA — satellite-derived flood extents (Aug 2022 – Feb 2026)

Method
------
1. Reuse the fixed 250 m city buffers from the event-validation analysis.
2. For each named event window, rasterise AI4G and PhilSA detections to the
   city grid.
3. Keep only event-city-source combinations with at least 20 flooded cells,
   matching the minimum-support rule used in the event distribution figure.
4. Aggregate per-cell event occurrence frequency for AI4G, PhilSA, and their
   within-event union ("Any satellite").
5. Plot NOAH hazard classes alongside source-specific flood-occurrence maps.

Outputs
-------
  output/noah_validation/events/noah_event_05_fourcity_maps.png
  output/noah_validation/events/noah_event_05_fourcity_maps.pdf
  output/noah_validation/events/noah_event_05_fourcity_metrics.csv
  output/noah_validation/events/noah_event_05_fourcity_cells.parquet
"""

import os
import re
import zipfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.colors as mcolors
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "noah_validation", "events")
AI4G_DIR = os.path.join(ROOT, "data", "ai4g")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
NOAH_BASE = os.path.join(ROOT, "data", "noah", "5yr")
WATER_MASK_PARQUET = os.path.join(
    ROOT, "output", "noah_validation", "philsa", "noah_philsa_allfiles_cells.parquet"
)
GHSL_DIR = os.path.join(ROOT, "data", "ghsl")
SELECTED_EVENTS_CSV = os.path.join(OUT_DIR, "noah_event_selected_events.csv")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(AI4G_DIR, "philippines_floods.parquet")

UTM = 32651
GRID_M = 250
MIN_FLOODED_CELLS = 20
AI4G_START = pd.Timestamp("2014-10-01")
AI4G_END = pd.Timestamp("2024-09-30")
PHILSA_START = pd.Timestamp("2022-08-05")
PHILSA_END = pd.Timestamp("2026-02-09")

TARGET_CITY_SLUGS = ["manila", "san_fernando", "naga", "ilagan"]
FALLBACK_PLOT_EVENT_SLUGS = [
    "sendong_2011",
    "lando_2015",
    "lawin_2016",
    "ompong_2018",
    "vamco_2020",
    "egay_2023",
    "kristine_2024",
    "crising_2025",
]

NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR = "#A8D5E2"
SOURCE_CMAP = plt.cm.YlOrRd
GHSL_CLASS_MAP = {
    10: "water",
    11: "rural",
    12: "rural",
    13: "rural",
    21: "peri-urban",
    22: "urban",
    23: "urban",
    30: "urban",
}
GHSL_COLORS = {
    "rural": "#B0BEC5",
    "peri-urban": "#7CB342",
    "urban": "#37474F",
    "water": WATER_COLOR,
    "unknown": "#BDBDBD",
}

EVENTS = [
    {"name": "Ondoy\n(Sep 2009)", "slug": "ondoy_2009", "start": "2009-09-21", "end": "2009-10-12"},
    {"name": "Sendong\n(Dec 2011)", "slug": "sendong_2011", "start": "2011-12-16", "end": "2011-12-22"},
    {"name": "Pablo\n(Dec 2012)", "slug": "pablo_2012", "start": "2012-11-20", "end": "2012-12-18"},
    {"name": "Lando\n(Oct 2015)", "slug": "lando_2015", "start": "2015-10-18", "end": "2015-10-26"},
    {"name": "Lawin\n(Oct 2016)", "slug": "lawin_2016", "start": "2016-10-17", "end": "2016-10-24"},
    {"name": "Ompong\n(Sep 2018)", "slug": "ompong_2018", "start": "2018-09-13", "end": "2018-09-20"},
    {"name": "Vamco\n(Nov 2020)", "slug": "vamco_2020", "start": "2020-11-11", "end": "2020-11-17"},
    {"name": "Odette\n(Dec 2021)", "slug": "odette_2021", "start": "2021-12-15", "end": "2021-12-22"},
    {"name": "Karding\n(Sep 2022)", "slug": "karding_2022", "start": "2022-09-24", "end": "2022-09-30"},
    {"name": "Paeng\n(Oct 2022)", "slug": "paeng_2022", "start": "2022-10-27", "end": "2022-11-03"},
    {"name": "Egay\n(Jul 2023)", "slug": "egay_2023", "start": "2023-07-23", "end": "2023-07-30"},
    {"name": "Carina\n(Jul 2024)", "slug": "carina_2024", "start": "2024-07-22", "end": "2024-07-26"},
    {"name": "Enteng\n(Sep 2024)", "slug": "enteng_2024", "start": "2024-09-02", "end": "2024-09-12"},
    {"name": "Kristine\n(Oct 2024)", "slug": "kristine_2024", "start": "2024-10-22", "end": "2024-10-29"},
    {"name": "Nov seq.\n(2024)", "slug": "nov_sequence_2024", "start": "2024-11-12", "end": "2024-11-20"},
    {"name": "Crising\n(Jul 2025)", "slug": "crising_2025", "start": "2025-07-16", "end": "2025-07-23"},
    {"name": "Nando\n(Sep 2025)", "slug": "nando_2025", "start": "2025-09-22", "end": "2025-09-24"},
    {"name": "Tino/Uwan\n(Nov 2025)", "slug": "tino_uwan_2025", "start": "2025-11-04", "end": "2025-11-11"},
]

CITIES = [
    {"name": "Ilagan", "slug": "ilagan", "lat": 17.1485, "lng": 121.8892,
     "radius_m": 10_000, "noah_province": "Isabela", "region": "Cagayan Valley"},
    {"name": "San Fernando", "slug": "san_fernando", "lat": 15.0286, "lng": 120.6940,
     "radius_m": 12_000, "noah_province": "Pampanga", "region": "Central Luzon"},
    {"name": "Manila", "slug": "manila", "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Naga", "slug": "naga", "lat": 13.6218, "lng": 123.1948,
     "radius_m": 10_000, "noah_province": "Camarines Sur", "region": "Bicol"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_grid(city):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM).iloc[0]
    )
    r = city["radius_m"]
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
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(NOAH_BASE, folder)
        if os.path.isdir(base):
            shps = [p for p in os.listdir(base) if p.lower().endswith(".shp")]
            if shps:
                return os.path.join(base, shps[0])
    return None


def _sample_noah(points, noah):
    out = np.zeros(len(points), dtype=int)
    if noah.empty:
        return out
    joined = gpd.sjoin(
        points[["xi", "yi", "geometry"]],
        noah[["Var", "geometry"]],
        how="left",
        predicate="intersects",
    )
    max_var = joined.groupby(["xi", "yi"])["Var"].max().reset_index()
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in max_var.iterrows():
        key = (int(row["xi"]), int(row["yi"]))
        if key in idx_map and not np.isnan(row["Var"]):
            out[idx_map[key]] = int(row["Var"])
    return out


def _binary_from_ai4g(df_event, pts):
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
        how="left",
        predicate="contains",
    )
    for _, row in joined.dropna(subset=["index_right"]).iterrows():
        key = (int(row["xi"]), int(row["yi"]))
        if key in idx_map:
            flooded[idx_map[key]] = True
    return flooded


def _binary_from_philsa(philsa_event_gdf, pts):
    flooded = np.zeros(len(pts), dtype=bool)
    if philsa_event_gdf is None or philsa_event_gdf.empty:
        return flooded

    idx_map = {(int(r.xi), int(r.yi)): i for i, r in pts.reset_index().iterrows()}
    joined = gpd.sjoin(
        pts[["xi", "yi", "geometry"]],
        philsa_event_gdf[["geometry"]],
        how="left",
        predicate="within",
    )
    for _, row in joined.dropna(subset=["index_right"]).iterrows():
        key = (int(row["xi"]), int(row["yi"]))
        if key in idx_map:
            flooded[idx_map[key]] = True
    return flooded


def load_ai4g():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"AI4G cache not found: {CACHE_PATH}. Run the AI4G loader first."
        )
    df = pd.read_parquet(CACHE_PATH)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df


def load_philsa():
    pat = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    parts = []
    if not os.path.isdir(PHILSA_DIR):
        return None

    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = pat.match(fname)
        if not m:
            continue
        date_str = m.group(1)
        fpath = os.path.join(PHILSA_DIR, fname)
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


def load_water_masks():
    if not os.path.exists(WATER_MASK_PARQUET):
        return {}
    df = pd.read_parquet(WATER_MASK_PARQUET)[["city", "lon", "lat", "is_water"]].copy()
    df["key"] = list(zip(df["lon"].round(6), df["lat"].round(6)))
    out = {}
    for city_name, grp in df.groupby("city"):
        out[city_name] = set(grp.loc[grp["is_water"], "key"].tolist())
    return out


def _spearman_safe(noah_cls, freq, valid_mask):
    if valid_mask.sum() < 3:
        return float("nan")
    rho = pd.Series(noah_cls[valid_mask]).corr(pd.Series(freq[valid_mask]), method="spearman")
    return float(rho) if pd.notna(rho) else float("nan")


def _sample_ghsl(points_wgs):
    tile_paths = sorted(Path(GHSL_DIR).glob("ghsl_smod_r*_c*.tif"))
    if not tile_paths:
        return np.array(["unknown"] * len(points_wgs), dtype=object)

    points = points_wgs.to_crs("ESRI:54009")
    coords = np.array([(geom.x, geom.y) for geom in points.geometry])
    values = np.full(len(points), -9999, dtype=int)

    for path in tile_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            mask = (
                (values == -9999)
                & (coords[:, 0] >= b.left)
                & (coords[:, 0] < b.right)
                & (coords[:, 1] >= b.bottom)
                & (coords[:, 1] < b.top)
            )
            if not mask.any():
                continue
            sampled = list(src.sample(coords[mask]))
            values[np.where(mask)[0]] = [int(v[0]) for v in sampled]

    return np.array([GHSL_CLASS_MAP.get(v, "unknown") for v in values], dtype=object)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 72)
print("NOAH vs satellite event-occurrence maps — four-city focus")
print("=" * 72)

if os.path.exists(SELECTED_EVENTS_CSV):
    selected_order = pd.read_csv(SELECTED_EVENTS_CSV)["event"].tolist()
else:
    selected_order = FALLBACK_PLOT_EVENT_SLUGS[:]

selected_events = []
event_lookup = {e["slug"]: e for e in EVENTS}
for slug in selected_order:
    if slug in event_lookup:
        selected_events.append(event_lookup[slug])

print("\n[1/4] Loading cached observation data…", flush=True)
ai4g_all = load_ai4g()
philsa_all = load_philsa()
water_lookup = load_water_masks()
print(
    f"  AI4G: {len(ai4g_all):,} detections | "
    f"{ai4g_all['date'].min().date()} → {ai4g_all['date'].max().date()}",
    flush=True,
)
if philsa_all is not None:
    print(
        f"  PhilSA: {len(philsa_all):,} polygons | "
        f"{philsa_all['event_date'].min().date()} → {philsa_all['event_date'].max().date()}",
        flush=True,
    )
else:
    print("  PhilSA: not available", flush=True)

print("\n[2/4] Building city grids and NOAH classes…", flush=True)
city_results = {}
metrics_rows = []
cell_frames = []

for city in [c for c in CITIES if c["slug"] in TARGET_CITY_SLUGS]:
    pts, buf = _build_grid(city)
    pts_wgs = pts.to_crs(epsg=4326)
    noah_shp = _find_noah_shp(city["noah_province"])
    noah = gpd.read_file(noah_shp)
    if noah.crs is None:
        noah = noah.set_crs(epsg=4326)
    if noah.crs.to_epsg() != UTM:
        noah = noah.to_crs(epsg=UTM)
    noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    noah = gpd.clip(noah[["Var", "geometry"]], buf)
    noah = noah[~noah.is_empty].copy()
    noah_cls = _sample_noah(pts, noah)

    lon = pts_wgs.geometry.x.round(6)
    lat = pts_wgs.geometry.y.round(6)
    water_keys = water_lookup.get(city["name"], set())
    is_water = np.array([(float(x), float(y)) in water_keys for x, y in zip(lon, lat)], dtype=bool)
    ghsl_cls = _sample_ghsl(pts_wgs)
    ghsl_cls = np.where(is_water, "water", ghsl_cls)

    ai4g_count = np.zeros(len(pts), dtype=float)
    philsa_count = np.zeros(len(pts), dtype=float)
    any_count = np.zeros(len(pts), dtype=float)
    ai4g_valid_events = []
    philsa_valid_events = []
    any_valid_events = []

    bbox_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(epsg=4326).iloc[0].bounds

    for event in selected_events:
        t0 = pd.Timestamp(event["start"])
        t1 = pd.Timestamp(event["end"])

        raw_ai4g = np.zeros(len(pts), dtype=bool)
        if t1 >= AI4G_START and t0 <= AI4G_END:
            ai4g_ev = ai4g_all[(ai4g_all["date"] >= t0) & (ai4g_all["date"] <= t1)].copy()
            ai4g_city = ai4g_ev[
                (ai4g_ev["lat"] >= bbox_wgs[1]) & (ai4g_ev["lat"] <= bbox_wgs[3]) &
                (ai4g_ev["lon"] >= bbox_wgs[0]) & (ai4g_ev["lon"] <= bbox_wgs[2])
            ].copy()
            raw_ai4g = _binary_from_ai4g(ai4g_city, pts)
            if int(raw_ai4g.sum()) >= MIN_FLOODED_CELLS:
                ai4g_count += raw_ai4g.astype(float)
                ai4g_valid_events.append(event["slug"])

        raw_philsa = np.zeros(len(pts), dtype=bool)
        if philsa_all is not None and t1 >= PHILSA_START and t0 <= PHILSA_END:
            philsa_ev = philsa_all[
                (philsa_all["event_date"] >= t0) & (philsa_all["event_date"] <= t1)
            ].copy()
            if not philsa_ev.empty:
                philsa_city = philsa_ev[philsa_ev.intersects(buf)].copy()
                if not philsa_city.empty:
                    philsa_city = gpd.clip(philsa_city, buf)
                    raw_philsa = _binary_from_philsa(philsa_city, pts)
            if int(raw_philsa.sum()) >= MIN_FLOODED_CELLS:
                philsa_count += raw_philsa.astype(float)
                philsa_valid_events.append(event["slug"])

        any_raw = raw_ai4g | raw_philsa
        if int(any_raw.sum()) >= MIN_FLOODED_CELLS:
            any_count += any_raw.astype(float)
            any_valid_events.append(event["slug"])

    ai4g_freq = ai4g_count / max(len(ai4g_valid_events), 1)
    philsa_freq = philsa_count / max(len(philsa_valid_events), 1)
    any_freq = any_count / max(len(any_valid_events), 1)

    valid_land = ~is_water
    rho_ai4g = _spearman_safe(noah_cls, ai4g_freq, valid_land)
    rho_philsa = _spearman_safe(noah_cls, philsa_freq, valid_land)
    rho_any = _spearman_safe(noah_cls, any_freq, valid_land)
    urban_share = float(np.mean(ghsl_cls[valid_land] == "urban")) if valid_land.any() else float("nan")
    peri_share = float(np.mean(ghsl_cls[valid_land] == "peri-urban")) if valid_land.any() else float("nan")

    extent = [
        float(pts_wgs.geometry.x.min()),
        float(pts_wgs.geometry.x.max()),
        float(pts_wgs.geometry.y.min()),
        float(pts_wgs.geometry.y.max()),
    ]

    cell_df = pd.DataFrame({
        "city": city["name"],
        "region": city["region"],
        "lon": pts_wgs.geometry.x.values,
        "lat": pts_wgs.geometry.y.values,
        "noah_cls": noah_cls,
        "is_water": is_water,
        "ghsl_cls": ghsl_cls,
        "ai4g_freq": ai4g_freq,
        "philsa_freq": philsa_freq,
        "any_sat_freq": any_freq,
    })
    cell_frames.append(cell_df)

    city_results[city["slug"]] = {
        "city": city,
        "plot_df": cell_df,
        "extent": extent,
        "n_noah_cells": int((noah_cls > 0).sum()),
        "ai4g_valid_events": ai4g_valid_events,
        "philsa_valid_events": philsa_valid_events,
        "any_valid_events": any_valid_events,
        "rho_ai4g": rho_ai4g,
        "rho_philsa": rho_philsa,
        "rho_any": rho_any,
        "urban_share": urban_share,
        "peri_share": peri_share,
    }

    metrics_rows.append({
        "city": city["name"],
        "region": city["region"],
        "noah_hazard_cells": int((noah_cls > 0).sum()),
        "water_cells": int(is_water.sum()),
        "ai4g_valid_events": len(ai4g_valid_events),
        "philsa_valid_events": len(philsa_valid_events),
        "any_sat_valid_events": len(any_valid_events),
        "rho_ai4g": rho_ai4g,
        "rho_philsa": rho_philsa,
        "rho_any_sat": rho_any,
        "urban_share": urban_share,
        "periurban_share": peri_share,
        "ai4g_event_list": ",".join(ai4g_valid_events),
        "philsa_event_list": ",".join(philsa_valid_events),
        "any_sat_event_list": ",".join(any_valid_events),
    })

    print(
        f"  {city['name']}: NOAH cells={int((noah_cls > 0).sum()):,} | "
        f"AI4G valid={len(ai4g_valid_events)} | PhilSA valid={len(philsa_valid_events)} | "
        f"Any valid={len(any_valid_events)}",
        flush=True,
    )

metrics_df = pd.DataFrame(metrics_rows)
metrics_path = os.path.join(OUT_DIR, "noah_event_05_fourcity_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
cells_path = os.path.join(OUT_DIR, "noah_event_05_fourcity_cells.parquet")
pd.concat(cell_frames, ignore_index=True).to_parquet(cells_path, index=False)
print(f"  Saved → {metrics_path}", flush=True)
print(f"  Saved → {cells_path}", flush=True)


print("\n[3/4] Rendering four-city map figure…", flush=True)
fig, axes = plt.subplots(
    len(TARGET_CITY_SLUGS), 6,
    figsize=(21.8, 3.35 * len(TARGET_CITY_SLUGS)),
    gridspec_kw={"width_ratios": [0.80, 1.0, 1.0, 1.0, 1.0, 1.0]},
)
fig.patch.set_facecolor("#F7F7F7")
fig.suptitle(
    "NOAH 5-yr Hazard vs Satellite Flood Occurrence Across Major Flood Event Windows\n"
    "Four-city comparison: Manila, San Fernando, Naga, and Ilagan",
    fontsize=13.5, fontweight="bold", y=0.992,
)
fig.text(
    0.5, 0.948,
    f"Same 250 m city buffers and same selected major flood-event set as the event-distribution figure. "
    f"Fractions are computed independently over the subset of selected events inside each source period: "
    f"AI4G (2014–2024) and PhilSA (2022–2026). The combined panel uses the within-event union of the two sources. "
    f"Only event-city-source cases with >= {MIN_FLOODED_CELLS} flooded cells are counted. Final panel adds GHSL urbanization context.",
    ha="center", fontsize=8.0, color="#444444",
)

headers = [
    "",
    "NOAH 5-yr classes",
    "AI4G event fraction\n(2014–2024)",
    "PhilSA event fraction\n(2022–2026)",
    "Any satellite event fraction",
    "Urban context (GHSL)",
]
for i, title in enumerate(headers):
    axes[0, i].set_title(title, fontsize=10, fontweight="bold", pad=6)

source_norm = mcolors.PowerNorm(gamma=0.65, vmin=0, vmax=1)

def _setup_ax(ax, city, extent):
    ax.set_facecolor("#F1EDE5")
    ax.plot(city["lng"], city["lat"], "r*", markersize=5, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.grid(alpha=0.18)


ordered_cities = [next(c for c in CITIES if c["slug"] == slug) for slug in TARGET_CITY_SLUGS]
for ri, city in enumerate(ordered_cities):
    d = city_results[city["slug"]]
    plot_df = d["plot_df"]
    water_pts = plot_df[plot_df["is_water"]]

    lax = axes[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"NOAH cells={d['n_noah_cells']:,}\n"
        f"AI4G events={len(d['ai4g_valid_events'])}\n"
        f"PhilSA events={len(d['philsa_valid_events'])}\n"
        f"Any sat events={len(d['any_valid_events'])}\n"
        f"Urban={d['urban_share']:.2f}\n"
        f"Peri={d['peri_share']:.2f}\n"
        f"ρ AI4G={d['rho_ai4g']:.2f}\n"
        f"ρ PhilSA={d['rho_philsa']:.2f}\n"
        f"ρ Any={d['rho_any']:.2f}",
        ha="center", va="center",
        fontsize=7.4, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.93, boxstyle="round,pad=0.35"),
    )

    # NOAH panel
    ax = axes[ri, 1]
    _setup_ax(ax, city, d["extent"])
    for klass in [0, 1, 2, 3]:
        sub = plot_df[plot_df["noah_cls"] == klass]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"], s=7, marker="s",
            c=NOAH_COLORS[klass], linewidths=0,
            alpha=0.28 if klass == 0 else 0.90,
        )
    if not water_pts.empty:
        ax.scatter(
            water_pts["lon"], water_pts["lat"], s=7, marker="s",
            c=WATER_COLOR, linewidths=0, alpha=0.90,
        )

    for ci, field in enumerate(["ai4g_freq", "philsa_freq", "any_sat_freq"], start=2):
        ax = axes[ri, ci]
        _setup_ax(ax, city, d["extent"])
        zero = plot_df[~plot_df["is_water"] & (plot_df[field] == 0)]
        active = plot_df[~plot_df["is_water"] & (plot_df[field] > 0)]
        if not zero.empty:
            ax.scatter(
                zero["lon"], zero["lat"], s=7, marker="s",
                c="#F1EDE5", linewidths=0, alpha=0.28,
            )
        if not active.empty:
            ax.scatter(
                active["lon"], active["lat"], s=7, marker="s",
                c=SOURCE_CMAP(source_norm(active[field].values)),
                linewidths=0, alpha=0.92,
            )
        if not water_pts.empty:
            ax.scatter(
                water_pts["lon"], water_pts["lat"], s=7, marker="s",
                c=WATER_COLOR, linewidths=0, alpha=0.90,
            )

    ax = axes[ri, 5]
    _setup_ax(ax, city, d["extent"])
    for klass in ["rural", "peri-urban", "urban"]:
        sub = plot_df[plot_df["ghsl_cls"] == klass]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"], sub["lat"], s=7, marker="s",
            c=GHSL_COLORS[klass], linewidths=0,
            alpha=0.90 if klass != "rural" else 0.55,
        )
    if not water_pts.empty:
        ax.scatter(
            water_pts["lon"], water_pts["lat"], s=7, marker="s",
            c=WATER_COLOR, linewidths=0, alpha=0.90,
        )

noah_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="NOAH Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="NOAH Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="NOAH High"),
    mpatches.Patch(color=WATER_COLOR, label="Permanent water"),
]
fig.legend(
    handles=noah_handles,
    loc="lower center", ncol=4, fontsize=8,
    framealpha=0.92, bbox_to_anchor=(0.24, -0.01),
)
ghsl_handles = [
    mpatches.Patch(color=GHSL_COLORS["urban"], label="Urban"),
    mpatches.Patch(color=GHSL_COLORS["peri-urban"], label="Peri-urban"),
    mpatches.Patch(color=GHSL_COLORS["rural"], label="Rural"),
    mpatches.Patch(color=WATER_COLOR, label="Water"),
]
fig.legend(
    handles=ghsl_handles,
    loc="lower center", ncol=4, fontsize=8,
    framealpha=0.92, bbox_to_anchor=(0.70, -0.01),
)
sm = plt.cm.ScalarMappable(cmap=SOURCE_CMAP, norm=source_norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.41, -0.008, 0.18, 0.014])
cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Fraction of valid named event windows with observed flooding", fontsize=7.2)
cb.ax.tick_params(labelsize=6)

fig.tight_layout(rect=[0, 0.02, 1, 0.935])
png_path = os.path.join(OUT_DIR, "noah_event_05_fourcity_maps.png")
pdf_path = os.path.join(OUT_DIR, "noah_event_05_fourcity_maps.pdf")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {png_path}", flush=True)
print(f"  Saved → {pdf_path}", flush=True)

print("\n[4/4] Done.", flush=True)
