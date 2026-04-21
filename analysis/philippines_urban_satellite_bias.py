"""
philippines_urban_satellite_bias.py
===================================
Summarize whether satellite flood mapping under-detects urban flooding in the
Philippines by comparing PhilSA, Groundsource, and NOAH across five study
cities.

Question
--------
Can Groundsource help show that satellite-derived flood extents are weaker in
urban areas, which would support using NOAH to retain city risk that satellites
may miss?

Method
------
For each city, on a 250 m grid:
- sample NOAH 5-year hazard class
- aggregate PhilSA all-file flood frequency (matched window: 2022-08-05 to 2026-02-09)
- aggregate and smooth Groundsource counts in the same matched window
- sample GHSL urban class

Then compare:
- PhilSA vs Groundsource agreement with NOAH
- any-hit support in NOAH-active cells
- hotspot support in NOAH-active cells
- diagnostic categories by urban class

Outputs
-------
  output/philippines_urban_satellite_bias_01_summary.png
  output/philippines_urban_satellite_bias_02_panelD.png
  output/philippines_urban_satellite_bias_02_panelD.pdf
  output/philippines_urban_satellite_bias_summary.csv
  output/philippines_urban_satellite_bias_by_urban.csv
"""

import glob
import os
import re
import signal
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "urban_bias"
PHILSA_DIR = ROOT / "data" / "philsa_satellite_flood"
GS_PARQUET = ROOT / "data" / "google_gemini_flood" / "groundsource_2026.parquet"
NOAH_BASE = ROOT / "data" / "noah" / "5yr"
GHSL_DIR = ROOT / "data" / "ghsl"

OUT_DIR.mkdir(exist_ok=True)
GHSL_DIR.mkdir(exist_ok=True)

UTM = 32651
GRID_M = 250
SIGMA_M = 750
SIGMA_CELLS = SIGMA_M / GRID_M
PERIOD_START = pd.Timestamp("2022-08-05")
PERIOD_END = pd.Timestamp("2026-02-09")
NOAH_ACTIVE = 2
GHSL_GLOBAL_X0 = -18041000.0
GHSL_GLOBAL_Y0 = 9009000.0
GHSL_TILE_M = 1_000_000.0

CITIES = [
    {"name": "Tuguegarao", "slug": "tuguegarao", "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan", "region": "Cagayan Valley"},
    {"name": "Dagupan", "slug": "dagupan", "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan", "region": "Ilocos"},
    {"name": "Manila", "slug": "manila", "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Cagayan de Oro", "slug": "cagayan_de_oro", "lat": 8.4772, "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental", "region": "Mindanao"},
    {"name": "Cotabato", "slug": "cotabato", "lat": 7.2236, "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao", "region": "BARMM"},
]

CITY_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]
DIAG_COLORS = {
    "supported": "#2E7D32",
    "sat_gap": "#6A1B9A",
    "weak_obs": "#546E7A",
    "philsa_only": "#E65100",
}
DIAG_LABELS = {
    "supported": "Both support NOAH",
    "sat_gap": "Groundsource supports, PhilSA misses",
    "weak_obs": "Weak in both",
    "philsa_only": "PhilSA only",
}


def _find_noah_shp(province):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = NOAH_BASE / folder
        if base.is_dir():
            shps = sorted(base.glob("*.shp"))
            if shps:
                return shps[0]
    return None


def _build_grid(city):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM)
        .iloc[0]
    )
    radius = city["radius_m"]
    xs = np.arange(centre.x - radius, centre.x + radius + GRID_M, GRID_M)
    ys = np.arange(centre.y - radius, centre.y + radius + GRID_M, GRID_M)
    xx, yy = np.meshgrid(xs, ys)
    inside = ((xx - centre.x) ** 2 + (yy - centre.y) ** 2) <= radius ** 2
    yi, xi = np.where(inside)
    points = gpd.GeoDataFrame(
        {"xi": xi.astype(int), "yi": yi.astype(int)},
        geometry=[Point(xx[y, x], yy[y, x]) for y, x in zip(yi, xi)],
        crs=UTM,
    )
    return points, xx, yy, inside, centre.buffer(radius)


def _load_water_mask(buf):
    buf_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(epsg=4326).iloc[0]

    def _timeout(signum, frame):
        raise TimeoutError

    parts = []
    for tags in [{"natural": "water"}, {"landuse": "reservoir"}]:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(30)
        try:
            gdf = ox.features_from_polygon(buf_wgs, tags=tags)
            signal.alarm(0)
            polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not polys.empty:
                parts.append(polys[["geometry"]])
        except Exception:
            signal.alarm(0)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry"], crs=UTM)
    water = pd.concat(parts, ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=4326).to_crs(epsg=UTM)
    water = gpd.clip(water[["geometry"]], buf)
    return water[~water.is_empty].copy()


def _apply_water_mask(points, water):
    if water.empty:
        return np.zeros(len(points), dtype=bool)
    joined = gpd.sjoin(
        points[["xi", "yi", "geometry"]],
        water[["geometry"]],
        how="left",
        predicate="within",
    )
    hits = set(joined.dropna(subset=["index_right"]).index)
    return np.array([idx in hits for idx in range(len(points))], dtype=bool)


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


def _rasterize_polygons(points, polys):
    counts = np.zeros(len(points), dtype=float)
    if polys.empty:
        return counts
    joined = gpd.sjoin(
        points[["xi", "yi", "geometry"]],
        polys[["geometry"]],
        how="left",
        predicate="within",
    )
    hit = (
        joined.dropna(subset=["index_right"])
        .groupby(["xi", "yi"])
        .size()
        .reset_index(name="n")
    )
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in hit.iterrows():
        counts[idx_map[(int(row["xi"]), int(row["yi"]))]] = float(row["n"])
    return counts


def _classify_positive_tertiles(score, water_mask):
    out = np.full(len(score), -1, dtype=int)
    valid = ~water_mask
    s = score[valid]
    cls = np.zeros(len(s), dtype=int)
    positive = s > 0
    if positive.sum() > 0:
        t33, t67 = np.percentile(s[positive], [33, 67])
        cls[positive & (s <= t33)] = 1
        cls[positive & (s > t33) & (s <= t67)] = 2
        cls[positive & (s > t67)] = 3
    out[valid] = cls
    return out


def _load_all_philsa():
    pattern = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    parts = []
    for fname in sorted(os.listdir(PHILSA_DIR)):
        match = pattern.match(fname)
        if not match:
            continue
        date_str, _, sensor, _ = match.groups()
        fpath = PHILSA_DIR / fname
        event_date = pd.to_datetime(date_str, format="%Y%m%d")
        try:
            with zipfile.ZipFile(fpath) as zf:
                shp_names = [name for name in zf.namelist() if name.lower().endswith(".shp")]
                if not shp_names:
                    continue
                gdf = gpd.read_file(f"zip://{fpath}!{shp_names[0]}")
            if gdf is None or gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
        except Exception as exc:
            print(f"    WARN PhilSA {fname}: {exc}", flush=True)
            continue
        event = gdf[["geometry"]].copy()
        event["event_date"] = event_date
        event["sensor"] = sensor.lower()
        event["event_key"] = os.path.splitext(fname)[0]
        parts.append(event)
    all_p = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_p, geometry="geometry", crs=4326)


def _load_groundsource():
    gdf = gpd.read_parquet(GS_PARQUET, columns=["start_date", "end_date", "geometry"])
    gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    gdf["end_date"] = pd.to_datetime(gdf["end_date"])
    gdf = gdf[(gdf["start_date"] <= PERIOD_END) & (gdf["end_date"] >= PERIOD_START)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()
    return gdf


def _ghsl_tile_indices(city):
    pt = gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326).to_crs("ESRI:54009").iloc[0]
    col = int((pt.x - GHSL_GLOBAL_X0) // GHSL_TILE_M) + 1
    row = int((GHSL_GLOBAL_Y0 - pt.y) // GHSL_TILE_M) + 1
    return row, col


def _get_ghsl_tile(row, col):
    zip_name = f"ghsl_smod_r{row}_c{col}.zip"
    tif_name = f"ghsl_smod_r{row}_c{col}.tif"
    local_zip = GHSL_DIR / zip_name
    tif_path = GHSL_DIR / tif_name
    if not local_zip.exists():
        url = (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/"
            f"V1-0/tiles/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R{row}_C{col}.zip"
        )
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        with open(local_zip, "wb") as handle:
            for chunk in response.iter_content(65536):
                handle.write(chunk)
    if not tif_path.exists():
        with zipfile.ZipFile(local_zip) as zf:
            tifs = [name for name in zf.namelist() if name.lower().endswith(".tif")]
            if not tifs:
                return None
            zf.extract(tifs[0], GHSL_DIR)
            extracted = GHSL_DIR / tifs[0]
            os.rename(extracted, tif_path)
    return tif_path


def _sample_urban_class(points_wgs, city):
    row, col = _ghsl_tile_indices(city)
    ghsl_path = _get_ghsl_tile(row, col)
    result = np.array(["unknown"] * len(points_wgs), dtype=object)
    if ghsl_path is None or not Path(ghsl_path).exists():
        return result
    img = Image.open(ghsl_path)
    arr = np.array(img)
    pts_moll = points_wgs.to_crs("ESRI:54009")
    tag = img.tag_v2
    px_scale = tag.get(33550, (1000.0, 1000.0, 0))
    tiepoint = tag.get(33922, (0, 0, 0, 0.0, 0.0, 0.0))
    x0 = float(tiepoint[3])
    y0 = float(tiepoint[4])
    dx = float(px_scale[0])
    dy = float(px_scale[1])
    for idx, geom in enumerate(pts_moll.geometry):
        col_idx = int((geom.x - x0) / dx)
        row_idx = int((y0 - geom.y) / dy)
        if 0 <= row_idx < arr.shape[0] and 0 <= col_idx < arr.shape[1]:
            val = int(arr[row_idx, col_idx])
            if val == 10:
                result[idx] = "water"
            elif val in (11, 12, 13):
                result[idx] = "rural"
            elif val == 21:
                result[idx] = "peri-urban"
            elif val in (22, 23, 30):
                result[idx] = "urban"
    return result


def _diag_category(noah_active, philsa_hot, gs_hot, is_water):
    if is_water:
        return "water"
    if noah_active and gs_hot and philsa_hot:
        return "supported"
    if noah_active and gs_hot and not philsa_hot:
        return "sat_gap"
    if noah_active and philsa_hot and not gs_hot:
        return "philsa_only"
    if noah_active and not gs_hot and not philsa_hot:
        return "weak_obs"
    return "background"


print("=" * 72, flush=True)
print("Philippines urban satellite-bias summary", flush=True)
print("=" * 72, flush=True)
print(
    f"Matched period: {PERIOD_START.date()} -> {PERIOD_END.date()} | "
    f"grid={GRID_M} m | GS sigma={SIGMA_M} m",
    flush=True,
)

print("\n[1/5] Loading PhilSA + Groundsource...", flush=True)
philsa = _load_all_philsa()
philsa_utm = philsa.to_crs(epsg=UTM)
gs = _load_groundsource()
gs_utm = gs.to_crs(epsg=UTM)
gs_utm = gs_utm[gs_utm.geometry.notna()].copy()
gs_utm = gs_utm[~gs_utm.is_empty].copy()
gs_utm = gs_utm[gs_utm.is_valid].copy()
print(
    f"  PhilSA files: {philsa['event_key'].nunique()} | "
    f"Groundsource polygons in window: {len(gs):,}",
    flush=True,
)

summary_rows = []
all_active_rows = []

print("\n[2/5] Per-city diagnosis...", flush=True)
for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    points, xx, yy, inside, buf = _build_grid(city)
    points_wgs = points.to_crs(epsg=4326)

    noah_shp = _find_noah_shp(city["noah_province"])
    if noah_shp is None:
        raise FileNotFoundError(f"No NOAH shapefile found for {city['noah_province']}")
    noah = gpd.read_file(noah_shp)
    if noah.crs is None:
        noah = noah.set_crs(epsg=4326)
    if noah.crs.to_epsg() != UTM:
        noah = noah.to_crs(epsg=UTM)
    noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).astype(int)
    noah = gpd.clip(noah[["Var", "geometry"]], buf)
    noah = noah[~noah.is_empty].copy()
    noah_cls = _sample_noah(points, noah)

    water = _load_water_mask(buf)
    water_mask = _apply_water_mask(points, water)

    philsa_city = gpd.clip(philsa_utm[philsa_utm.intersects(buf)].copy(), buf)
    philsa_city = philsa_city[~philsa_city.is_empty].copy()
    philsa_count = np.zeros(len(points), dtype=float)
    for _, grp in philsa_city.groupby("event_key"):
        philsa_count += (_rasterize_polygons(points, grp) > 0).astype(float)
    n_philsa_files = philsa_city["event_key"].nunique()
    philsa_cls = _classify_positive_tertiles(philsa_count, water_mask)
    philsa_any = philsa_count > 0

    gs_city = gpd.clip(gs_utm[gs_utm.intersects(buf)].copy(), buf)
    gs_city = gs_city[~gs_city.is_empty].copy()
    gs_count = _rasterize_polygons(points, gs_city)
    gs_any = gs_count > 0
    gs_grid = np.zeros_like(xx, dtype=float)
    gs_grid[points["yi"].astype(int), points["xi"].astype(int)] = gs_count
    gs_smooth = gaussian_filter(gs_grid, sigma=SIGMA_CELLS)
    gs_smooth[~inside] = np.nan
    gs_score = np.log1p(gs_smooth[inside])
    gs_cls = _classify_positive_tertiles(gs_score, water_mask)

    urban_cls = _sample_urban_class(points_wgs, city)

    valid = ~water_mask
    noah_active = noah_cls >= NOAH_ACTIVE
    philsa_hot = np.where(philsa_cls == -1, 0, philsa_cls) >= 2
    gs_hot = np.where(gs_cls == -1, 0, gs_cls) >= 2
    diag = np.array(
        [
            _diag_category(bool(noah_active[i]), bool(philsa_hot[i]), bool(gs_hot[i]), bool(water_mask[i]))
            for i in range(len(points))
        ],
        dtype=object,
    )

    philsa_rho = pd.Series(noah_cls[valid]).corr(pd.Series(philsa_count[valid]), method="spearman")
    gs_rho = pd.Series(noah_cls[valid]).corr(pd.Series(gs_score[valid]), method="spearman")

    active_mask = valid & noah_active
    active_rows = pd.DataFrame(
        {
            "city": city["name"],
            "slug": city["slug"],
            "urban_cls": urban_cls[active_mask],
            "diag": diag[active_mask],
        }
    )
    all_active_rows.append(active_rows)

    summary_rows.append(
        {
            "city": city["name"],
            "slug": city["slug"],
            "region": city["region"],
            "philsa_files_in_buffer": int(n_philsa_files),
            "groundsource_polygons_in_buffer": int(len(gs_city)),
            "noah_active_cells": int(active_mask.sum()),
            "philsa_spearman_noah_vs_count": float(philsa_rho) if pd.notna(philsa_rho) else np.nan,
            "groundsource_spearman_noah_vs_score": float(gs_rho) if pd.notna(gs_rho) else np.nan,
            "active_cells_with_any_philsa_share": float(philsa_any[active_mask].mean()),
            "active_cells_with_any_groundsource_share": float(gs_any[active_mask].mean()),
            "active_cells_with_hot_philsa_share": float(philsa_hot[active_mask].mean()),
            "active_cells_with_hot_groundsource_share": float(gs_hot[active_mask].mean()),
            "active_cells_supported_share": float((diag[active_mask] == "supported").mean()),
            "active_cells_sat_gap_share": float((diag[active_mask] == "sat_gap").mean()),
            "active_cells_weak_obs_share": float((diag[active_mask] == "weak_obs").mean()),
            "active_cells_philsa_only_share": float((diag[active_mask] == "philsa_only").mean()),
        }
    )
    print(
        f"    philsa_rho={philsa_rho:.3f} | gs_rho={gs_rho:.3f} | "
        f"any PhilSA={philsa_any[active_mask].mean():.2f} | "
        f"sat_gap={((diag[active_mask] == 'sat_gap').mean()):.2f}",
        flush=True,
    )

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "philippines_urban_satellite_bias_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n[3/5] Aggregating urban class breakdown...", flush=True)
all_active = pd.concat(all_active_rows, ignore_index=True)
urban_breakdown = pd.crosstab(
    all_active["urban_cls"],
    all_active["diag"],
    normalize="index",
).reindex(index=["rural", "peri-urban", "urban"], fill_value=0.0)
urban_breakdown = urban_breakdown.reindex(
    columns=["supported", "sat_gap", "weak_obs", "philsa_only"],
    fill_value=0.0,
)
urban_counts = (
    all_active["urban_cls"]
    .value_counts()
    .reindex(["rural", "peri-urban", "urban"], fill_value=0)
    .astype(int)
)
urban_out = urban_breakdown.copy()
urban_out.insert(0, "n_active_cells", urban_counts.values)
urban_csv = OUT_DIR / "philippines_urban_satellite_bias_by_urban.csv"
urban_out.to_csv(urban_csv, index_label="urban_cls")

print("\n[4/5] Rendering summary figure...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
fig.patch.set_facecolor("#F7F7F7")
fig.suptitle(
    "Do satellite flood extents under-detect urban flooding in the Philippines?\n"
    "PhilSA vs Groundsource vs NOAH across five city case studies",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
fig.text(
    0.5,
    0.953,
    "Matched window: 2022-08-05 to 2026-02-09. Groundsource is auxiliary evidence used to test whether PhilSA misses NOAH-active urban cells.",
    ha="center",
    fontsize=8.5,
    color="#444444",
)

city_order = summary_df["city"].tolist()
city_x = np.arange(len(city_order))
bar_w = 0.36

# A: Spearman by city
ax = axes[0, 0]
ax.bar(city_x - bar_w / 2, summary_df["philsa_spearman_noah_vs_count"], width=bar_w, color="#FB8C00", label="PhilSA vs NOAH")
ax.bar(city_x + bar_w / 2, summary_df["groundsource_spearman_noah_vs_score"], width=bar_w, color="#2E7D32", label="Groundsource vs NOAH")
ax.set_xticks(city_x, city_order, rotation=20, ha="right")
ax.set_ylabel("Spearman rho")
ax.set_title("A. Which source tracks NOAH better?", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8, framealpha=0.9)

# B: Any-hit support in active cells
ax = axes[0, 1]
ax.bar(city_x - bar_w / 2, summary_df["active_cells_with_any_philsa_share"], width=bar_w, color="#FB8C00", label="PhilSA any hit")
ax.bar(city_x + bar_w / 2, summary_df["active_cells_with_any_groundsource_share"], width=bar_w, color="#2E7D32", label="Groundsource any hit")
ax.set_xticks(city_x, city_order, rotation=20, ha="right")
ax.set_ylim(0, 1.02)
ax.set_ylabel("Share of NOAH-active cells")
ax.set_title("B. Any observed flooding inside NOAH-active cells", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8, framealpha=0.9)

# C: Diagnosis by city
ax = axes[1, 0]
bottom = np.zeros(len(summary_df))
for key in ["supported", "sat_gap", "weak_obs", "philsa_only"]:
    col_name = f"active_cells_{key}_share"
    vals = summary_df[col_name].values
    ax.bar(summary_df["city"], vals, bottom=bottom, color=DIAG_COLORS[key], alpha=0.9, label=key)
    bottom += vals
ax.set_ylim(0, 1.02)
ax.set_ylabel("Share of NOAH-active cells")
ax.set_title("C. How NOAH-active cells split by city", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=7, framealpha=0.9)

# D: Diagnosis by urban class
ax = axes[1, 1]
left = np.zeros(len(urban_breakdown.index))
for key in ["supported", "sat_gap", "weak_obs", "philsa_only"]:
    vals = urban_breakdown[key].values
    ax.barh(urban_breakdown.index, vals, left=left, color=DIAG_COLORS[key], alpha=0.9, label=key)
    left += vals
ax.set_xlim(0, 1)
ax.set_xlabel("Share of NOAH-active cells")
ax.set_title("D. NOAH-active cells by urban class", fontweight="bold")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = OUT_DIR / "philippines_urban_satellite_bias_01_summary.png"
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# Stand-alone paper-ready version of Panel D
paneld_fig, paneld_ax = plt.subplots(figsize=(8.6, 4.8))
paneld_fig.patch.set_facecolor("white")
paneld_ax.set_facecolor("white")
left = np.zeros(len(urban_breakdown.index))
for key in ["supported", "sat_gap", "weak_obs", "philsa_only"]:
    vals = urban_breakdown[key].values
    paneld_ax.barh(
        urban_breakdown.index,
        vals,
        left=left,
        color=DIAG_COLORS[key],
        alpha=0.96,
        label=DIAG_LABELS[key],
        edgecolor="white",
        linewidth=0.8,
    )
    for i, val in enumerate(vals):
        if val >= 0.08:
            x = left[i] + val / 2
            txt_color = "white" if key in {"sat_gap", "weak_obs"} else "black"
            paneld_ax.text(
                x,
                i,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
                fontweight="bold",
            )
    left += vals

paneld_ax.set_xlim(0, 1)
paneld_ax.set_xlabel("Share of NOAH-active cells")
paneld_ax.set_ylabel("")
paneld_ax.set_title(
    "Urban bias in satellite flood detection\n"
    "NOAH-active cells split by urban class",
    fontsize=12,
    fontweight="bold",
)
paneld_ax.grid(axis="x", alpha=0.2)
paneld_ax.set_yticklabels(
    [
        f"Rural  (n={urban_counts['rural']:,})",
        f"Peri-urban  (n={urban_counts['peri-urban']:,})",
        f"Urban  (n={urban_counts['urban']:,})",
    ]
)
paneld_ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.34),
    ncol=2,
    fontsize=8,
    framealpha=0.95,
)
paneld_fig.text(
    0.5,
    0.01,
    "Matched window: 2022-08-05 to 2026-02-09. Active cells are NOAH class >= medium, aggregated across five city case studies.",
    ha="center",
    fontsize=8,
    color="#444444",
)
paneld_fig.tight_layout(rect=[0, 0.08, 1, 0.92])
paneld_png = OUT_DIR / "philippines_urban_satellite_bias_02_panelD.png"
paneld_pdf = OUT_DIR / "philippines_urban_satellite_bias_02_panelD.pdf"
paneld_fig.savefig(paneld_png, dpi=180, bbox_inches="tight")
paneld_fig.savefig(paneld_pdf, bbox_inches="tight")
plt.close(paneld_fig)

print(f"  Saved -> {fig_path}", flush=True)
print(f"  Saved -> {paneld_png}", flush=True)
print(f"  Saved -> {paneld_pdf}", flush=True)
print(f"  Saved -> {summary_csv}", flush=True)
print(f"  Saved -> {urban_csv}", flush=True)

print("\n[5/5] Done.", flush=True)
