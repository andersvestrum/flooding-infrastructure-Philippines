"""
manila_philsa_noah_diagnosis.py
================================
Diagnose why PhilSA satellite flood mapping looks weak in Manila when
compared with the NOAH 5-year hazard map.

Question
--------
Is PhilSA weak in Manila because:
1. satellite flood detection is difficult in dense urban terrain, or
2. flooding drains too quickly / does not persist long enough to be seen?

Approach
--------
Use Manila only, on a common 250 m grid, and compare:
- NOAH 5-year hazard classes
- PhilSA all-file flood frequency (2022-08-05 to 2026-02-09)
- Groundsource observed flood polygons in the same matched window
- GHSL urbanisation class

Groundsource is not the headline comparison product here; it is auxiliary
evidence to help interpret whether PhilSA's weakness is a satellite-detection
problem or a genuine lack of persistent flooding.

Key diagnostic categories
-------------------------
- supported:      NOAH active and both PhilSA + Groundsource hotspot agree
- sat_gap:        NOAH active and Groundsource hotspot agrees, PhilSA does not
- weak_obs:       NOAH active but neither PhilSA nor Groundsource hotspot agree
- philsa_only:    NOAH active and PhilSA hotspot agrees, Groundsource does not
- obs_only:       NOAH not active but one or both observation sources are hot

Outputs
-------
  output/manila_philsa_noah_diagnosis_01_maps.png
  output/manila_philsa_noah_diagnosis_02_stats.png
  output/manila_philsa_noah_diagnosis_summary.csv
  output/manila_philsa_noah_diagnosis_by_urban.csv
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
import matplotlib.ticker as mticker
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output"
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

CITY = {
    "name": "Manila",
    "slug": "manila",
    "lat": 14.5995,
    "lng": 120.9842,
    "radius_m": 20_000,
    "noah_province": "Metropolitan Manila",
    "region": "NCR",
}

NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR = "#A8D5E2"
URBAN_COLORS = {
    "rural": "#A5D6A7",
    "peri-urban": "#FFF176",
    "urban": "#EF9A9A",
    "unknown": "#EEEEEE",
    "water": WATER_COLOR,
}
DIAG_COLORS = {
    "background": "#E8E8E8",
    "supported": "#2E7D32",
    "sat_gap": "#6A1B9A",
    "weak_obs": "#546E7A",
    "philsa_only": "#E65100",
    "obs_only": "#B2182B",
    "water": WATER_COLOR,
}
PHILSA_CMAP = {0: "#F1EDE5", 1: "#FFD54F", 2: "#FB8C00", 3: "#B71C1C"}
GS_CMAP = {0: "#F1EDE5", 1: "#C5E1A5", 2: "#66BB6A", 3: "#1B5E20"}

GHSL_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/"
    "V1-0/tiles/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0_R8_C30.zip"
)


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


def _get_ghsl_smod():
    local_zip = GHSL_DIR / "ghsl_smod_manila_r8_c30.zip"
    tif_path = GHSL_DIR / "ghsl_smod_manila_r8_c30.tif"
    if not local_zip.exists():
        response = requests.get(GHSL_URL, timeout=120, stream=True)
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


def _sample_urban_class(points_wgs, ghsl_path):
    result = np.array(["unknown"] * len(points_wgs), dtype=object)
    if ghsl_path is None or not Path(ghsl_path).exists():
        return result
    img = Image.open(ghsl_path)
    arr = np.array(img)
    pts_moll = points_wgs.to_crs("ESRI:54009")
    tag = img.tag_v2
    px_scale = tag.get(33550, (1000.0, 1000.0, 0))
    tiepoint = tag.get(33922, (0, 0, 0, -20037508.34, 9009964.0, 0))
    x0 = float(tiepoint[3])
    y0 = float(tiepoint[4])
    dx = float(px_scale[0])
    dy = float(px_scale[1])

    for idx, geom in enumerate(pts_moll.geometry):
        col = int((geom.x - x0) / dx)
        row = int((y0 - geom.y) / dy)
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            val = int(arr[row, col])
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
    if (not noah_active) and (philsa_hot or gs_hot):
        return "obs_only"
    return "background"


print("=" * 72, flush=True)
print("Manila PhilSA vs NOAH diagnostic", flush=True)
print("=" * 72, flush=True)
print(
    f"Period: {PERIOD_START.date()} -> {PERIOD_END.date()} | "
    f"grid={GRID_M} m | GS sigma={SIGMA_M} m",
    flush=True,
)

print("\n[1/6] Building Manila grid...", flush=True)
points, xx, yy, inside, buf = _build_grid(CITY)
points_wgs = points.to_crs(epsg=4326)

print("[2/6] Loading NOAH + water mask...", flush=True)
noah_shp = _find_noah_shp(CITY["noah_province"])
if noah_shp is None:
    raise FileNotFoundError(f"No NOAH shapefile found for {CITY['noah_province']}")
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

print("[3/6] Loading PhilSA all files...", flush=True)
philsa = _load_all_philsa()
philsa_utm = philsa.to_crs(epsg=UTM)
philsa_city = gpd.clip(philsa_utm[philsa_utm.intersects(buf)].copy(), buf)
philsa_city = philsa_city[~philsa_city.is_empty].copy()

philsa_count = np.zeros(len(points), dtype=float)
for _, grp in philsa_city.groupby("event_key"):
    philsa_count += (_rasterize_polygons(points, grp) > 0).astype(float)
n_philsa_files = philsa_city["event_key"].nunique()
philsa_freq = philsa_count / max(n_philsa_files, 1)
philsa_cls = _classify_positive_tertiles(philsa_count, water_mask)
philsa_sensor_counts = (
    philsa_city.groupby("sensor")["event_key"].nunique().sort_values(ascending=False)
)

print("[4/6] Loading matched-window Groundsource...", flush=True)
gs = _load_groundsource()
gs_utm = gs.to_crs(epsg=UTM)
gs_utm = gs_utm[gs_utm.geometry.notna()].copy()
gs_utm = gs_utm[~gs_utm.is_empty].copy()
gs_utm = gs_utm[gs_utm.is_valid].copy()
gs_city = gpd.clip(gs_utm[gs_utm.intersects(buf)].copy(), buf)
gs_city = gs_city[~gs_city.is_empty].copy()

gs_count = _rasterize_polygons(points, gs_city)
gs_grid = np.zeros_like(xx, dtype=float)
gs_grid[points["yi"].astype(int), points["xi"].astype(int)] = gs_count
gs_smooth = gaussian_filter(gs_grid, sigma=SIGMA_CELLS)
gs_smooth[~inside] = np.nan
gs_score = np.log1p(gs_smooth[inside])
gs_cls = _classify_positive_tertiles(gs_score, water_mask)
n_gs_records = len(gs_city)

print("[5/6] Sampling urbanisation + computing diagnostics...", flush=True)
ghsl_path = _get_ghsl_smod()
urban_cls = _sample_urban_class(points_wgs, ghsl_path)

valid = ~water_mask
noah_active = noah_cls >= NOAH_ACTIVE
philsa_hot = np.where(philsa_cls == -1, 0, philsa_cls) >= 2
gs_hot = np.where(gs_cls == -1, 0, gs_cls) >= 2
philsa_any = philsa_count > 0
gs_any = gs_count > 0
diag = np.array(
    [
        _diag_category(bool(noah_active[i]), bool(philsa_hot[i]), bool(gs_hot[i]), bool(water_mask[i]))
        for i in range(len(points))
    ],
    dtype=object,
)

philsa_rho = pd.Series(noah_cls[valid]).corr(pd.Series(philsa_count[valid]), method="spearman")
gs_rho = pd.Series(noah_cls[valid]).corr(pd.Series(gs_score[valid]), method="spearman")

philsa_norm = philsa_freq.copy()
philsa_norm_hi = np.percentile(philsa_norm[valid], 99.5) if valid.any() else 1.0
philsa_norm_hi = max(float(philsa_norm_hi), 1e-9)
philsa_norm = np.clip(philsa_norm / philsa_norm_hi, 0, 1)

gs_norm = gs_score.copy()
gs_norm_hi = np.percentile(gs_norm[valid], 99.5) if valid.any() else 1.0
gs_norm_hi = max(float(gs_norm_hi), 1e-9)
gs_norm = np.clip(gs_norm / gs_norm_hi, 0, 1)

records = pd.DataFrame(
    {
        "lon": points_wgs.geometry.x,
        "lat": points_wgs.geometry.y,
        "noah_cls": noah_cls,
        "philsa_cls": np.where(philsa_cls == -1, 0, philsa_cls),
        "gs_cls": np.where(gs_cls == -1, 0, gs_cls),
        "philsa_freq": philsa_freq,
        "gs_score": gs_score,
        "philsa_norm": philsa_norm,
        "gs_norm": gs_norm,
        "is_water": water_mask,
        "urban_cls": urban_cls,
        "diag": diag,
    }
)

active_df = records[(~records["is_water"]) & (records["noah_cls"] >= NOAH_ACTIVE)].copy()
urban_breakdown = pd.crosstab(
    active_df["urban_cls"],
    active_df["diag"],
    normalize="index",
).reindex(index=["rural", "peri-urban", "urban"], fill_value=0.0)
urban_breakdown = urban_breakdown.reindex(
    columns=["supported", "sat_gap", "weak_obs", "philsa_only"],
    fill_value=0.0,
)
urban_csv_path = OUT_DIR / "manila_philsa_noah_diagnosis_by_urban.csv"
urban_breakdown.to_csv(urban_csv_path)

summary = pd.DataFrame(
    [
        {
            "city": CITY["name"],
            "period_start": PERIOD_START.date().isoformat(),
            "period_end": PERIOD_END.date().isoformat(),
            "grid_m": GRID_M,
            "gs_sigma_m": SIGMA_M,
            "philsa_files_in_buffer": int(n_philsa_files),
            "groundsource_polygons_in_buffer": int(n_gs_records),
            "total_cells": int(len(records)),
            "water_cells": int(water_mask.sum()),
            "noah_active_cells": int((~records["is_water"] & (records["noah_cls"] >= NOAH_ACTIVE)).sum()),
            "philsa_spearman_noah_vs_count": float(philsa_rho) if pd.notna(philsa_rho) else np.nan,
            "groundsource_spearman_noah_vs_score": float(gs_rho) if pd.notna(gs_rho) else np.nan,
            "active_cells_supported_share": float((active_df["diag"] == "supported").mean()),
            "active_cells_sat_gap_share": float((active_df["diag"] == "sat_gap").mean()),
            "active_cells_weak_obs_share": float((active_df["diag"] == "weak_obs").mean()),
            "active_cells_philsa_only_share": float((active_df["diag"] == "philsa_only").mean()),
            "active_cells_with_any_philsa_share": float(philsa_any[valid & noah_active].mean()),
            "active_cells_with_any_groundsource_share": float(gs_any[valid & noah_active].mean()),
            "active_cells_with_hot_philsa_share": float(philsa_hot[valid & noah_active].mean()),
            "active_cells_with_hot_groundsource_share": float(gs_hot[valid & noah_active].mean()),
            "urban_active_supported_share": float(
                (active_df[active_df["urban_cls"] == "urban"]["diag"] == "supported").mean()
            ),
            "urban_active_sat_gap_share": float(
                (active_df[active_df["urban_cls"] == "urban"]["diag"] == "sat_gap").mean()
            ),
            "urban_active_weak_obs_share": float(
                (active_df[active_df["urban_cls"] == "urban"]["diag"] == "weak_obs").mean()
            ),
            "urban_active_philsa_only_share": float(
                (active_df[active_df["urban_cls"] == "urban"]["diag"] == "philsa_only").mean()
            ),
            "philsa_mean_norm_nohazard": float(records.loc[records["noah_cls"] == 0, "philsa_norm"].mean()),
            "philsa_mean_norm_low": float(records.loc[records["noah_cls"] == 1, "philsa_norm"].mean()),
            "philsa_mean_norm_medium": float(records.loc[records["noah_cls"] == 2, "philsa_norm"].mean()),
            "philsa_mean_norm_high": float(records.loc[records["noah_cls"] == 3, "philsa_norm"].mean()),
            "philsa_any_share_nohazard": float(philsa_any[(~water_mask) & (noah_cls == 0)].mean()),
            "philsa_any_share_low": float(philsa_any[(~water_mask) & (noah_cls == 1)].mean()),
            "philsa_any_share_medium": float(philsa_any[(~water_mask) & (noah_cls == 2)].mean()),
            "philsa_any_share_high": float(philsa_any[(~water_mask) & (noah_cls == 3)].mean()),
            "gs_mean_norm_nohazard": float(records.loc[records["noah_cls"] == 0, "gs_norm"].mean()),
            "gs_mean_norm_low": float(records.loc[records["noah_cls"] == 1, "gs_norm"].mean()),
            "gs_mean_norm_medium": float(records.loc[records["noah_cls"] == 2, "gs_norm"].mean()),
            "gs_mean_norm_high": float(records.loc[records["noah_cls"] == 3, "gs_norm"].mean()),
            "groundsource_any_share_nohazard": float(gs_any[(~water_mask) & (noah_cls == 0)].mean()),
            "groundsource_any_share_low": float(gs_any[(~water_mask) & (noah_cls == 1)].mean()),
            "groundsource_any_share_medium": float(gs_any[(~water_mask) & (noah_cls == 2)].mean()),
            "groundsource_any_share_high": float(gs_any[(~water_mask) & (noah_cls == 3)].mean()),
        }
    ]
)
summary_csv_path = OUT_DIR / "manila_philsa_noah_diagnosis_summary.csv"
summary.to_csv(summary_csv_path, index=False)

extent = [
    float(records["lon"].min()),
    float(records["lon"].max()),
    float(records["lat"].min()),
    float(records["lat"].max()),
]

print(
    f"  PhilSA rho={philsa_rho:.3f} | Groundsource rho={gs_rho:.3f} | "
    f"sat_gap(active)={(active_df['diag'] == 'sat_gap').mean():.3f} | "
    f"weak_obs(active)={(active_df['diag'] == 'weak_obs').mean():.3f}",
    flush=True,
)

print("[6/6] Rendering figures...", flush=True)

# Figure 1: maps
fig1, axes1 = plt.subplots(2, 2, figsize=(12.0, 11.0))
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    "Manila: NOAH vs PhilSA vs Groundsource Diagnostic\n"
    "Matched period 2022-08-05 to 2026-02-09, 250 m grid",
    fontsize=14,
    fontweight="bold",
    y=0.985,
)
fig1.text(
    0.5,
    0.958,
    "PhilSA uses all source files individually; Groundsource is smoothed (750 m sigma) and used as auxiliary evidence.",
    ha="center",
    fontsize=9,
    color="#444444",
)

map_specs = [
    ("noah_cls", "NOAH 5-year hazard", NOAH_COLORS),
    ("philsa_cls", f"PhilSA classes ({n_philsa_files} files)", PHILSA_CMAP),
    ("gs_cls", f"Groundsource classes ({n_gs_records:,} polygons)", GS_CMAP),
    ("diag", "Diagnosis", DIAG_COLORS),
]

for ax, (field, title, palette) in zip(axes1.ravel(), map_specs):
    ax.set_facecolor("#F1EDE5")
    ax.set_title(title, fontweight="bold")
    if field == "diag":
        order = ["background", "supported", "sat_gap", "weak_obs", "philsa_only", "obs_only", "water"]
        for key in order:
            if key == "water":
                sub = records[records["is_water"]]
            else:
                sub = records[records["diag"] == key]
            if sub.empty:
                continue
            alpha = 0.22 if key == "background" else 0.92
            ax.scatter(sub["lon"], sub["lat"], s=8, marker="s", c=palette[key], linewidths=0, alpha=alpha)
    else:
        for klass in [0, 1, 2, 3]:
            sub = records[records[field] == klass]
            if sub.empty:
                continue
            alpha = 0.25 if klass == 0 else 0.90
            ax.scatter(sub["lon"], sub["lat"], s=8, marker="s", c=palette[klass], linewidths=0, alpha=alpha)
        water_pts = records[records["is_water"]]
        if not water_pts.empty:
            ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s", c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.plot(CITY["lng"], CITY["lat"], "r*", markersize=6, zorder=10)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.grid(alpha=0.18)

legend_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="High"),
    mpatches.Patch(color=DIAG_COLORS["supported"], label="NOAH + PhilSA + GS support"),
    mpatches.Patch(color=DIAG_COLORS["sat_gap"], label="GS supports NOAH, PhilSA misses"),
    mpatches.Patch(color=DIAG_COLORS["weak_obs"], label="Neither observation source strongly supports"),
    mpatches.Patch(color=DIAG_COLORS["obs_only"], label="Observation hotspot outside NOAH"),
    mpatches.Patch(color=WATER_COLOR, label="Permanent water"),
]
fig1.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=8,
    framealpha=0.92,
    bbox_to_anchor=(0.5, -0.01),
)
fig1.tight_layout(rect=[0, 0.05, 1, 0.94])
fig1_path = OUT_DIR / "manila_philsa_noah_diagnosis_01_maps.png"
fig1.savefig(fig1_path, dpi=160, bbox_inches="tight")
plt.close(fig1)

# Figure 2: stats
fig2, axes2 = plt.subplots(2, 2, figsize=(12.0, 8.6))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle(
    "Manila diagnostic statistics: does PhilSA miss urban flooding?",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

# A: mean support by NOAH class
ax = axes2[0, 0]
classes = [0, 1, 2, 3]
labels = ["No hazard", "Low", "Medium", "High"]
x = np.arange(len(classes))
w = 0.34
philsa_means = [records.loc[records["noah_cls"] == k, "philsa_norm"].mean() for k in classes]
gs_means = [records.loc[records["noah_cls"] == k, "gs_norm"].mean() for k in classes]
ax.bar(x - w / 2, philsa_means, width=w, color="#FB8C00", label="PhilSA")
ax.bar(x + w / 2, gs_means, width=w, color="#2E7D32", label="Groundsource")
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.02)
ax.set_ylabel("Mean normalized support")
ax.set_title("A. Support rises with NOAH class?", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.legend(framealpha=0.9, fontsize=8)

# B: active NOAH cells by urban class
ax = axes2[0, 1]
order = ["supported", "sat_gap", "weak_obs", "philsa_only"]
left = np.zeros(len(urban_breakdown.index))
for key in order:
    vals = urban_breakdown[key].values if key in urban_breakdown.columns else np.zeros(len(urban_breakdown.index))
    ax.barh(urban_breakdown.index, vals, left=left, color=DIAG_COLORS[key], alpha=0.9, label=key)
    left += vals
ax.set_xlim(0, 1)
ax.set_xlabel("Share of NOAH-active cells")
ax.set_title("B. NOAH-active cells by urban class", fontweight="bold")
ax.grid(axis="x", alpha=0.25)
ax.legend(
    handles=[
        mpatches.Patch(color=DIAG_COLORS["supported"], label="Supported"),
        mpatches.Patch(color=DIAG_COLORS["sat_gap"], label="Satellite gap"),
        mpatches.Patch(color=DIAG_COLORS["weak_obs"], label="Weak in both"),
        mpatches.Patch(color=DIAG_COLORS["philsa_only"], label="PhilSA only"),
    ],
    fontsize=7,
    framealpha=0.9,
    loc="lower right",
)

# C: PhilSA sensor mix
ax = axes2[1, 0]
sensor_labels = philsa_sensor_counts.index.tolist()
sensor_vals = philsa_sensor_counts.values.tolist()
ax.bar(sensor_labels, sensor_vals, color="#1976D2", alpha=0.9)
ax.set_ylabel("Files intersecting Manila")
ax.set_title("C. PhilSA sensor mix in Manila", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", labelrotation=30)

# D: summary text
ax = axes2[1, 1]
ax.axis("off")
urban_active = active_df[active_df["urban_cls"] == "urban"]
summary_text = (
    f"PhilSA rho (NOAH vs count): {philsa_rho:.3f}\n"
    f"Groundsource rho (NOAH vs score): {gs_rho:.3f}\n\n"
    f"NOAH-active cells: {len(active_df):,}\n"
    f"  any PhilSA hit: {philsa_any[valid & noah_active].mean():.1%}\n"
    f"  any Groundsource hit: {gs_any[valid & noah_active].mean():.1%}\n"
    f"  supported by both: {(active_df['diag'] == 'supported').mean():.1%}\n"
    f"  GS supports, PhilSA misses: {(active_df['diag'] == 'sat_gap').mean():.1%}\n"
    f"  weak in both obs sources: {(active_df['diag'] == 'weak_obs').mean():.1%}\n\n"
    f"Urban NOAH-active cells only:\n"
    f"  supported by both: {(urban_active['diag'] == 'supported').mean():.1%}\n"
    f"  GS supports, PhilSA misses: {(urban_active['diag'] == 'sat_gap').mean():.1%}\n"
    f"  weak in both obs sources: {(urban_active['diag'] == 'weak_obs').mean():.1%}\n\n"
    "Interpretation:\n"
    "Large 'sat_gap' with much stronger GS than PhilSA\n"
    "points to urban satellite detectability limits.\n"
    "Large 'weak_obs' would point more toward drainage,\n"
    "timing, or NOAH overprediction."
)
ax.text(
    0.02,
    0.98,
    summary_text,
    va="top",
    ha="left",
    fontsize=9,
    family="monospace",
    bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.95, boxstyle="round,pad=0.4"),
)
ax.set_title("D. Key numbers", fontweight="bold", loc="left")

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2_path = OUT_DIR / "manila_philsa_noah_diagnosis_02_stats.png"
fig2.savefig(fig2_path, dpi=160, bbox_inches="tight")
plt.close(fig2)

print(f"  Saved -> {fig1_path}", flush=True)
print(f"  Saved -> {fig2_path}", flush=True)
print(f"  Saved -> {summary_csv_path}", flush=True)
print(f"  Saved -> {urban_csv_path}", flush=True)
