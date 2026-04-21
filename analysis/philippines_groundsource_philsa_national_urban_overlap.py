"""
philippines_groundsource_philsa_national_urban_overlap.py
=========================================================
National supporting analysis for urban satellite bias.

Question
--------
Across the Philippines, do Groundsource-observed flood polygons overlap
PhilSA flood extents less often in urban settings than in rural settings?

This analysis does not use NOAH directly. It is designed as a national
supporting check because NOAH coverage in this repository is not nationwide.

Method
------
1. Load all PhilSA flood polygons in the matched window (2022-08-05 to
   2026-02-09).
2. Load all Groundsource polygons in the same window within a Philippines
   bounding box.
3. Use each Groundsource polygon's representative point to assign a GHSL
   urban class (rural / peri-urban / urban).
4. Mark a Groundsource polygon as "seen by PhilSA" if it intersects any
   PhilSA polygon during the matched window.
5. Summarize overlap rates by urban class using both polygon counts and
   Groundsource area weights.

Outputs
-------
  output/philippines_groundsource_philsa_national_urban_overlap_01_summary.png
  output/philippines_groundsource_philsa_national_urban_overlap_01_summary.pdf
  output/philippines_groundsource_philsa_national_urban_overlap_summary.csv
"""

import os
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "urban_bias"
PHILSA_DIR = ROOT / "data" / "philsa_satellite_flood"
GS_PARQUET = ROOT / "data" / "google_gemini_flood" / "groundsource_2026.parquet"
GHSL_DIR = ROOT / "data" / "ghsl"

OUT_DIR.mkdir(exist_ok=True)
GHSL_DIR.mkdir(exist_ok=True)

PERIOD_START = pd.Timestamp("2022-08-05")
PERIOD_END = pd.Timestamp("2026-02-09")
PH_BBOX = (116.0, 4.0, 127.0, 22.0)
GHSL_GLOBAL_X0 = -18041000.0
GHSL_GLOBAL_Y0 = 9009000.0
GHSL_TILE_M = 1_000_000.0

GHSL_TILES = [(7, 30), (8, 30), (8, 31), (9, 30), (9, 31)]

COLORS = {"rural": "#2E7D32", "peri-urban": "#F9A825", "urban": "#6A1B9A"}


def _load_all_philsa():
    pattern = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    parts = []
    for fname in sorted(os.listdir(PHILSA_DIR)):
        match = pattern.match(fname)
        if not match:
            continue
        date_str, _, sensor, _ = match.groups()
        event_date = pd.to_datetime(date_str, format="%Y%m%d")
        if not (PERIOD_START <= event_date <= PERIOD_END):
            continue
        fpath = PHILSA_DIR / fname
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
        event["event_key"] = os.path.splitext(fname)[0]
        parts.append(event)
    all_p = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_p, geometry="geometry", crs=4326)


def _load_groundsource_ph():
    gdf = gpd.read_parquet(GS_PARQUET, columns=["start_date", "end_date", "area_km2", "geometry"])
    gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    gdf["end_date"] = pd.to_datetime(gdf["end_date"])
    gdf = gdf[(gdf["start_date"] <= PERIOD_END) & (gdf["end_date"] >= PERIOD_START)].copy()
    gdf = gdf.cx[PH_BBOX[0]:PH_BBOX[2], PH_BBOX[1]:PH_BBOX[3]].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()
    return gdf


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


def _ensure_ghsl_tiles():
    tile_paths = {}
    for row, col in GHSL_TILES:
        tile_paths[(row, col)] = _get_ghsl_tile(row, col)
    return tile_paths


def _sample_urban_classes(points_wgs, tile_paths):
    points_moll = points_wgs.to_crs("ESRI:54009")
    result = np.array(["unknown"] * len(points_wgs), dtype=object)

    tile_cache = {}
    for key, path in tile_paths.items():
        img = Image.open(path)
        tile_cache[key] = {
            "arr": np.array(img),
            "tag": img.tag_v2,
        }

    for idx, geom in enumerate(points_moll.geometry):
        col = int((geom.x - GHSL_GLOBAL_X0) // GHSL_TILE_M) + 1
        row = int((GHSL_GLOBAL_Y0 - geom.y) // GHSL_TILE_M) + 1
        key = (row, col)
        if key not in tile_cache:
            continue
        arr = tile_cache[key]["arr"]
        tag = tile_cache[key]["tag"]
        px_scale = tag.get(33550, (1000.0, 1000.0, 0))
        tiepoint = tag.get(33922, (0, 0, 0, 0.0, 0.0, 0.0))
        x0 = float(tiepoint[3])
        y0 = float(tiepoint[4])
        dx = float(px_scale[0])
        dy = float(px_scale[1])
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


print("=" * 72, flush=True)
print("Philippines national PhilSA vs Groundsource urban overlap", flush=True)
print("=" * 72, flush=True)
print(f"Matched window: {PERIOD_START.date()} -> {PERIOD_END.date()}", flush=True)

print("\n[1/4] Loading data...", flush=True)
philsa = _load_all_philsa()
philsa = philsa.cx[PH_BBOX[0]:PH_BBOX[2], PH_BBOX[1]:PH_BBOX[3]].copy()
gs = _load_groundsource_ph()
print(f"  PhilSA polygons: {len(philsa):,}", flush=True)
print(f"  Groundsource polygons: {len(gs):,}", flush=True)

print("\n[2/4] GHSL urban classes...", flush=True)
tile_paths = _ensure_ghsl_tiles()
gs_points = gs.representative_point()
gs_points = gpd.GeoDataFrame(gs.drop(columns="geometry"), geometry=gs_points, crs=4326)
gs["urban_cls"] = _sample_urban_classes(gs_points, tile_paths)
gs = gs[gs["urban_cls"].isin(["rural", "peri-urban", "urban"])].copy()
print(gs["urban_cls"].value_counts().to_string(), flush=True)

print("\n[3/4] PhilSA overlap...", flush=True)
joined = gpd.sjoin(gs[["geometry"]], philsa[["geometry"]], how="left", predicate="intersects")
overlap_idx = set(joined.dropna(subset=["index_right"]).index)
gs["philsa_overlap"] = gs.index.isin(overlap_idx)
gs["area_km2"] = pd.to_numeric(gs["area_km2"], errors="coerce").fillna(0.0)

summary_rows = []
for urban in ["rural", "peri-urban", "urban"]:
    sub = gs[gs["urban_cls"] == urban].copy()
    poly_overlap = float(sub["philsa_overlap"].mean()) if len(sub) else np.nan
    area_total = float(sub["area_km2"].sum())
    area_overlap = float(sub.loc[sub["philsa_overlap"], "area_km2"].sum())
    area_weighted_overlap = area_overlap / area_total if area_total > 0 else np.nan
    summary_rows.append(
        {
            "urban_cls": urban,
            "n_groundsource_polygons": int(len(sub)),
            "groundsource_area_km2": area_total,
            "philsa_overlap_polygon_share": poly_overlap,
            "philsa_overlap_area_weighted_share": area_weighted_overlap,
        }
    )
    print(
        f"  {urban}: polygon overlap={poly_overlap:.3f} | area-weighted overlap={area_weighted_overlap:.3f}",
        flush=True,
    )

summary = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "philippines_groundsource_philsa_national_urban_overlap_summary.csv"
summary.to_csv(summary_csv, index=False)

print("\n[4/4] Rendering figure...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
fig.patch.set_facecolor("white")
fig.suptitle(
    "National Philippines check:\nPhilSA overlap with Groundsource floods by urban class",
    fontsize=12,
    fontweight="bold",
    y=0.975,
)
fig.text(
    0.5,
    0.875,
    "Groundsource polygons in the matched window are the reference set; urban class is sampled from GHSL at polygon representative points.",
    ha="center",
    fontsize=8,
    color="#444444",
)

order = ["rural", "peri-urban", "urban"]
ylabels = [
    f"Rural  (n={int(summary.loc[summary['urban_cls']=='rural','n_groundsource_polygons'].iloc[0]):,})",
    f"Peri-urban  (n={int(summary.loc[summary['urban_cls']=='peri-urban','n_groundsource_polygons'].iloc[0]):,})",
    f"Urban  (n={int(summary.loc[summary['urban_cls']=='urban','n_groundsource_polygons'].iloc[0]):,})",
]

for ax, col, title in [
    (axes[0], "philsa_overlap_polygon_share", "A. Polygon overlap share"),
    (axes[1], "philsa_overlap_area_weighted_share", "B. Area-weighted overlap share"),
]:
    vals = summary.set_index("urban_cls").loc[order, col].values
    colors = [COLORS[u] for u in order]
    ax.barh(order, vals, color=colors, alpha=0.92)
    for i, val in enumerate(vals):
        ax.text(val + 0.01, i, f"{val * 100:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, min(1.0, max(vals) + 0.12))
    ax.set_xlabel("Share overlapped by PhilSA")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

axes[0].set_yticks(order, ylabels)
axes[1].set_yticks(order, ylabels)

fig.tight_layout(rect=[0, 0, 1, 0.82])
fig_png = OUT_DIR / "philippines_groundsource_philsa_national_urban_overlap_01_summary.png"
fig_pdf = OUT_DIR / "philippines_groundsource_philsa_national_urban_overlap_01_summary.pdf"
fig.savefig(fig_png, dpi=180, bbox_inches="tight")
fig.savefig(fig_pdf, bbox_inches="tight")
plt.close(fig)

print(f"  Saved -> {fig_png}", flush=True)
print(f"  Saved -> {fig_pdf}", flush=True)
print(f"  Saved -> {summary_csv}", flush=True)
