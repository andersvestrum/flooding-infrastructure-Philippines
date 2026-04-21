"""
noah_groundsource_grid_validation.py
=====================================
Grid-based validation of NOAH 5-year flood hazard with Google Groundsource.

Why a grid?
-----------
Direct polygon overlap is useful for maps, but Groundsource footprints can be
spatially broad. A common 500 m validation grid makes the comparison more
intuitive:
  * assign each grid cell a NOAH class: none / low / medium / high
  * count Groundsource observations touching each cell in strict 5-year windows
  * ask whether observations concentrate inside NOAH hazard cells and whether
    observation rates increase with NOAH severity

Outputs:
  output/noah_groundsource_validation_summary.csv
  output/noah_groundsource_validation_by_class.csv
  output/noah_groundsource_validation_01_dashboard.png
  output/noah_groundsource_validation_02_hazard_gradient.png
  output/noah_groundsource_validation_03_grid_agreement_2021_25.png
"""

import glob
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "noah_validation", "groundsource")
PARQUET = os.path.join(ROOT, "data", "google_gemini_flood", "groundsource_2026.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

UTM = 32651
GRID_M = 500
CELL_AREA_KM2 = (GRID_M * GRID_M) / 1e6

PERIODS = [
    (2001, 2005, "2001-05"),
    (2006, 2010, "2006-10"),
    (2011, 2015, "2011-15"),
    (2016, 2020, "2016-20"),
    (2021, 2025, "2021-25"),
]

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

CLASS_LABELS = {0: "No hazard", 1: "Low", 2: "Medium", 3: "High"}
CLASS_COLORS = {0: "#D9D9D9", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
CITY_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]
AGREE_COLORS = {
    "both": "#1B5E20",
    "groundsource_only": "#F57F17",
    "noah_only": "#3949AB",
    "neither": "#D7D7D7",
}


def _find_noah_shp(province, period="5yr"):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(ROOT, "data", "noah", period, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


def _build_city_grid(city):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM)
        .iloc[0]
    )
    buf = centre.buffer(city["radius_m"])
    xs = np.arange(centre.x - city["radius_m"], centre.x + city["radius_m"] + GRID_M, GRID_M)
    ys = np.arange(centre.y - city["radius_m"], centre.y + city["radius_m"] + GRID_M, GRID_M)
    pts = []
    cell_ids = []
    idx = 0
    for y in ys:
        for x in xs:
            pt = Point(x, y)
            if buf.contains(pt) or buf.touches(pt):
                pts.append(pt)
                cell_ids.append(idx)
                idx += 1
    grid = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=pts, crs=UTM)
    grid["area_km2"] = CELL_AREA_KM2
    return grid, buf


def _load_noah_for_city(city, buf_utm):
    shp = _find_noah_shp(city["noah_province"])
    if shp is None:
        return gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)
    noah = gpd.read_file(shp)
    if noah.crs is None:
        noah = noah.set_crs(4326)
    if noah.crs.to_epsg() != UTM:
        noah = noah.to_crs(epsg=UTM)
    noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).astype(int)
    noah = gpd.clip(noah[["Var", "geometry"]], buf_utm)
    return noah[~noah.is_empty].copy()


def _assign_noah(grid, noah):
    out = grid.copy()
    out["noah_class"] = 0
    if noah.empty:
        return out
    joined = gpd.sjoin(
        out[["cell_id", "geometry"]],
        noah[["Var", "geometry"]],
        how="left",
        predicate="intersects",
    )
    max_var = joined.groupby("cell_id")["Var"].max()
    out = out.join(max_var.rename("assigned_var"), on="cell_id")
    out["noah_class"] = out["assigned_var"].fillna(0).astype(int)
    out = out.drop(columns=["assigned_var"])
    return out


def _groundsource_counts(grid, groundsource):
    counts = pd.Series(0, index=grid["cell_id"], dtype=int)
    if groundsource.empty:
        return counts
    joined = gpd.sjoin(
        grid[["cell_id", "geometry"]],
        groundsource[["geometry"]],
        how="left",
        predicate="intersects",
    )
    hits = joined.dropna(subset=["index_right"]).groupby("cell_id").size()
    counts.loc[hits.index] = hits.astype(int)
    return counts


def _safe_rate(num, den):
    return num / den if den and den > 0 else np.nan


def _smoothed_share(observed_cells, total_cells):
    # Jeffreys-like smoothing avoids infinite lift when the background has zero
    # observed cells, while preserving the interpretation of a cell share.
    return (observed_cells + 0.5) / (total_cells + 1.0) if total_cells > 0 else np.nan


def _summarize_period(city, label, grid, count_col, mode):
    rows_by_class = []
    for klass in [0, 1, 2, 3]:
        sub = grid[grid["noah_class"] == klass]
        cells = len(sub)
        area = cells * CELL_AREA_KM2
        obs_cells = int((sub[count_col] > 0).sum())
        hits = int(sub[count_col].sum())
        rows_by_class.append({
            "city": city["name"],
            "slug": city["slug"],
            "period": label,
            "mode": mode,
            "noah_class": klass,
            "noah_label": CLASS_LABELS[klass],
            "cells": cells,
            "area_km2": area,
            "observed_cells": obs_cells,
            "observed_share": _safe_rate(obs_cells, cells),
            "groundsource_cell_hits": hits,
            "hits_per_km2": _safe_rate(hits, area),
        })

    cls = pd.DataFrame(rows_by_class)
    hazard = grid[grid["noah_class"] > 0]
    nonhaz = grid[grid["noah_class"] == 0]
    observed = grid[grid[count_col] > 0]
    hazard_obs = hazard[hazard[count_col] > 0]
    nonhaz_obs = nonhaz[nonhaz[count_col] > 0]
    hazard_cells = len(hazard)
    nonhaz_cells = len(nonhaz)
    observed_cells = len(observed)
    hazard_hits = int(hazard[count_col].sum())
    nonhaz_hits = int(nonhaz[count_col].sum())
    total_hits = int(grid[count_col].sum())

    hazard_share = _safe_rate(len(hazard_obs), hazard_cells)
    nonhaz_share = _safe_rate(len(nonhaz_obs), nonhaz_cells)
    if observed_cells > 0:
        hazard_share_s = _smoothed_share(len(hazard_obs), hazard_cells)
        nonhaz_share_s = _smoothed_share(len(nonhaz_obs), nonhaz_cells)
        observed_lift = hazard_share_s / nonhaz_share_s if nonhaz_share_s and nonhaz_share_s > 0 else np.nan
    else:
        observed_lift = np.nan

    hazard_area = hazard_cells * CELL_AREA_KM2
    nonhaz_area = nonhaz_cells * CELL_AREA_KM2
    hazard_hit_density = _safe_rate(hazard_hits, hazard_area)
    nonhaz_hit_density = _safe_rate(nonhaz_hits, nonhaz_area)
    if total_hits > 0:
        hit_density_lift = (
            hazard_hit_density / nonhaz_hit_density
            if nonhaz_hit_density and nonhaz_hit_density > 0
            else np.nan
        )
    else:
        hit_density_lift = np.nan

    summary = {
        "city": city["name"],
        "slug": city["slug"],
        "period": label,
        "mode": mode,
        "grid_m": GRID_M,
        "total_cells": len(grid),
        "hazard_cells": hazard_cells,
        "nonhazard_cells": nonhaz_cells,
        "observed_cells": observed_cells,
        "total_groundsource_cell_hits": total_hits,
        "grid_recall": _safe_rate(len(hazard_obs), hazard_cells),
        "grid_precision": _safe_rate(len(observed[observed["noah_class"] > 0]), observed_cells),
        "hit_precision": _safe_rate(hazard_hits, total_hits),
        "hazard_observed_share": hazard_share,
        "nonhazard_observed_share": nonhaz_share,
        "observed_share_lift": observed_lift,
        "hazard_hits_per_km2": hazard_hit_density,
        "nonhazard_hits_per_km2": nonhaz_hit_density,
        "hit_density_lift": hit_density_lift,
    }
    return summary, rows_by_class


print("=" * 72, flush=True)
print("Grid validation: NOAH 5-year hazard vs Google Groundsource", flush=True)
print("=" * 72, flush=True)
print(f"Grid resolution: {GRID_M} m ({CELL_AREA_KM2:.2f} km2 per sample cell)", flush=True)

print("[1/4] Loading Groundsource parquet...", flush=True)
raw = gpd.read_parquet(PARQUET, columns=["start_date", "geometry"])
phil = raw.cx[116:127, 4:22].copy()
phil["start_date"] = pd.to_datetime(phil["start_date"])
phil["year"] = phil["start_date"].dt.year
excluded_2026 = int((phil["year"] == 2026).sum())
phil = phil[(phil["year"] >= PERIODS[0][0]) & (phil["year"] <= PERIODS[-1][1])].copy()
phil_utm = phil.to_crs(epsg=UTM)
print(f"  Using {len(phil):,} Philippines records from 2001-2025; excluded 2026 partial records: {excluded_2026:,}", flush=True)

summary_rows = []
class_rows = []
map_payload = {}

print("[2/4] Building grids and metrics...", flush=True)
for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    grid, buf = _build_city_grid(city)
    noah = _load_noah_for_city(city, buf)
    grid = _assign_noah(grid, noah)

    gs_city = phil_utm[phil_utm.intersects(buf)].copy()
    gs_city = gpd.clip(gs_city, buf)
    gs_city = gs_city[~gs_city.is_empty].copy()

    cumulative_counts = pd.Series(0, index=grid["cell_id"], dtype=int)
    for start, end, label in PERIODS:
        period_gs = gs_city[(gs_city["year"] >= start) & (gs_city["year"] <= end)].copy()
        period_counts = _groundsource_counts(grid, period_gs)
        cumulative_counts = cumulative_counts + period_counts

        for mode, counts in [("period", period_counts), ("cumulative", cumulative_counts.copy())]:
            count_col = f"{mode}_{label}"
            grid[count_col] = counts.values
            summary, by_class = _summarize_period(city, label, grid, count_col, mode)
            summary_rows.append(summary)
            class_rows.extend(by_class)

        print(
            f"    {label}: events={len(period_gs):>4}, "
            f"period recall={summary_rows[-2]['grid_recall']:.3f}, "
            f"precision={summary_rows[-2]['grid_precision']:.3f}, "
            f"lift={summary_rows[-2]['observed_share_lift']:.2f}",
            flush=True,
        )

    latest = "period_2021-25"
    grid_wgs = grid.to_crs(epsg=4326)
    grid_wgs["agreement_2021_25"] = np.select(
        [
            (grid_wgs["noah_class"] > 0) & (grid_wgs[latest] > 0),
            (grid_wgs["noah_class"] > 0) & (grid_wgs[latest] == 0),
            (grid_wgs["noah_class"] == 0) & (grid_wgs[latest] > 0),
        ],
        ["both", "noah_only", "groundsource_only"],
        default="neither",
    )
    map_payload[city["slug"]] = {
        "city": city,
        "grid": grid_wgs,
        "centre": (city["lng"], city["lat"]),
    }

summary = pd.DataFrame(summary_rows)
by_class = pd.DataFrame(class_rows)
summary_path = os.path.join(OUT_DIR, "noah_groundsource_validation_summary.csv")
class_path = os.path.join(OUT_DIR, "noah_groundsource_validation_by_class.csv")
summary.to_csv(summary_path, index=False)
by_class.to_csv(class_path, index=False)
print(f"  Saved -> {summary_path}", flush=True)
print(f"  Saved -> {class_path}", flush=True)

print("[3/4] Rendering validation dashboard...", flush=True)
period_labels = [p[2] for p in PERIODS]
latest_label = "2021-25"
period_summary = summary[summary["mode"] == "period"].copy()
cum_summary = summary[summary["mode"] == "cumulative"].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
fig.suptitle(
    "NOAH Validation with Google Groundsource on a Common 500 m Grid",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.945,
    "Lift > 1 means Groundsource-observed cells are more common inside NOAH hazard zones than in nearby non-hazard cells.",
    ha="center",
    fontsize=9,
    color="#444444",
)
x = np.arange(len(period_labels))
width = 0.16

ax = axes[0, 0]
for i, city in enumerate(CITIES):
    vals = period_summary[period_summary["slug"] == city["slug"]].set_index("period").loc[period_labels, "observed_share_lift"]
    ax.plot(x, vals, marker="o", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
ax.set_title("A. Observed-cell lift: NOAH hazard vs non-hazard", fontweight="bold")
ax.set_ylabel("Lift ratio")
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=25, ha="right")
ax.grid(alpha=0.3)

ax = axes[0, 1]
for i, city in enumerate(CITIES):
    vals = period_summary[period_summary["slug"] == city["slug"]].set_index("period").loc[period_labels, "grid_recall"]
    ax.plot(x, vals, marker="s", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.set_title("B. Grid recall of NOAH hazard cells", fontweight="bold")
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=25, ha="right")
ax.grid(alpha=0.3)

ax = axes[0, 2]
for i, city in enumerate(CITIES):
    vals = period_summary[period_summary["slug"] == city["slug"]].set_index("period").loc[period_labels, "grid_precision"]
    ax.plot(x, vals, marker="^", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.set_title("C. Grid precision of Groundsource observations", fontweight="bold")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=25, ha="right")
ax.grid(alpha=0.3)

latest = period_summary[period_summary["period"] == latest_label].set_index("slug")
ax = axes[1, 0]
for i, city in enumerate(CITIES):
    vals = [
        latest.loc[city["slug"], "hazard_observed_share"],
        latest.loc[city["slug"], "nonhazard_observed_share"],
    ]
    ax.bar(np.array([0, 1]) + (i - 2) * width, vals, width, color=CITY_COLORS[i], alpha=0.85, label=city["name"])
ax.set_title("D. 2021-25 observed-cell share inside vs outside NOAH", fontweight="bold")
ax.set_ylabel("Observed share of grid cells")
ax.set_xticks([0, 1])
ax.set_xticklabels(["NOAH hazard", "Non-hazard"])
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 1]
latest_class = by_class[(by_class["mode"] == "period") & (by_class["period"] == latest_label)]
for i, city in enumerate(CITIES):
    vals = latest_class[latest_class["slug"] == city["slug"]].set_index("noah_class").loc[[0, 1, 2, 3], "hits_per_km2"]
    ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.set_title("E. 2021-25 Groundsource hit density by NOAH class", fontweight="bold")
ax.set_ylabel("Groundsource cell-hits / km2")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["None", "Low", "Med", "High"])
ax.grid(alpha=0.3)

ax = axes[1, 2]
for i, city in enumerate(CITIES):
    vals = cum_summary[cum_summary["slug"] == city["slug"]].set_index("period").loc[period_labels, "grid_recall"]
    ax.plot(x, vals, marker="D", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.set_title("F. Cumulative grid recall at 5-year endpoints", fontweight="bold")
ax.set_ylabel("Cumulative recall")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels([f"to {p[1]}" for p in PERIODS], rotation=25, ha="right")
ax.grid(alpha=0.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=[0, 0.04, 1, 0.925])
dashboard_path = os.path.join(OUT_DIR, "noah_groundsource_validation_01_dashboard.png")
fig.savefig(dashboard_path, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {dashboard_path}", flush=True)

print("[4/4] Rendering hazard gradient and grid map...", flush=True)
fig2, axes2 = plt.subplots(1, len(CITIES), figsize=(20, 4.3), sharey=False)
fig2.suptitle("Groundsource Observation Density by NOAH Hazard Class and 5-year Window", fontsize=14, fontweight="bold")
for i, city in enumerate(CITIES):
    ax = axes2[i]
    sub = by_class[(by_class["slug"] == city["slug"]) & (by_class["mode"] == "period")]
    for j, label in enumerate(period_labels):
        vals = sub[sub["period"] == label].set_index("noah_class").loc[[0, 1, 2, 3], "hits_per_km2"]
        ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=1.8, markersize=4, label=label)
    ax.set_title(city["name"], fontweight="bold", color=CITY_COLORS[i])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["None", "Low", "Med", "High"], rotation=25, ha="right")
    ax.set_ylabel("Cell-hits / km2" if i == 0 else "")
    ax.grid(alpha=0.3)
axes2[-1].legend(title="Window", fontsize=8, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig2.tight_layout(rect=[0, 0, 0.98, 0.90])
gradient_path = os.path.join(OUT_DIR, "noah_groundsource_validation_02_hazard_gradient.png")
fig2.savefig(gradient_path, dpi=170, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved -> {gradient_path}", flush=True)

fig3, axes3 = plt.subplots(1, len(CITIES), figsize=(20, 4.4))
fig3.suptitle("Cell-level Agreement in Latest Full 5-year Window (2021-2025)", fontsize=14, fontweight="bold")
for i, city in enumerate(CITIES):
    ax = axes3[i]
    payload = map_payload[city["slug"]]
    grid = payload["grid"]
    for cat in ["neither", "noah_only", "groundsource_only", "both"]:
        sub = grid[grid["agreement_2021_25"] == cat]
        if sub.empty:
            continue
        alpha = 0.20 if cat == "neither" else 0.85
        size = 5 if cat == "neither" else 9
        ax.scatter(sub.geometry.x, sub.geometry.y, s=size, c=AGREE_COLORS[cat], marker="s", alpha=alpha, linewidths=0)
    ax.plot(city["lng"], city["lat"], "r*", markersize=7)
    ax.set_title(city["name"], fontweight="bold", color=CITY_COLORS[i])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.grid(alpha=0.25)

legend_handles = [
    mpatches.Patch(color=AGREE_COLORS["both"], label="Both"),
    mpatches.Patch(color=AGREE_COLORS["groundsource_only"], label="Groundsource only"),
    mpatches.Patch(color=AGREE_COLORS["noah_only"], label="NOAH only"),
    mpatches.Patch(color=AGREE_COLORS["neither"], label="Neither"),
]
fig3.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig3.tight_layout(rect=[0, 0.06, 1, 0.90])
map_path = os.path.join(OUT_DIR, "noah_groundsource_validation_03_grid_agreement_2021_25.png")
fig3.savefig(map_path, dpi=170, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved -> {map_path}", flush=True)

for path in [summary_path, class_path, dashboard_path, gradient_path, map_path]:
    print(path, flush=True)
