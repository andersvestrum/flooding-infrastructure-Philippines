"""
noah_groundsource_5yr_intervals.py
===================================
Compare NOAH 5-year flood hazard to Google Groundsource observed floods using
strict 5-year observation windows.

The figures are intentionally lighter than city_temporal_comparison.py: they
avoid repeated detailed coastline rendering and use centroid-binned
Groundsource density for fast, readable multi-panel maps.

Outputs (all in output/):
  noah_groundsource_5yr_01_windows.png
      NOAH 5-year hazard plus Groundsource flood density in each 5-year window.
  noah_groundsource_5yr_02_agreement.png
      Agreement maps by 5-year window.
  noah_groundsource_5yr_03_metrics.png
      Time-series metrics comparing Groundsource windows to NOAH.
  noah_groundsource_5yr_04_cumulative_endpoints.png
      Running cumulative Groundsource footprint at 5-year endpoints.
  noah_groundsource_5yr_metrics.csv
      Per-city, per-window metrics used in the plots.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "noah_validation", "groundsource")
PARQUET = os.path.join(ROOT, "data", "google_gemini_flood", "groundsource_2026.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

UTM = 32651

# Strict five-year calendar windows. The 2026 records are excluded from the
# matched-window figures because they would make the final bin a partial sixth
# year rather than a 5-year comparison window.
PERIODS = [
    (2001, 2005, "2001-05"),
    (2006, 2010, "2006-10"),
    (2011, 2015, "2011-15"),
    (2016, 2020, "2016-20"),
    (2021, 2025, "2021-25"),
]

CITIES = [
    {"name": "Tuguegarao", "slug": "tuguegarao", "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan", "density": 876, "region": "Cagayan Valley"},
    {"name": "Dagupan", "slug": "dagupan", "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan", "density": 2_191, "region": "Ilocos"},
    {"name": "Manila", "slug": "manila", "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "density": 19_751, "region": "NCR"},
    {"name": "Cagayan de Oro", "slug": "cagayan_de_oro", "lat": 8.4772, "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental", "density": 2_243, "region": "Mindanao"},
    {"name": "Cotabato", "slug": "cotabato", "lat": 7.2236, "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao", "density": 1_790, "region": "BARMM"},
]

NOAH_COLORS = {1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
AGREE_COLORS = {
    "both": "#1B5E20",
    "groundsource_only": "#F57F17",
    "noah_only": "#3949AB",
    "neither": "#F0EDE8",
}
CITY_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]
CMAP_DENSITY = plt.cm.inferno
LAND_BG = "#F0EDE8"


def _find_noah_shp(province, period="5yr"):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(ROOT, "data", "noah", period, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


def _build_buffer(city):
    centre_utm = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM)
        .iloc[0]
    )
    buf_utm = centre_utm.buffer(city["radius_m"])
    buf_wgs = gpd.GeoSeries([buf_utm], crs=UTM).to_crs(epsg=4326).iloc[0]
    pad = city["radius_m"] / 111_000 * 1.25
    extent = [
        city["lng"] - pad,
        city["lng"] + pad,
        city["lat"] - pad,
        city["lat"] + pad,
    ]
    return buf_utm, buf_wgs, extent


def _simplify_wgs(gdf, tolerance_m=35):
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    geom_utm = out.to_crs(epsg=UTM).geometry.simplify(
        tolerance_m, preserve_topology=True
    )
    out["geometry"] = gpd.GeoSeries(geom_utm, crs=UTM).to_crs(epsg=4326).values
    return out[~out.is_empty].copy()


def _union(gdf_utm):
    if gdf_utm is None or len(gdf_utm) == 0:
        return None
    geom = unary_union(gdf_utm.geometry)
    return None if geom.is_empty else geom


def _agreement(noah_union, gs_union):
    if noah_union is not None and gs_union is not None:
        both = noah_union.intersection(gs_union)
        noah_only = noah_union.difference(gs_union)
        gs_only = gs_union.difference(noah_union)
        intersect_km2 = both.area / 1e6
        union_km2 = noah_union.union(gs_union).area / 1e6
        noah_area_km2 = noah_union.area / 1e6
        gs_area_km2 = gs_union.area / 1e6
        jaccard = intersect_km2 / union_km2 if union_km2 > 0 else 0
        recall = intersect_km2 / noah_area_km2 if noah_area_km2 > 0 else 0
        precision = intersect_km2 / gs_area_km2 if gs_area_km2 > 0 else 0
    elif noah_union is not None:
        both, noah_only, gs_only = None, noah_union, None
        intersect_km2 = jaccard = recall = precision = 0
    elif gs_union is not None:
        both, noah_only, gs_only = None, None, gs_union
        intersect_km2 = jaccard = recall = precision = 0
    else:
        both = noah_only = gs_only = None
        intersect_km2 = jaccard = recall = precision = 0

    return {
        "both": both,
        "noah_only": noah_only,
        "groundsource_only": gs_only,
        "intersect_km2": intersect_km2,
        "jaccard": jaccard,
        "recall": recall,
        "precision": precision,
    }


def _centroid_density(gdf_wgs, extent, city, n=84, sigma=1.15):
    lon_min, lon_max, lat_min, lat_max = extent
    grid = np.zeros((n, n), dtype=float)
    if gdf_wgs is None or gdf_wgs.empty:
        return grid, [lon_min, lon_max, lat_min, lat_max]

    cent_utm = gdf_wgs.to_crs(epsg=UTM).geometry.centroid
    cent_wgs = gpd.GeoSeries(cent_utm, crs=UTM).to_crs(epsg=4326)
    lons = cent_wgs.x.values
    lats = cent_wgs.y.values

    valid = (
        (lons >= lon_min) & (lons <= lon_max) &
        (lats >= lat_min) & (lats <= lat_max)
    )
    if valid.any():
        hist, _, _ = np.histogram2d(
            lats[valid],
            lons[valid],
            bins=n,
            range=[[lat_min, lat_max], [lon_min, lon_max]],
        )
        grid = gaussian_filter(hist, sigma=sigma)

    lon_lin = np.linspace(lon_min, lon_max, n)
    lat_lin = np.linspace(lat_min, lat_max, n)
    lon_m, lat_m = np.meshgrid(lon_lin, lat_lin)
    dx = (lon_m - city["lng"]) * 111_000 * np.cos(np.radians(city["lat"]))
    dy = (lat_m - city["lat"]) * 111_000
    outside = np.sqrt(dx ** 2 + dy ** 2) > city["radius_m"]
    grid[outside] = 0
    return grid, [lon_min, lon_max, lat_min, lat_max]


def _plot_utm_geom(ax, geom, color, alpha=0.72, zorder=3, tolerance_m=40):
    if geom is None or geom.is_empty:
        return
    simp = geom.simplify(tolerance_m, preserve_topology=True)
    gdf = gpd.GeoDataFrame(geometry=[simp], crs=UTM).to_crs(epsg=4326)
    gdf = gdf[~gdf.is_empty]
    if not gdf.empty:
        gdf.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=zorder)


def _draw_buffer(ax, buf_wgs):
    gpd.GeoDataFrame(geometry=[buf_wgs], crs=4326).boundary.plot(
        ax=ax, color="#CC0000", linewidth=0.8, linestyle="--", alpha=0.85, zorder=7
    )


def _style_map(ax, extent):
    ax.set_facecolor(LAND_BG)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.tick_params(labelsize=5.8, pad=1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.grid(color="white", linewidth=0.3, alpha=0.55, zorder=0)


def _label_cell(ax, city, noah_area):
    ax.axis("off")
    txt = (
        f"{city['name']}\n"
        f"{city['region']}\n"
        f"r={city['radius_m'] // 1000} km\n"
        f"rho={city['density']:,}/km2\n"
        f"NOAH={noah_area:.0f} km2"
    )
    ax.text(
        0.5,
        0.5,
        txt,
        ha="center",
        va="center",
        fontsize=8.2,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92, boxstyle="round,pad=0.35"),
    )


def _metric_list(city_records, slug, key, mode="period"):
    return [city_records[slug][mode][label][key] for _, _, label in PERIODS]


print("=" * 72, flush=True)
print("NOAH 5-year hazard vs Google Groundsource in strict 5-year windows", flush=True)
print("=" * 72, flush=True)

print("[1/4] Loading Groundsource parquet...", flush=True)
raw = gpd.read_parquet(PARQUET, columns=["start_date", "end_date", "geometry"])
phil = raw.cx[116:127, 4:22].copy()
phil["start_date"] = pd.to_datetime(phil["start_date"])
phil["end_date"] = pd.to_datetime(phil["end_date"])
phil["year"] = phil["start_date"].dt.year

excluded_2026 = int((phil["year"] == 2026).sum())
phil_5yr = phil[(phil["year"] >= PERIODS[0][0]) & (phil["year"] <= PERIODS[-1][1])].copy()
phil_utm = phil_5yr.to_crs(epsg=UTM)
print(
    f"  Philippines events: {len(phil):,} total, {len(phil_5yr):,} used in 2001-2025; "
    f"{excluded_2026:,} partial-2026 records excluded",
    flush=True,
)

print("[2/4] Extracting city windows and overlap metrics...", flush=True)
city_records = {}
rows = []

for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    slug = city["slug"]
    buf_utm, buf_wgs, extent = _build_buffer(city)

    noah_shp = _find_noah_shp(city["noah_province"])
    if noah_shp is None:
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=4326)
    else:
        nr = gpd.read_file(noah_shp)
        if nr.crs is None or nr.crs.to_epsg() != 4326:
            nr = nr.to_crs(epsg=4326)
        nr["Var"] = pd.to_numeric(nr["Var"], errors="coerce").fillna(0).astype(int)
        noah = gpd.clip(nr[["Var", "geometry"]], buf_wgs)
        noah = noah[~noah.is_empty].copy()

    noah_utm = noah.to_crs(epsg=UTM) if not noah.empty else noah
    noah_union = _union(noah_utm)
    noah_area = noah_union.area / 1e6 if noah_union is not None else 0
    noah_plot = _simplify_wgs(noah, tolerance_m=35)

    sub_utm = phil_utm[phil_utm.intersects(buf_utm)].copy()
    sub_wgs = sub_utm.to_crs(epsg=4326)
    gs_all = gpd.clip(sub_wgs, buf_wgs)
    gs_all = gs_all[~gs_all.is_empty].copy()

    period_records = {}
    cumulative_records = {}

    for start, end, label in PERIODS:
        gs_period = gs_all[(gs_all["year"] >= start) & (gs_all["year"] <= end)].copy()
        gs_cum = gs_all[gs_all["year"] <= end].copy()

        for mode, gs in [("period", gs_period), ("cumulative", gs_cum)]:
            gs_utm = gs.to_crs(epsg=UTM) if not gs.empty else gs
            gs_union = _union(gs_utm)
            gs_area = gs_union.area / 1e6 if gs_union is not None else 0
            agree = _agreement(noah_union, gs_union)
            density, grid_extent = _centroid_density(gs, extent, city)
            rec = {
                "groundsource": gs,
                "groundsource_union": gs_union,
                "groundsource_area_km2": gs_area,
                "density": density,
                "grid_extent": grid_extent,
                "n_events": int(len(gs)),
                **agree,
            }
            if mode == "period":
                period_records[label] = rec
            else:
                cumulative_records[label] = rec

        period_row = period_records[label]
        cumulative_row = cumulative_records[label]
        rows.append({
            "city": city["name"],
            "slug": slug,
            "period": label,
            "start_year": start,
            "end_year": end,
            "noah_area_km2": noah_area,
            "period_events": period_row["n_events"],
            "period_groundsource_area_km2": period_row["groundsource_area_km2"],
            "period_intersection_km2": period_row["intersect_km2"],
            "period_jaccard": period_row["jaccard"],
            "period_recall": period_row["recall"],
            "period_precision": period_row["precision"],
            "cumulative_events": cumulative_row["n_events"],
            "cumulative_groundsource_area_km2": cumulative_row["groundsource_area_km2"],
            "cumulative_intersection_km2": cumulative_row["intersect_km2"],
            "cumulative_jaccard": cumulative_row["jaccard"],
            "cumulative_recall": cumulative_row["recall"],
            "cumulative_precision": cumulative_row["precision"],
        })

        print(
            f"    {label}: n={period_row['n_events']:>4}, "
            f"J={period_row['jaccard']:.3f}, R={period_row['recall']:.3f}; "
            f"cum J={cumulative_row['jaccard']:.3f}, cum R={cumulative_row['recall']:.3f}",
            flush=True,
        )

    city_records[slug] = {
        "city": city,
        "buf_wgs": buf_wgs,
        "extent": extent,
        "noah": noah,
        "noah_plot": noah_plot,
        "noah_union": noah_union,
        "noah_area": noah_area,
        "period": period_records,
        "cumulative": cumulative_records,
    }

metrics = pd.DataFrame(rows)
metrics_path = os.path.join(OUT_DIR, "noah_groundsource_5yr_metrics.csv")
metrics.to_csv(metrics_path, index=False)
print(f"  Saved metrics -> {metrics_path}", flush=True)

print("[3/4] Rendering figures...", flush=True)

period_labels = [label for _, _, label in PERIODS]
N_CITIES = len(CITIES)
N_PERIODS = len(PERIODS)

fig_note = (
    "Strict 5-year Groundsource windows; 2026 partial records excluded from maps "
    f"(Philippines 2026 records excluded: {excluded_2026:,})."
)

# Figure 1: NOAH panel + period density panels.
fig1, axes1 = plt.subplots(
    N_CITIES,
    N_PERIODS + 2,
    figsize=(3.25 * (N_PERIODS + 2), 3.35 * N_CITIES),
    gridspec_kw={"width_ratios": [0.70, 1.0] + [1.0] * N_PERIODS},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    "NOAH 5-year Flood Hazard vs Google Groundsource Flood Density by 5-year Window",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig1.text(0.5, 0.965, fig_note, ha="center", fontsize=9, color="#444444")

headers = ["", "NOAH 5yr"] + period_labels
for ci, header in enumerate(headers):
    axes1[0, ci].set_title(header, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    cr = city_records[slug]
    _label_cell(axes1[ri, 0], city, cr["noah_area"])

    ax = axes1[ri, 1]
    _style_map(ax, cr["extent"])
    noah_plot = cr["noah_plot"]
    for lev, color in NOAH_COLORS.items():
        sub = noah_plot[noah_plot["Var"] == lev] if noah_plot is not None and not noah_plot.empty else noah_plot
        if sub is not None and not sub.empty:
            sub.plot(ax=ax, color=color, alpha=0.74, edgecolor="none", zorder=3)
    _draw_buffer(ax, cr["buf_wgs"])
    ax.plot(city["lng"], city["lat"], "r*", markersize=5, zorder=9)
    ax.text(
        0.02,
        0.98,
        f"{cr['noah_area']:.1f} km2",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.15"),
    )

    all_density = np.concatenate([
        cr["period"][label]["density"].ravel()
        for label in period_labels
    ])
    positive = all_density[all_density > 0]
    vmax = max(np.percentile(positive, 97), 1) if len(positive) else 1

    for pi, label in enumerate(period_labels):
        ax = axes1[ri, pi + 2]
        _style_map(ax, cr["extent"])
        density = cr["period"][label]["density"]
        if density.max() > 0:
            masked = np.ma.masked_where(density <= 0, density)
            ax.imshow(
                masked,
                extent=cr["period"][label]["grid_extent"],
                origin="lower",
                cmap=CMAP_DENSITY,
                vmin=0,
                vmax=vmax,
                aspect="auto",
                interpolation="bilinear",
                alpha=0.92,
                zorder=2,
            )
        for lev, color in NOAH_COLORS.items():
            sub = noah_plot[noah_plot["Var"] == lev] if noah_plot is not None and not noah_plot.empty else noah_plot
            if sub is not None and not sub.empty:
                sub.boundary.plot(ax=ax, edgecolor=color, linewidth=0.7, alpha=0.90, zorder=4)
        _draw_buffer(ax, cr["buf_wgs"])
        ax.plot(city["lng"], city["lat"], "r*", markersize=5, zorder=9)
        r = cr["period"][label]
        ax.text(
            0.02,
            0.98,
            f"n={r['n_events']:,}\nJ={r['jaccard']:.3f}\nR={r['recall']:.3f}",
            transform=ax.transAxes,
            fontsize=6.2,
            va="top",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.86, boxstyle="round,pad=0.14"),
        )

noah_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="NOAH low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="NOAH medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="NOAH high"),
]
fig1.legend(handles=noah_handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
sm = ScalarMappable(cmap=CMAP_DENSITY, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cb_ax = fig1.add_axes([0.36, 0.018, 0.30, 0.009])
cb = fig1.colorbar(sm, cax=cb_ax, orientation="horizontal")
cb.set_label("Groundsource event density, scaled per city", fontsize=8)
fig1.tight_layout(rect=[0.01, 0.04, 0.99, 0.945])
out1 = os.path.join(OUT_DIR, "noah_groundsource_5yr_01_windows.png")
fig1.savefig(out1, dpi=170, facecolor=fig1.get_facecolor(), bbox_inches="tight")
plt.close(fig1)
print(f"  Saved -> {out1}", flush=True)

# Figure 2: agreement maps for each 5-year period.
fig2, axes2 = plt.subplots(
    N_CITIES,
    N_PERIODS + 1,
    figsize=(3.25 * (N_PERIODS + 1), 3.35 * N_CITIES),
    gridspec_kw={"width_ratios": [0.70] + [1.0] * N_PERIODS},
)
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle(
    "Agreement by 5-year Window: NOAH Hazard vs Google Groundsource",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig2.text(0.5, 0.965, "Green=both, blue=NOAH only, orange=Groundsource only. " + fig_note,
          ha="center", fontsize=9, color="#444444")

for ci, header in enumerate([""] + period_labels):
    axes2[0, ci].set_title(header, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    cr = city_records[slug]
    _label_cell(axes2[ri, 0], city, cr["noah_area"])
    for pi, label in enumerate(period_labels):
        ax = axes2[ri, pi + 1]
        _style_map(ax, cr["extent"])
        r = cr["period"][label]
        _plot_utm_geom(ax, r["noah_only"], AGREE_COLORS["noah_only"], alpha=0.72, zorder=2)
        _plot_utm_geom(ax, r["groundsource_only"], AGREE_COLORS["groundsource_only"], alpha=0.62, zorder=3)
        _plot_utm_geom(ax, r["both"], AGREE_COLORS["both"], alpha=0.82, zorder=4)
        _draw_buffer(ax, cr["buf_wgs"])
        ax.plot(city["lng"], city["lat"], "r*", markersize=5, zorder=9)
        ax.text(
            0.02,
            0.98,
            f"J={r['jaccard']:.3f}\nR={r['recall']:.3f}\nP={r['precision']:.3f}",
            transform=ax.transAxes,
            fontsize=6.2,
            va="top",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.86, boxstyle="round,pad=0.14"),
        )

agree_handles = [
    mpatches.Patch(color=AGREE_COLORS["both"], label="Both"),
    mpatches.Patch(color=AGREE_COLORS["groundsource_only"], label="Groundsource only"),
    mpatches.Patch(color=AGREE_COLORS["noah_only"], label="NOAH only"),
]
fig2.legend(handles=agree_handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig2.tight_layout(rect=[0.01, 0.035, 0.99, 0.945])
out2 = os.path.join(OUT_DIR, "noah_groundsource_5yr_02_agreement.png")
fig2.savefig(out2, dpi=170, facecolor=fig2.get_facecolor(), bbox_inches="tight")
plt.close(fig2)
print(f"  Saved -> {out2}", flush=True)

# Figure 3: temporal metrics.
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10.5))
fig3.suptitle(
    "Temporal Metrics: Groundsource 5-year Windows vs NOAH 5-year Hazard",
    fontsize=15,
    fontweight="bold",
)
x = np.arange(N_PERIODS)
w = 0.15

metric_specs = [
    (axes3[0, 0], "jaccard", "A. Jaccard similarity", "Jaccard", 1.0, "o"),
    (axes3[0, 1], "recall", "B. Recall of NOAH hazard", "Recall", 1.05, "s"),
    (axes3[0, 2], "precision", "C. Precision vs NOAH hazard", "Precision", 1.05, "^"),
]
for ax, key, title, ylabel, ymax, marker in metric_specs:
    for ci, city in enumerate(CITIES):
        vals = _metric_list(city_records, city["slug"], key, mode="period")
        ax.plot(x, vals, marker=marker, color=CITY_COLORS[ci], linewidth=2, markersize=5, label=city["name"])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=25, ha="right")
    ax.grid(alpha=0.3)

ax = axes3[1, 0]
for ci, city in enumerate(CITIES):
    vals = _metric_list(city_records, city["slug"], "n_events", mode="period")
    ax.bar(x + (ci - 2) * w, vals, w, color=CITY_COLORS[ci], alpha=0.85, edgecolor="white", linewidth=0.3, label=city["name"])
ax.set_title("D. Groundsource events per 5-year window", fontweight="bold")
ax.set_ylabel("Events")
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=25, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{int(val):,}"))
ax.grid(axis="y", alpha=0.3)

ax = axes3[1, 1]
for ci, city in enumerate(CITIES):
    slug = city["slug"]
    vals = _metric_list(city_records, slug, "groundsource_area_km2", mode="period")
    ax.plot(x, vals, marker="o", color=CITY_COLORS[ci], linewidth=2, markersize=5, label=city["name"])
    ax.axhline(city_records[slug]["noah_area"], color=CITY_COLORS[ci], linewidth=1.0, linestyle=":", alpha=0.7)
ax.set_title("E. Window Groundsource area vs NOAH area", fontweight="bold")
ax.set_ylabel("Area (km2); dotted = NOAH")
ax.set_xticks(x)
ax.set_xticklabels(period_labels, rotation=25, ha="right")
ax.grid(alpha=0.3)

ax = axes3[1, 2]
for ci, city in enumerate(CITIES):
    vals = _metric_list(city_records, city["slug"], "recall", mode="cumulative")
    ax.plot(x, vals, marker="D", color=CITY_COLORS[ci], linewidth=2, markersize=5, label=city["name"])
ax.set_title("F. Cumulative recall at 5-year endpoints", fontweight="bold")
ax.set_ylabel("Cumulative recall")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels([f"to {end}" for _, end, _ in PERIODS], rotation=25, ha="right")
ax.grid(alpha=0.3)

handles, labels = axes3[0, 0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig3.tight_layout(rect=[0, 0.04, 1, 0.94])
out3 = os.path.join(OUT_DIR, "noah_groundsource_5yr_03_metrics.png")
fig3.savefig(out3, dpi=170, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved -> {out3}", flush=True)

# Figure 4: cumulative endpoint maps.
fig4, axes4 = plt.subplots(
    N_CITIES,
    N_PERIODS + 1,
    figsize=(3.25 * (N_PERIODS + 1), 3.35 * N_CITIES),
    gridspec_kw={"width_ratios": [0.70] + [1.0] * N_PERIODS},
)
fig4.patch.set_facecolor("#F7F7F7")
fig4.suptitle(
    "Cumulative Google Groundsource Floods at 5-year Endpoints vs NOAH 5-year Hazard",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig4.text(0.5, 0.965, fig_note, ha="center", fontsize=9, color="#444444")

for ci, header in enumerate([""] + [f"to {end}" for _, end, _ in PERIODS]):
    axes4[0, ci].set_title(header, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    cr = city_records[slug]
    _label_cell(axes4[ri, 0], city, cr["noah_area"])
    all_density = np.concatenate([
        cr["cumulative"][label]["density"].ravel()
        for label in period_labels
    ])
    positive = all_density[all_density > 0]
    vmax = max(np.percentile(positive, 97), 1) if len(positive) else 1
    noah_plot = cr["noah_plot"]

    for pi, label in enumerate(period_labels):
        ax = axes4[ri, pi + 1]
        _style_map(ax, cr["extent"])
        r = cr["cumulative"][label]
        density = r["density"]
        if density.max() > 0:
            masked = np.ma.masked_where(density <= 0, density)
            ax.imshow(
                masked,
                extent=r["grid_extent"],
                origin="lower",
                cmap=CMAP_DENSITY,
                vmin=0,
                vmax=vmax,
                aspect="auto",
                interpolation="bilinear",
                alpha=0.92,
                zorder=2,
            )
        for lev, color in NOAH_COLORS.items():
            sub = noah_plot[noah_plot["Var"] == lev] if noah_plot is not None and not noah_plot.empty else noah_plot
            if sub is not None and not sub.empty:
                sub.boundary.plot(ax=ax, edgecolor=color, linewidth=0.7, alpha=0.90, zorder=4)
        _draw_buffer(ax, cr["buf_wgs"])
        ax.plot(city["lng"], city["lat"], "r*", markersize=5, zorder=9)
        ax.text(
            0.02,
            0.98,
            f"n={r['n_events']:,}\nJ={r['jaccard']:.3f}\nR={r['recall']:.3f}",
            transform=ax.transAxes,
            fontsize=6.2,
            va="top",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.86, boxstyle="round,pad=0.14"),
        )

fig4.legend(handles=noah_handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
cb_ax = fig4.add_axes([0.36, 0.018, 0.30, 0.009])
cb = fig4.colorbar(sm, cax=cb_ax, orientation="horizontal")
cb.set_label("Cumulative Groundsource event density, scaled per city", fontsize=8)
fig4.tight_layout(rect=[0.01, 0.04, 0.99, 0.945])
out4 = os.path.join(OUT_DIR, "noah_groundsource_5yr_04_cumulative_endpoints.png")
fig4.savefig(out4, dpi=170, facecolor=fig4.get_facecolor(), bbox_inches="tight")
plt.close(fig4)
print(f"  Saved -> {out4}", flush=True)

print("[4/4] Done.", flush=True)
for path in [out1, out2, out3, out4, metrics_path]:
    print(f"  {path}", flush=True)
