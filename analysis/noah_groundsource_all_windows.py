"""
noah_groundsource_all_windows.py
=================================
Aggregate Groundsource empirical flood hazard across all valid 5-year windows
(2011-2015 through 2021-2025), masking out permanent water bodies via OSM.

Method
------
1. For each sliding 5-year window, build a smoothed Groundsource support
   surface on a 250 m grid (Gaussian sigma 750 m).
2. Normalize each window's support surface to [0, 1].
3. Average the normalized surfaces across all windows → temporal-mean hazard.
4. Mask grid cells that fall in permanent water bodies (OSM natural=water /
   waterway polygons) so coastal/river pixels are not misclassified as flood.
5. Area-match the temporal-mean surface to NOAH low/medium/high class shares
   (excluding water-masked cells).

Outputs
-------
  output/noah_groundsource_allwindows_01_maps.png
  output/noah_groundsource_allwindows_02_diagnostics.png
  output/noah_groundsource_allwindows_summary.csv
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
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
import osmnx as ox

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "noah_validation", "groundsource")
PARQUET = os.path.join(ROOT, "data", "google_gemini_flood", "groundsource_2026.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

UTM = 32651
GRID_M = 250
SIGMA_M = 750
SIGMA_CELLS = SIGMA_M / GRID_M

# All 5-year windows with sufficient Groundsource data
WINDOWS = [(y, y + 4) for y in range(2011, 2022)]  # 2011-15 … 2021-25

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

NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR = "#A8D5E2"
DIFF_COLORS = {
    "noah_much_higher": "#08306B",
    "noah_higher": "#6BAED6",
    "match": "#2E7D32",
    "gs_higher": "#FDAE61",
    "gs_much_higher": "#B2182B",
}
CITY_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]


def _find_noah_shp(province, period="5yr"):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(ROOT, "data", "noah", period, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


def _build_grid(city):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM)
        .iloc[0]
    )
    xs = np.arange(centre.x - city["radius_m"], centre.x + city["radius_m"] + GRID_M, GRID_M)
    ys = np.arange(centre.y - city["radius_m"], centre.y + city["radius_m"] + GRID_M, GRID_M)
    xx, yy = np.meshgrid(xs, ys)
    inside = ((xx - centre.x) ** 2 + (yy - centre.y) ** 2) <= city["radius_m"] ** 2
    yi, xi = np.where(inside)
    pts = gpd.GeoDataFrame(
        {"xi": xi, "yi": yi},
        geometry=[Point(xx[y, x], yy[y, x]) for y, x in zip(yi, xi)],
        crs=UTM,
    )
    buf = centre.buffer(city["radius_m"])
    return pts, xx, yy, inside, buf


def _load_noah(city, buf):
    shp = _find_noah_shp(city["noah_province"])
    if shp is None:
        return gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)
    noah = gpd.read_file(shp)
    if noah.crs is None:
        noah = noah.set_crs(4326)
    if noah.crs.to_epsg() != UTM:
        noah = noah.to_crs(epsg=UTM)
    noah["Var"] = pd.to_numeric(noah["Var"], errors="coerce").fillna(0).astype(int)
    noah = gpd.clip(noah[["Var", "geometry"]], buf)
    return noah[~noah.is_empty].copy()


def _sample_noah(points, noah):
    out = points.copy()
    out["noah_class"] = 0
    if noah.empty:
        return out
    joined = gpd.sjoin(
        out[["xi", "yi", "geometry"]],
        noah[["Var", "geometry"]],
        how="left",
        predicate="intersects",
    )
    max_var = joined.groupby(["xi", "yi"])["Var"].max().reset_index()
    out = out.merge(max_var, on=["xi", "yi"], how="left")
    out["noah_class"] = out["Var"].fillna(0).astype(int)
    return out.drop(columns=["Var"])


def _load_water_mask(city, buf):
    """Download OSM permanent water bodies within city buffer.

    Uses a single combined tag query with a short timeout to avoid hanging
    on large coastal features.  Returns a GeoDataFrame in UTM, or empty.
    """
    import signal

    buf_wgs = (
        gpd.GeoSeries([buf], crs=UTM).to_crs(epsg=4326).iloc[0]
    )
    # Focus on inland permanent water only — skip ocean/bay/strait which
    # can return enormous polygons for coastal cities.
    water_tags = {
        "natural": "water",       # lakes, rivers, ponds, reservoirs mapped as areas
        "landuse": "reservoir",
    }

    def _timeout_handler(signum, frame):
        raise TimeoutError("OSM fetch timed out")

    frames = []
    for tags in [{"natural": "water"}, {"landuse": "reservoir"}]:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)  # 30-second hard timeout per query
        try:
            gdf = ox.features_from_polygon(buf_wgs, tags=tags)
            signal.alarm(0)
            polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            if not polys.empty:
                frames.append(polys[["geometry"]])
        except Exception:
            signal.alarm(0)
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], crs=UTM)
    water = pd.concat(frames, ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=4326)
    water = water.to_crs(epsg=UTM)
    water = gpd.clip(water[["geometry"]], buf)
    return water[~water.is_empty].copy()


def _apply_water_mask(points, water):
    """Return boolean array (len=points): True where cell is permanent water."""
    if water.empty:
        return np.zeros(len(points), dtype=bool)
    joined = gpd.sjoin(
        points[["xi", "yi", "geometry"]],
        water[["geometry"]],
        how="left",
        predicate="within",
    )
    is_water_idx = set(joined.dropna(subset=["index_right"]).index)
    mask = np.array([i in is_water_idx for i in range(len(points))], dtype=bool)
    return mask


def _groundsource_counts(points, gs):
    counts = np.zeros(len(points), dtype=float)
    if gs.empty:
        return counts
    joined = gpd.sjoin(
        points[["xi", "yi", "geometry"]],
        gs[["geometry"]],
        how="left",
        predicate="within",
    )
    hit = joined.dropna(subset=["index_right"]).groupby(["xi", "yi"]).size().reset_index(name="n")
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in hit.iterrows():
        counts[idx_map[(int(row["xi"]), int(row["yi"]))]] = float(row["n"])
    return counts


def _classify_area_matched(score_inside, noah_inside, water_inside):
    """Area-match classification, treating water cells as class=-1 (excluded)."""
    valid = ~water_inside
    out = np.full(len(score_inside), -1, dtype=int)  # -1 = water
    score_valid = score_inside[valid]
    noah_valid = noah_inside[valid]

    n_high = int((noah_valid == 3).sum())
    n_med = int((noah_valid == 2).sum())
    n_low = int((noah_valid == 1).sum())

    cls = np.zeros(len(score_valid), dtype=int)
    order = np.argsort(score_valid)[::-1]
    if n_high > 0:
        cls[order[:n_high]] = 3
    if n_med > 0:
        cls[order[n_high:n_high + n_med]] = 2
    if n_low > 0:
        cls[order[n_high + n_med:n_high + n_med + n_low]] = 1

    out[valid] = cls
    return out


def _diff_bucket(diff):
    if diff <= -2:
        return "noah_much_higher"
    if diff == -1:
        return "noah_higher"
    if diff == 0:
        return "match"
    if diff == 1:
        return "gs_higher"
    return "gs_much_higher"


# ── Load all Groundsource data ────────────────────────────────────────────────
print("=" * 72, flush=True)
print("Groundsource empirical hazard — all 5-year windows, water-masked", flush=True)
print("=" * 72, flush=True)
print(f"Windows: {WINDOWS[0][0]}-{WINDOWS[0][1]} → {WINDOWS[-1][0]}-{WINDOWS[-1][1]}", flush=True)

print("[1/4] Loading Groundsource data...", flush=True)
raw = gpd.read_parquet(PARQUET, columns=["start_date", "geometry"])
phil = raw.cx[116:127, 4:22].copy()
phil["start_date"] = pd.to_datetime(phil["start_date"])
phil["year"] = phil["start_date"].dt.year
phil_utm = phil.to_crs(epsg=UTM)
print(f"  {len(phil):,} total Groundsource records in Philippines bounding box", flush=True)

# ── Per-city processing ───────────────────────────────────────────────────────
print("[2/4] Fetching OSM water bodies...", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    # Need buf for this step — rebuild cheaply
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM).iloc[0]
    )
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    n = len(water_by_city[city["slug"]])
    print(f"    {n} water polygon(s)", flush=True)

print("[3/4] Building per-window support surfaces & averaging...", flush=True)
city_data = {}
summary_rows = []

for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    points, xx, yy, inside, buf = _build_grid(city)
    noah = _load_noah(city, buf)
    points = _sample_noah(points, noah)
    water = water_by_city[city["slug"]]
    water_mask = _apply_water_mask(points, water)  # True = permanent water

    gs_city_all = phil_utm[phil_utm.intersects(buf)].copy()
    gs_city_all = gpd.clip(gs_city_all, buf)
    gs_city_all = gs_city_all[~gs_city_all.is_empty].copy()

    # Accumulate normalized support surfaces across windows
    accum = np.zeros(len(points), dtype=float)
    window_event_counts = []

    for w_start, w_end in WINDOWS:
        gs_win = gs_city_all[
            (gs_city_all["year"] >= w_start) & (gs_city_all["year"] <= w_end)
        ].copy()
        window_event_counts.append(len(gs_win))

        counts = _groundsource_counts(points, gs_win)
        raw_grid = np.zeros_like(xx, dtype=float)
        raw_grid[points["yi"].astype(int), points["xi"].astype(int)] = counts
        smooth = gaussian_filter(raw_grid, sigma=SIGMA_CELLS)
        score = np.log1p(smooth[inside])

        mx = score.max()
        if mx > 0:
            score = score / mx
        accum += score

    mean_score = accum / len(WINDOWS)

    # Classify with water mask
    noah_inside = points["noah_class"].values.astype(int)
    water_inside = water_mask
    emp_inside = _classify_area_matched(mean_score, noah_inside, water_inside)

    # Metrics on non-water cells only
    valid = ~water_inside
    emp_v = emp_inside[valid]
    noah_v = noah_inside[valid]
    score_v = mean_score[valid]

    exact = float(np.mean(emp_v == noah_v))
    within_one = float(np.mean(np.abs(emp_v - noah_v) <= 1))
    spearman = pd.Series(noah_v).corr(pd.Series(score_v), method="spearman")
    gs_higher_share = float(np.mean(emp_v > noah_v))
    match_share = float(np.mean(emp_v == noah_v))
    noah_higher_share = float(np.mean(emp_v < noah_v))

    score_norm_v = score_v / score_v.max() if score_v.max() > 0 else score_v
    mean_scores = {
        klass: float(score_norm_v[noah_v == klass].mean())
        if (noah_v == klass).any() else np.nan
        for klass in [0, 1, 2, 3]
    }

    # Build plot DataFrame
    points_wgs = points.to_crs(epsg=4326)
    plot_df = pd.DataFrame({
        "lon": points_wgs.geometry.x,
        "lat": points_wgs.geometry.y,
        "noah_class": noah_inside,
        "emp_class": emp_inside,   # -1 = water
        "is_water": water_inside,
        "diff_bucket": [
            "water" if emp_inside[i] == -1 else _diff_bucket(int(emp_inside[i] - noah_inside[i]))
            for i in range(len(emp_inside))
        ],
    })

    extent = [
        float(plot_df["lon"].min()), float(plot_df["lon"].max()),
        float(plot_df["lat"].min()), float(plot_df["lat"].max()),
    ]

    city_data[city["slug"]] = {
        "city": city,
        "plot_df": plot_df,
        "extent": extent,
        "total_events": int(len(gs_city_all)),
        "window_event_counts": window_event_counts,
        "n_windows_with_data": int(sum(c > 0 for c in window_event_counts)),
        "water_cells": int(water_inside.sum()),
        "exact": exact,
        "within_one": within_one,
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "gs_higher_share": gs_higher_share,
        "match_share": match_share,
        "noah_higher_share": noah_higher_share,
        "mean_scores": mean_scores,
    }
    summary_rows.append({
        "city": city["name"],
        "slug": city["slug"],
        "n_windows": len(WINDOWS),
        "window_range": f"{WINDOWS[0][0]}-{WINDOWS[-1][1]}",
        "total_groundsource_events": int(len(gs_city_all)),
        "water_cells_masked": int(water_inside.sum()),
        "exact_accuracy": exact,
        "within_one_accuracy": within_one,
        "spearman_noah_vs_score": float(spearman) if pd.notna(spearman) else np.nan,
        "gs_higher_share": gs_higher_share,
        "match_share": match_share,
        "noah_higher_share": noah_higher_share,
    })
    rho_str = f"{spearman:.3f}" if pd.notna(spearman) else "nan"
    print(
        f"    total_events={len(gs_city_all):>5} | water_cells={water_inside.sum():>4} | "
        f"exact={exact:.3f} | within1={within_one:.3f} | rho={rho_str}",
        flush=True,
    )

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "noah_groundsource_allwindows_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"  Saved -> {summary_path}", flush=True)

# ── Figures ───────────────────────────────────────────────────────────────────
print("[4/4] Rendering figures...", flush=True)

window_range_label = f"{WINDOWS[0][0]}–{WINDOWS[-1][1]}"

# Figure 1: map panels
fig1, axes1 = plt.subplots(
    len(CITIES), 4,
    figsize=(15.8, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.72, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    f"NOAH vs Temporally-Aggregated Groundsource Hazard ({window_range_label}, {len(WINDOWS)} windows)",
    fontsize=14, fontweight="bold", y=0.997,
)
fig1.text(
    0.5, 0.970,
    f"Mean of {len(WINDOWS)} normalized 5-year support surfaces (Gaussian σ={SIGMA_M} m, {GRID_M} m grid). "
    "Permanent water bodies (OSM) masked in blue.",
    ha="center", fontsize=8.5, color="#444444",
)

headers = ["", "NOAH classes", f"GS aggregated ({len(WINDOWS)} windows)", "Difference (GS − NOAH)"]
for i, header in enumerate(headers):
    axes1[0, i].set_title(header, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d = city_data[slug]
    lbl_ax = axes1[ri, 0]
    lbl_ax.axis("off")
    lbl_ax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"windows={d['n_windows_with_data']}/{len(WINDOWS)}\n"
        f"events={d['total_events']:,}\n"
        f"water={d['water_cells']:,} cells\n"
        f"exact={d['exact']:.2f}\nwithin1={d['within_one']:.2f}",
        ha="center", va="center",
        fontsize=7.8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92, boxstyle="round,pad=0.35"),
    )

    for ci, field in enumerate(["noah_class", "emp_class", "diff_bucket"]):
        ax = axes1[ri, ci + 1]
        ax.set_facecolor("#F1EDE5")
        plot_df = d["plot_df"]

        if field == "noah_class":
            for klass in [0, 1, 2, 3]:
                sub = plot_df[plot_df[field] == klass]
                if sub.empty:
                    continue
                alpha = 0.28 if klass == 0 else 0.90
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=NOAH_COLORS[klass], linewidths=0, alpha=alpha)
            # overlay water mask
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)

        elif field == "emp_class":
            # non-water cells
            non_water = plot_df[~plot_df["is_water"]]
            for klass in [0, 1, 2, 3]:
                sub = non_water[non_water[field] == klass]
                if sub.empty:
                    continue
                alpha = 0.28 if klass == 0 else 0.90
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=NOAH_COLORS[klass], linewidths=0, alpha=alpha)
            # water cells
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)

        else:  # diff_bucket
            for bucket in ["noah_much_higher", "noah_higher", "match", "gs_higher", "gs_much_higher"]:
                sub = plot_df[plot_df["diff_bucket"] == bucket]
                if sub.empty:
                    continue
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
            # water as its own color in diff panel too
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)
            ax.text(
                0.02, 0.98,
                f"rho={d['spearman']:.2f}\nGS>NOAH={d['gs_higher_share']:.2f}\nmatch={d['match_share']:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.3, family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.14"),
            )

        ax.plot(city["lng"], city["lat"], "r*", markersize=5)
        ax.set_xlim(d["extent"][0], d["extent"][1])
        ax.set_ylim(d["extent"][2], d["extent"][3])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.grid(alpha=0.2)

class_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="High"),
    mpatches.Patch(color=WATER_COLOR, label="Permanent water (masked)"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"], label="NOAH much higher"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"], label="NOAH higher"),
    mpatches.Patch(color=DIFF_COLORS["match"], label="Match"),
    mpatches.Patch(color=DIFF_COLORS["gs_higher"], label="GS higher"),
    mpatches.Patch(color=DIFF_COLORS["gs_much_higher"], label="GS much higher"),
]
fig1.legend(
    handles=class_handles + diff_handles,
    loc="lower center", ncol=9, fontsize=7.5, bbox_to_anchor=(0.5, -0.003),
)
fig1.tight_layout(rect=[0.01, 0.04, 0.99, 0.958])
out1 = os.path.join(OUT_DIR, "noah_groundsource_allwindows_01_maps.png")
fig1.savefig(out1, dpi=170, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f"  Saved -> {out1}", flush=True)


# Figure 2: diagnostics
fig2, axes2 = plt.subplots(2, 3, figsize=(17, 10))
fig2.suptitle(
    f"Diagnostics — Aggregated Groundsource ({len(WINDOWS)} windows) vs NOAH",
    fontsize=14, fontweight="bold",
)

# A. Mean score by NOAH class
ax = axes2[0, 0]
for i, city in enumerate(CITIES):
    vals = [city_data[city["slug"]]["mean_scores"][k] for k in [0, 1, 2, 3]]
    ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=2, markersize=5,
            color=CITY_COLORS[i], label=city["name"])
ax.set_title("A. Mean GS score by NOAH class", fontweight="bold")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["No hazard", "Low", "Medium", "High"], rotation=20, ha="right")
ax.set_ylabel("Mean normalized score")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# B. Classification agreement
ax = axes2[0, 1]
x = np.arange(len(CITIES))
w = 0.34
exact_vals = [city_data[c["slug"]]["exact"] for c in CITIES]
within_vals = [city_data[c["slug"]]["within_one"] for c in CITIES]
ax.bar(x - w / 2, exact_vals, w, color="#2E7D32", label="Exact class match")
ax.bar(x + w / 2, within_vals, w, color="#1565C0", label="Within one class")
ax.set_title("B. Classification agreement", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Share of grid cells")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)

# C. Spearman correlation
ax = axes2[0, 2]
rho_vals = [city_data[c["slug"]]["spearman"] for c in CITIES]
ax.bar(x, rho_vals, color=CITY_COLORS, alpha=0.88)
ax.set_title("C. Rank correlation: NOAH class vs mean score", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylabel("Spearman ρ")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

# D. Direction of mismatch
ax = axes2[1, 0]
gs_gt = [city_data[c["slug"]]["gs_higher_share"] for c in CITIES]
match_s = [city_data[c["slug"]]["match_share"] for c in CITIES]
noah_gt = [city_data[c["slug"]]["noah_higher_share"] for c in CITIES]
ax.bar(x, noah_gt, color="#6BAED6", label="NOAH > GS")
ax.bar(x, match_s, bottom=noah_gt, color="#2E7D32", label="Match")
ax.bar(x, gs_gt, bottom=np.array(noah_gt) + np.array(match_s), color="#FDAE61", label="GS > NOAH")
ax.set_title("D. Direction of mismatch", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylabel("Share of grid cells")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)

# E. Events per window per city
ax = axes2[1, 1]
window_labels = [f"{s}-{str(e)[-2:]}" for s, e in WINDOWS]
for i, city in enumerate(CITIES):
    counts = city_data[city["slug"]]["window_event_counts"]
    ax.plot(range(len(WINDOWS)), counts, marker="o", linewidth=1.8, markersize=4,
            color=CITY_COLORS[i], label=city["name"])
ax.set_title("E. Groundsource events per 5-year window", fontweight="bold")
ax.set_xticks(range(len(WINDOWS)))
ax.set_xticklabels(window_labels, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("Events in city buffer")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# F. Water cells masked
ax = axes2[1, 2]
water_counts = [city_data[c["slug"]]["water_cells"] for c in CITIES]
ax.bar(x, water_counts, color=WATER_COLOR, edgecolor="#555", linewidth=0.7)
ax.set_title("F. Permanent water cells masked (OSM)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylabel("Grid cells masked")
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout(rect=[0, 0.01, 1, 0.94])
out2 = os.path.join(OUT_DIR, "noah_groundsource_allwindows_02_diagnostics.png")
fig2.savefig(out2, dpi=170, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved -> {out2}", flush=True)

print("Done. Outputs:", flush=True)
for path in [out1, out2, summary_path]:
    print(f"  {path}", flush=True)
