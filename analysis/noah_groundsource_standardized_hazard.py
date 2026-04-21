"""
noah_groundsource_standardized_hazard.py
========================================
Standardize NOAH and Groundsource onto one shared hazard scale.

Goal
----
Create a single pooled Groundsource hazard scale across all five cities so
"high" means the same thing everywhere, then compare that standardized
Groundsource hazard to NOAH's low / medium / high classes.

Approach
--------
1. Use a common 250 m grid and a 750 m Gaussian smoothing radius on
   Groundsource counts for 2021-2025.
2. Pool smoothed Groundsource scores across all cities.
3. Assign one global set of thresholds so the pooled Groundsource class shares
   match the pooled NOAH class shares across all grid cells.
4. Apply those same thresholds back to each city.

Outputs
-------
  output/noah_groundsource_standardized_01_maps_2021_25.png
  output/noah_groundsource_standardized_02_diagnostics_2021_25.png
  output/noah_groundsource_standardized_2021_25_summary.csv
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

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "noah_validation", "groundsource")
PARQUET = os.path.join(ROOT, "data", "google_gemini_flood", "groundsource_2026.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

UTM = 32651
GRID_M = 250
SIGMA_M = 750
SIGMA_CELLS = SIGMA_M / GRID_M
WINDOW = (2021, 2025)
WINDOW_LABEL = "2021-25"

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

CLASS_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
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


def _classify_from_thresholds(score, thresholds):
    t_low, t_med, t_high = thresholds
    out = np.zeros(len(score), dtype=int)
    out[score >= t_low] = 1
    out[score >= t_med] = 2
    out[score >= t_high] = 3
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


print("=" * 72, flush=True)
print("Standardized smoothed Groundsource hazard vs NOAH 5-year hazard", flush=True)
print("=" * 72, flush=True)
print(
    f"Window: {WINDOW[0]}-{WINDOW[1]} | grid: {GRID_M} m | Gaussian sigma: {SIGMA_M} m",
    flush=True,
)

print("[1/4] Loading Groundsource data...", flush=True)
raw = gpd.read_parquet(PARQUET, columns=["start_date", "geometry"])
phil = raw.cx[116:127, 4:22].copy()
phil["start_date"] = pd.to_datetime(phil["start_date"])
phil["year"] = phil["start_date"].dt.year
phil = phil[(phil["year"] >= WINDOW[0]) & (phil["year"] <= WINDOW[1])].copy()
phil_utm = phil.to_crs(epsg=UTM)
print(f"  Using {len(phil):,} Groundsource records in {WINDOW_LABEL}", flush=True)

city_data = {}
all_scores = []
all_noah = []

print("[2/4] Building pooled smoothed Groundsource score surface...", flush=True)
for city in CITIES:
    print(f"  {city['name']}...", flush=True)
    points, xx, yy, inside, buf = _build_grid(city)
    noah = _load_noah(city, buf)
    points = _sample_noah(points, noah)

    gs_city = phil_utm[phil_utm.intersects(buf)].copy()
    gs_city = gpd.clip(gs_city, buf)
    gs_city = gs_city[~gs_city.is_empty].copy()

    counts = _groundsource_counts(points, gs_city)
    raw_grid = np.zeros_like(xx, dtype=float)
    raw_grid[points["yi"].astype(int), points["xi"].astype(int)] = counts
    smooth = gaussian_filter(raw_grid, sigma=SIGMA_CELLS)
    smooth[~inside] = np.nan
    score = np.log1p(smooth[inside])

    points_wgs = points.to_crs(epsg=4326)
    plot_df = pd.DataFrame({
        "lon": points_wgs.geometry.x,
        "lat": points_wgs.geometry.y,
        "noah_class": points["noah_class"].values.astype(int),
        "score": score,
    })
    extent = [
        float(plot_df["lon"].min()),
        float(plot_df["lon"].max()),
        float(plot_df["lat"].min()),
        float(plot_df["lat"].max()),
    ]

    city_data[city["slug"]] = {
        "city": city,
        "plot_df": plot_df,
        "extent": extent,
        "n_events": int(len(gs_city)),
    }
    all_scores.append(score)
    all_noah.append(points["noah_class"].values.astype(int))
    print(f"    events={len(gs_city):>4}", flush=True)

pooled_scores = np.concatenate(all_scores)
pooled_noah = np.concatenate(all_noah)

# Robust normalization for plotting only.
score_hi = np.percentile(pooled_scores, 99.5) if len(pooled_scores) else 1.0
score_hi = max(score_hi, pooled_scores.max() if len(pooled_scores) else 1.0, 1e-9)
pooled_norm = np.clip(pooled_scores / score_hi, 0, 1)

# Global class shares from NOAH pooled across all cities.
n_high = int((pooled_noah == 3).sum())
n_med = int((pooled_noah == 2).sum())
n_low = int((pooled_noah == 1).sum())
order = np.argsort(pooled_scores)[::-1]
gs_class_pooled = np.zeros(len(pooled_scores), dtype=int)
gs_class_pooled[order[:n_high]] = 3
gs_class_pooled[order[n_high:n_high + n_med]] = 2
gs_class_pooled[order[n_high + n_med:n_high + n_med + n_low]] = 1

if n_low > 0:
    t_low = pooled_scores[order[n_high + n_med + n_low - 1]]
else:
    t_low = np.inf
if n_med > 0:
    t_med = pooled_scores[order[n_high + n_med - 1]]
else:
    t_med = np.inf
if n_high > 0:
    t_high = pooled_scores[order[n_high - 1]]
else:
    t_high = np.inf
thresholds = (t_low, t_med, t_high)

print("[3/4] Applying one shared Groundsource hazard scale to every city...", flush=True)
summary_rows = []
offset = 0
for city in CITIES:
    slug = city["slug"]
    d = city_data[slug]
    n = len(d["plot_df"])
    loc_scores = pooled_scores[offset:offset + n]
    loc_norm = pooled_norm[offset:offset + n]
    loc_noah = pooled_noah[offset:offset + n]
    loc_gs = gs_class_pooled[offset:offset + n]
    offset += n

    d["plot_df"]["score_norm"] = loc_norm
    d["plot_df"]["gs_class"] = loc_gs
    d["plot_df"]["diff_bucket"] = [_diff_bucket(int(g - n_)) for g, n_ in zip(loc_gs, loc_noah)]

    spearman = pd.Series(loc_noah).corr(pd.Series(loc_scores), method="spearman")
    exact = float(np.mean(loc_gs == loc_noah))
    within_one = float(np.mean(np.abs(loc_gs - loc_noah) <= 1))
    gs_higher = float(np.mean(loc_gs > loc_noah))
    match = float(np.mean(loc_gs == loc_noah))
    noah_higher = float(np.mean(loc_gs < loc_noah))
    mean_scores = {
        k: float(loc_norm[loc_noah == k].mean()) if (loc_noah == k).any() else np.nan
        for k in [0, 1, 2, 3]
    }

    summary_rows.append({
        "city": city["name"],
        "slug": slug,
        "window": WINDOW_LABEL,
        "grid_m": GRID_M,
        "sigma_m": SIGMA_M,
        "groundsource_events": d["n_events"],
        "global_threshold_low": t_low,
        "global_threshold_medium": t_med,
        "global_threshold_high": t_high,
        "exact_accuracy": exact,
        "within_one_accuracy": within_one,
        "spearman_noah_vs_score": float(spearman) if pd.notna(spearman) else np.nan,
        "gs_higher_share": gs_higher,
        "match_share": match,
        "noah_higher_share": noah_higher,
        "mean_score_no_hazard": mean_scores[0],
        "mean_score_low": mean_scores[1],
        "mean_score_medium": mean_scores[2],
        "mean_score_high": mean_scores[3],
    })
    print(
        f"  {city['name']}: exact={exact:.3f} | within1={within_one:.3f} | "
        f"rho={spearman:.3f}" if pd.notna(spearman) else
        f"  {city['name']}: exact={exact:.3f} | within1={within_one:.3f} | rho=nan",
        flush=True,
    )

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "noah_groundsource_standardized_2021_25_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"  Saved -> {summary_path}", flush=True)

print("[4/4] Rendering figures...", flush=True)

fig1, axes1 = plt.subplots(
    len(CITIES),
    4,
    figsize=(15.8, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.72, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    "NOAH vs Standardized Smoothed Groundsource Hazard, Shared Scale (2021-2025)",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig1.text(
    0.5,
    0.968,
    "One pooled Groundsource score scale across all cities; class thresholds chosen so pooled GS class shares match pooled NOAH class shares.",
    ha="center",
    fontsize=9,
    color="#444444",
)
headers = ["", "NOAH classes", "Groundsource classes (shared scale)", "Difference (GS - NOAH)"]
for i, header in enumerate(headers):
    axes1[0, i].set_title(header, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d = city_data[slug]
    stat = summary[summary["slug"] == slug].iloc[0]
    lbl_ax = axes1[ri, 0]
    lbl_ax.axis("off")
    lbl_ax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n{WINDOW_LABEL}\n"
        f"events={d['n_events']:,}\nexact={stat['exact_accuracy']:.2f}\nwithin1={stat['within_one_accuracy']:.2f}",
        ha="center", va="center",
        fontsize=8.2, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92, boxstyle="round,pad=0.35"),
    )

    for ci, field in enumerate(["noah_class", "gs_class", "diff_bucket"]):
        ax = axes1[ri, ci + 1]
        ax.set_facecolor("#F1EDE5")
        plot_df = d["plot_df"]
        if field in ["noah_class", "gs_class"]:
            for klass in [0, 1, 2, 3]:
                sub = plot_df[plot_df[field] == klass]
                if sub.empty:
                    continue
                alpha = 0.28 if klass == 0 else 0.90
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s", c=CLASS_COLORS[klass], linewidths=0, alpha=alpha)
        else:
            for bucket in ["noah_much_higher", "noah_higher", "match", "gs_higher", "gs_much_higher"]:
                sub = plot_df[plot_df["diff_bucket"] == bucket]
                if sub.empty:
                    continue
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s", c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
            ax.text(
                0.02, 0.98,
                f"rho={stat['spearman_noah_vs_score']:.2f}\nGS>NOAH={stat['gs_higher_share']:.2f}\nmatch={stat['match_share']:.2f}",
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
    mpatches.Patch(color=CLASS_COLORS[1], label="Low"),
    mpatches.Patch(color=CLASS_COLORS[2], label="Medium"),
    mpatches.Patch(color=CLASS_COLORS[3], label="High"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"], label="NOAH much higher"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"], label="NOAH higher"),
    mpatches.Patch(color=DIFF_COLORS["match"], label="Match"),
    mpatches.Patch(color=DIFF_COLORS["gs_higher"], label="GS higher"),
    mpatches.Patch(color=DIFF_COLORS["gs_much_higher"], label="GS much higher"),
]
fig1.legend(handles=class_handles + diff_handles, loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.003))
fig1.tight_layout(rect=[0.01, 0.04, 0.99, 0.955])
out1 = os.path.join(OUT_DIR, "noah_groundsource_standardized_01_maps_2021_25.png")
fig1.savefig(out1, dpi=170, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f"  Saved -> {out1}", flush=True)

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle(
    "Diagnostics for Standardized Smoothed Groundsource Hazard vs NOAH (2021-2025)",
    fontsize=15,
    fontweight="bold",
)
fig2.text(
    0.5,
    0.945,
    f"Global GS score thresholds: low={t_low:.3f}, medium={t_med:.3f}, high={t_high:.3f} (log-smoothed score units).",
    ha="center",
    fontsize=9,
    color="#444444",
)

ax = axes2[0, 0]
for i, city in enumerate(CITIES):
    vals = [summary.loc[summary["slug"] == city["slug"], f"mean_score_{k}"].iloc[0] for k in ["no_hazard", "low", "medium", "high"]]
    ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=2, markersize=5, color=CITY_COLORS[i], label=city["name"])
ax.set_title("A. Mean normalized Groundsource score by NOAH class", fontweight="bold")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["No hazard", "Low", "Medium", "High"], rotation=20, ha="right")
ax.set_ylabel("Mean normalized support score")
ax.grid(alpha=0.3)

ax = axes2[0, 1]
x = np.arange(len(CITIES))
w = 0.34
ax.bar(x - w / 2, summary["exact_accuracy"], w, color="#2E7D32", label="Exact class match")
ax.bar(x + w / 2, summary["within_one_accuracy"], w, color="#1565C0", label="Within one class")
ax.set_title("B. Classification agreement", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Share of grid cells")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)

ax = axes2[1, 0]
ax.bar(x, summary["spearman_noah_vs_score"], color=CITY_COLORS, alpha=0.88)
ax.set_title("C. Rank correlation: NOAH class vs GS support", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

ax = axes2[1, 1]
ax.bar(x, summary["noah_higher_share"], color="#6BAED6", label="NOAH > GS")
ax.bar(x, summary["match_share"], bottom=summary["noah_higher_share"], color="#2E7D32", label="Match")
ax.bar(x, summary["gs_higher_share"], bottom=summary["noah_higher_share"] + summary["match_share"], color="#FDAE61", label="GS > NOAH")
ax.set_title("D. Direction of mismatch", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in CITIES], rotation=25, ha="right")
ax.set_ylabel("Share of grid cells")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)

handles, labels = axes2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig2.tight_layout(rect=[0, 0.04, 1, 0.95])
out2 = os.path.join(OUT_DIR, "noah_groundsource_standardized_02_diagnostics_2021_25.png")
fig2.savefig(out2, dpi=170, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved -> {out2}", flush=True)

print("Outputs:", flush=True)
for path in [out1, out2, summary_path]:
    print(f"  {path}", flush=True)
