"""
noah_source_robustness.py
=========================
Two robustness plots for the NOAH 5-yr hazard validation:

  Plot A — Robustness over spatiality (data sources)
            Heatmap + grouped bars: Spearman ρ for each city × data source.
            Sources: AI4G Sentinel-1 SAR (10-yr), PhilSA multi-sensor SAR
            (2022-2026), WRI Aqueduct RP-5 riverine model.

  Plot B — Robustness over spatial scale
            For each city, vary the analysis radius (40 % → 120 % of nominal)
            and plot how Spearman ρ changes (NOAH vs Aqueduct, using the
            per-cell parquet saved by noah_aqueduct_comparison.py).

Prerequisites
-------------
Run these first:
  python3 analysis/noah_ai4g_comparison.py          → output/noah_validation/ai4g/noah_ai4g_summary.csv
  python3 analysis/noah_philsa_allfiles_comparison.py → output/noah_validation/philsa/noah_philsa_allfiles_summary.csv
  python3 analysis/noah_aqueduct_comparison.py      → output/noah_validation/aqueduct/noah_aqueduct_summary.csv
                                                       output/noah_validation/aqueduct/noah_aqueduct_cells.parquet

Outputs
-------
  output/noah_validation/noah_robustness_A_sources.png
  output/noah_validation/noah_robustness_B_scale.png
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(ROOT, "output", "noah_validation")
OUT_DIR = VAL_DIR
os.makedirs(OUT_DIR, exist_ok=True)

AI4G_CSV    = os.path.join(VAL_DIR, "ai4g",    "noah_ai4g_summary.csv")
PHILSA_CSV  = os.path.join(VAL_DIR, "philsa",  "noah_philsa_allfiles_summary.csv")
AQ_CSV      = os.path.join(VAL_DIR, "aqueduct","noah_aqueduct_summary.csv")
AQ_CELLS    = os.path.join(VAL_DIR, "aqueduct","noah_aqueduct_cells.parquet")

# Canonical 10-city order (matches paper tables)
CITY_ORDER = [
    "Tuguegarao", "Ilagan", "Dagupan", "San Fernando", "Manila",
    "Naga", "Daet", "Cagayan de Oro", "Butuan", "Cotabato",
]

# Nominal radii (km) used in the comparison scripts
NOMINAL_RADII = {
    "Manila":         20_000,
    "San Fernando":   12_000,
    "Dagupan":        12_000,
    "Naga":           10_000,
    "Daet":            8_000,
    "Cagayan de Oro": 12_000,
    "Butuan":         10_000,
    "Tuguegarao":     10_000,
    "Ilagan":         10_000,
    "Cotabato":       10_000,
}

REGION_LABELS = {
    "Manila":         "NCR",
    "San Fernando":   "C. Luzon",
    "Dagupan":        "Ilocos",
    "Naga":           "Bicol",
    "Daet":           "Bicol",
    "Cagayan de Oro": "N. Mindanao",
    "Butuan":         "Caraga",
    "Tuguegarao":     "Cagayan\nValley",
    "Ilagan":         "Cagayan\nValley",
    "Cotabato":       "BARMM",
}

# Colours for sources
SOURCE_COLORS = {
    "AI4G (SAR)":       "#1565C0",
    "PhilSA (SAR)":     "#6A1B9A",
    "Aqueduct (RP-5)":  "#2E7D32",
}

CITY_COLORS = [
    "#1565C0", "#37474F", "#6A1B9A", "#2E7D32", "#B71C1C",
    "#00796B", "#E65100", "#880E4F", "#F57C00", "#1B5E20",
]

RADIUS_FRACTIONS = [0.40, 0.55, 0.70, 0.85, 1.00, 1.15]


# ===========================================================================
# Load summary CSVs
# ===========================================================================

def _load_summary(path, source_label):
    if not os.path.exists(path):
        print(f"  WARN: {path} not found — run the corresponding comparison script first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = source_label
    return df


print("=" * 72, flush=True)
print("NOAH robustness analysis — loading comparison results…", flush=True)
print("=" * 72, flush=True)

ai4g_df   = _load_summary(AI4G_CSV,   "AI4G (SAR)")
philsa_df = _load_summary(PHILSA_CSV, "PhilSA (SAR)")
aq_df     = _load_summary(AQ_CSV,     "Aqueduct (RP-5)")

sources_available = []
if not ai4g_df.empty:
    sources_available.append("AI4G (SAR)")
    print(f"  AI4G: {len(ai4g_df)} cities", flush=True)
if not philsa_df.empty:
    sources_available.append("PhilSA (SAR)")
    print(f"  PhilSA: {len(philsa_df)} cities", flush=True)
if not aq_df.empty:
    sources_available.append("Aqueduct (RP-5)")
    print(f"  Aqueduct: {len(aq_df)} cities", flush=True)

if not sources_available:
    raise SystemExit("No comparison CSVs found. Run noah_ai4g_comparison.py and "
                     "noah_aqueduct_comparison.py first.")

# Build a combined rho table (cities × sources)
combined = pd.concat([ai4g_df, philsa_df, aq_df], ignore_index=True)
rho_pivot = (combined.pivot_table(index="city", columns="source",
                                  values="spearman", aggfunc="first")
             .reindex(CITY_ORDER))

print("\nSpearman ρ by city and source:")
print(rho_pivot.to_string(float_format=lambda v: f"{v:.3f}"))

# ===========================================================================
# Plot A — Robustness over spatiality (data sources)
# ===========================================================================
print("\n[1/2] Generating Plot A: robustness over data sources…", flush=True)

n_cities  = len(CITY_ORDER)
n_sources = len(sources_available)

fig_a = plt.figure(figsize=(17, 8))
fig_a.patch.set_facecolor("#F7F7F7")
gs = gridspec.GridSpec(1, 2, figure=fig_a, width_ratios=[1.1, 1.4], wspace=0.30)

# ── Left: heatmap ────────────────────────────────────────────────────────
ax_heat = fig_a.add_subplot(gs[0])
heatmap_data = rho_pivot[sources_available].values.astype(float)   # cities × sources

# Mask NaN cells
masked = np.ma.masked_invalid(heatmap_data)
cmap   = plt.cm.RdYlGn
cmap.set_bad(color="#E0E0E0")
im = ax_heat.imshow(masked, cmap=cmap, vmin=-0.2, vmax=0.7,
                    aspect="auto", interpolation="nearest")

ax_heat.set_xticks(range(n_sources))
ax_heat.set_xticklabels(sources_available, fontsize=10, fontweight="bold")
ax_heat.set_yticks(range(n_cities))
ax_heat.set_yticklabels(
    [f"{c}  [{REGION_LABELS.get(c, '')}]" for c in CITY_ORDER],
    fontsize=9,
)
ax_heat.set_title("Spearman ρ: NOAH vs each data source", fontweight="bold", pad=10)

# Annotate cells
for i in range(n_cities):
    for j in range(n_sources):
        v = heatmap_data[i, j]
        txt = f"{v:.2f}" if not np.isnan(v) else "—"
        col = "white" if (not np.isnan(v) and abs(v) > 0.35) else "#333333"
        ax_heat.text(j, i, txt, ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=col)

cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label("Spearman ρ", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── Right: grouped bar chart ──────────────────────────────────────────────
ax_bar = fig_a.add_subplot(gs[1])
x       = np.arange(n_cities)
bar_w   = 0.35 if n_sources == 2 else 0.22
offsets = np.linspace(-(n_sources - 1) / 2 * bar_w,
                       (n_sources - 1) / 2 * bar_w,
                       n_sources)

for j, src in enumerate(sources_available):
    rho_vals = rho_pivot[src].values.astype(float)
    bars = ax_bar.bar(
        x + offsets[j],
        np.where(np.isnan(rho_vals), 0, rho_vals),
        bar_w,
        label=src,
        color=SOURCE_COLORS.get(src, f"C{j}"),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    for xi, (bar, v) in enumerate(zip(bars, rho_vals)):
        if np.isnan(v):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, 0.01,
                        "—", ha="center", va="bottom", fontsize=7, color="#888")
        else:
            ypos = bar.get_height() + (0.015 if v >= 0 else -0.04)
            ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)

ax_bar.axhline(0, color="#666", linewidth=0.9, linestyle="--")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(CITY_ORDER, rotation=35, ha="right", fontsize=8.5)
ax_bar.set_ylabel("Spearman ρ", fontsize=10)
ax_bar.set_ylim(-0.25, 0.85)
ax_bar.set_title("Spearman ρ by city and source", fontweight="bold", pad=10)
ax_bar.legend(fontsize=9, loc="upper right")
ax_bar.grid(axis="y", alpha=0.3, linewidth=0.7)
ax_bar.set_facecolor("#FAFAFA")

# Shade background by region (every other region)
region_groups = {}
for city in CITY_ORDER:
    region_groups.setdefault(REGION_LABELS.get(city, city), []).append(
        CITY_ORDER.index(city)
    )
for k, (region, idxs) in enumerate(region_groups.items()):
    if k % 2 == 0:
        ax_bar.axvspan(min(idxs) - 0.5, max(idxs) + 0.5,
                       color="#F0F0F0", alpha=0.5, zorder=0)

fig_a.suptitle(
    "Robustness across data sources\n"
    "NOAH 5-yr hazard validated against independent observational and model-based datasets",
    fontsize=12, fontweight="bold", y=1.005,
)
fig_a.tight_layout()

p_a = os.path.join(OUT_DIR, "noah_robustness_A_sources.png")
fig_a.savefig(p_a, dpi=150, bbox_inches="tight")
plt.close(fig_a)
print(f"  Saved → {p_a}", flush=True)


# ===========================================================================
# Plot B — Robustness over spatial scale
# ===========================================================================
print("\n[2/2] Generating Plot B: robustness over spatial scale…", flush=True)

if not os.path.exists(AQ_CELLS):
    print(f"  WARN: {AQ_CELLS} not found — run noah_aqueduct_comparison.py first.")
    raise SystemExit("Cannot generate Plot B without per-cell Aqueduct data.")

cells_df = pd.read_parquet(AQ_CELLS)

# For each city × radius fraction, compute Spearman ρ between noah_cls and aq_cls
# using only cells within that radius (approximated by filtering dist_m column)
print("  Computing ρ at each radius fraction…", flush=True)
results = {}

for city_name in CITY_ORDER:
    nominal_r = NOMINAL_RADII.get(city_name, 10_000)
    city_cells = cells_df[cells_df["city"] == city_name].copy()
    if city_cells.empty:
        print(f"    {city_name}: no cells in parquet", flush=True)
        continue

    # Compute distance from centroid (use mean of non-water cells as proxy)
    non_water = city_cells[~city_cells["is_water"]]
    if non_water.empty:
        continue

    # dist_m is pre-computed in noah_aqueduct_comparison.py as distance from
    # mean of all grid points — we re-derive it properly here
    # Re-project to UTM and compute distance from grid centre
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        city_cells,
        geometry=gpd.points_from_xy(city_cells["lon"], city_cells["lat"]),
        crs=4326,
    ).to_crs(epsg=32651)

    cx = gdf.geometry.x.mean()
    cy = gdf.geometry.y.mean()
    dist = np.sqrt((gdf.geometry.x.values - cx) ** 2 +
                   (gdf.geometry.y.values - cy) ** 2)
    city_cells = city_cells.copy()
    city_cells["dist_recalc"] = dist

    rho_by_frac = []
    for frac in RADIUS_FRACTIONS:
        sub = city_cells[city_cells["dist_recalc"] <= frac * nominal_r]
        valid = sub[~sub["is_water"]]
        n_cls = valid["noah_cls"].values.astype(int)
        a_cls = valid["aq_cls"].values.astype(int)
        if len(np.unique(n_cls)) > 1 and len(np.unique(a_cls)) > 1 and len(valid) >= 10:
            rho = pd.Series(n_cls).corr(pd.Series(a_cls), method="spearman")
        else:
            rho = np.nan
        rho_by_frac.append(rho)
    results[city_name] = rho_by_frac
    print(f"    {city_name}: {[f'{v:.2f}' if not np.isnan(v) else 'nan' for v in rho_by_frac]}",
          flush=True)

# ── Figure layout: 2-panel (line chart + heatmap) ────────────────────────
fig_b, (ax_line, ax_hm) = plt.subplots(
    1, 2, figsize=(15, 6),
    gridspec_kw={"width_ratios": [1.6, 1.0]},
)
fig_b.patch.set_facecolor("#F7F7F7")

# ── Left: line chart ─────────────────────────────────────────────────────
radius_labels = [f"{int(f*100)}%" for f in RADIUS_FRACTIONS]

for ci, city_name in enumerate([c for c in CITY_ORDER if c in results]):
    rho_vals = results[city_name]
    nominal_r = NOMINAL_RADII.get(city_name, 10_000)
    radii_km  = [f * nominal_r / 1000 for f in RADIUS_FRACTIONS]
    valid_x   = [r for r, v in zip(radii_km, rho_vals) if not np.isnan(v)]
    valid_y   = [v for v in rho_vals if not np.isnan(v)]
    if not valid_x:
        continue
    ax_line.plot(valid_x, valid_y, marker="o", linewidth=2, markersize=5,
                 color=CITY_COLORS[ci], label=city_name, alpha=0.88)

ax_line.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax_line.set_xlabel("Analysis radius (km)", fontsize=10)
ax_line.set_ylabel("Spearman ρ  (NOAH vs Aqueduct)", fontsize=10)
ax_line.set_title(
    "NOAH validation robustness over spatial scale\n"
    "(NOAH vs Aqueduct RP-5, varying city radius)",
    fontweight="bold",
)
ax_line.grid(alpha=0.3)
ax_line.set_facecolor("#FAFAFA")
ax_line.legend(fontsize=8, loc="lower right", ncol=2)
ax_line.set_ylim(-0.3, 1.0)

# ── Right: heatmap ρ(city, radius fraction) ───────────────────────────────
cities_with_results = [c for c in CITY_ORDER if c in results]
hm_data = np.array([results[c] for c in cities_with_results])

cmap2 = plt.cm.RdYlGn
cmap2.set_bad(color="#E0E0E0")
masked2 = np.ma.masked_invalid(hm_data)
im2 = ax_hm.imshow(masked2, cmap=cmap2, vmin=-0.3, vmax=0.85,
                   aspect="auto", interpolation="nearest")

ax_hm.set_xticks(range(len(RADIUS_FRACTIONS)))
ax_hm.set_xticklabels(radius_labels, fontsize=9)
ax_hm.set_yticks(range(len(cities_with_results)))
ax_hm.set_yticklabels(cities_with_results, fontsize=9)
ax_hm.set_xlabel("Radius fraction (% of nominal)", fontsize=9)
ax_hm.set_title("ρ by city × radius fraction", fontweight="bold", pad=8)

for i in range(len(cities_with_results)):
    for j in range(len(RADIUS_FRACTIONS)):
        v = hm_data[i, j]
        txt = f"{v:.2f}" if not np.isnan(v) else "—"
        col = "white" if (not np.isnan(v) and abs(v) > 0.4) else "#333333"
        ax_hm.text(j, i, txt, ha="center", va="center",
                   fontsize=8.5, fontweight="bold", color=col)

cbar2 = plt.colorbar(im2, ax=ax_hm, fraction=0.046, pad=0.04)
cbar2.set_label("Spearman ρ", fontsize=8)
cbar2.ax.tick_params(labelsize=7.5)

# Nominal radius marker on x-axis of heatmap (fraction = 1.00 → index 4)
nom_idx = RADIUS_FRACTIONS.index(1.00)
ax_hm.axvline(nom_idx, color="#B71C1C", linewidth=2, linestyle="--", alpha=0.6,
              label="Nominal radius")
ax_hm.text(nom_idx + 0.08, -0.7, "nominal\nradius",
           fontsize=7, color="#B71C1C", ha="left", va="top")

fig_b.suptitle(
    "Robustness across spatial scale\n"
    "NOAH 5-yr hazard vs WRI Aqueduct RP-5 at varying analysis radii",
    fontsize=12, fontweight="bold", y=1.02,
)
fig_b.tight_layout()

p_b = os.path.join(OUT_DIR, "noah_robustness_B_scale.png")
fig_b.savefig(p_b, dpi=150, bbox_inches="tight")
plt.close(fig_b)
print(f"  Saved → {p_b}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p_a}", flush=True)
print(f"  {p_b}", flush=True)
