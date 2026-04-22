"""
noah_aqueduct_comparison.py
===========================
Compare NOAH 5-yr flood hazard maps against WRI Aqueduct Floods v2 (WATCH
historical, 5-year return period riverine inundation depth) across 10 Philippine
cities.

Data sources
------------
NOAH       — static modelled hazard (Var 1/2/3 = Low/Medium/High, 5-yr return period)
             Province shapefiles, UTM Zone 51N
Aqueduct   — WRI Aqueduct Floods v2, WATCH historical, RP 5-year riverine depth
             GeoTIFF, ~1 km resolution, global coverage, float32 (metres)
             URL: https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/
                  inunriver_historical_000000000WATCH_1980_rp00005.tif
             Licence: Creative Commons Attribution 4.0 (CC BY 4.0)

Method
------
1. Download WRI Aqueduct 5-yr riverine GeoTIFF (once, cached at data/aqueduct/).
2. Build 250 m UTM grid per city (circle of radius_m).
3. Sample Aqueduct flood depth at each grid cell centre using rasterio.
4. Classify depth into NOAH-compatible hazard classes:
     nodata / <= 0 m → class 0 (no flood)
     0 < depth ≤ 0.5 m  → class 1 (Low,    matches NOAH Var=1 <0.5 m)
     0.5 < depth ≤ 1.5 m → class 2 (Medium, matches NOAH Var=2 0.5–1.5 m)
     depth > 1.5 m       → class 3 (High,   matches NOAH Var=3 >1.5 m)
5. Compare Aqueduct class vs NOAH class: Spearman ρ, exact match, within-1.
6. Produce consensus risk maps and diagnostics.

Outputs
-------
  output/noah_validation/aqueduct/noah_aqueduct_01_maps.png
  output/noah_validation/aqueduct/noah_aqueduct_02_diagnostics.png
  output/noah_validation/aqueduct/noah_aqueduct_summary.csv
  output/noah_validation/aqueduct/noah_aqueduct_cells.parquet   ← per-cell data
"""

import glob
import os
import urllib.request
import warnings

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point
import osmnx as ox

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(ROOT, "output", "noah_validation", "aqueduct")
AQ_DIR       = os.path.join(ROOT, "data", "aqueduct")
NOAH_BASE    = os.path.join(ROOT, "data", "noah", "5yr")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AQ_DIR,  exist_ok=True)

AQ_TIF_URL  = (
    "https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"
    "inunriver_historical_000000000WATCH_1980_rp00005.tif"
)
AQ_TIF_PATH = os.path.join(AQ_DIR, "inunriver_historical_000000000WATCH_1980_rp00005.tif")

UTM    = 32651    # WGS84 / UTM Zone 51N
GRID_M = 250

NOAH_ACTIVE_CLASS = 2   # Var ≥ 2 counts as "NOAH active" for consensus

# Depth thresholds (metres) — aligned with NOAH Var breakpoints
AQ_BREAKS = [0.0, 0.5, 1.5]   # (0, 0.5] → 1, (0.5, 1.5] → 2, > 1.5 → 3

CITIES = [
    {"name": "Manila",         "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "San Fernando",   "slug": "san_fernando",   "lat": 15.0286, "lng": 120.6940,
     "radius_m": 12_000, "noah_province": "Pampanga",            "region": "Central Luzon"},
    {"name": "Dagupan",        "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan",          "region": "Ilocos"},
    {"name": "Naga",           "slug": "naga",           "lat": 13.6218, "lng": 123.1948,
     "radius_m": 10_000, "noah_province": "Camarines Sur",       "region": "Bicol"},
    {"name": "Daet",           "slug": "daet",           "lat": 14.1167, "lng": 122.9500,
     "radius_m":  8_000, "noah_province": "Camarines Norte",     "region": "Bicol"},
    {"name": "Cagayan de Oro", "slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental",    "region": "Northern Mindanao"},
    {"name": "Butuan",         "slug": "butuan",         "lat": 8.9515,  "lng": 125.5277,
     "radius_m": 10_000, "noah_province": None,                  "region": "Caraga"},
    {"name": "Tuguegarao",     "slug": "tuguegarao",     "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan",             "region": "Cagayan Valley"},
    {"name": "Ilagan",         "slug": "ilagan",         "lat": 17.1485, "lng": 121.8892,
     "radius_m": 10_000, "noah_province": "Isabela",             "region": "Cagayan Valley"},
    {"name": "Cotabato",       "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao",         "region": "BARMM"},
]

# Colour palettes
NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
AQ_COLORS   = {0: "#F1EDE5", 1: "#64B5F6", 2: "#1565C0", 3: "#0D1B4E"}
WATER_COLOR = "#2CBBB4"

CONSENSUS_COLORS = {
    "confirmed": "#B71C1C",
    "modelled":  "#1565C0",
    "empirical": "#E65100",
    "low":       "#E8E8E8",
    "water":     WATER_COLOR,
}

DIFF_COLORS = {
    "noah_much_higher": "#54278F",
    "noah_higher":      "#9467BD",
    "match":            "#2E7D32",
    "aq_higher":        "#FDAE61",
    "aq_much_higher":   "#B2182B",
}

CITY_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C",
    "#00796B", "#F57C00", "#37474F", "#880E4F", "#1B5E20",
]


# ===========================================================================
# Step 1 — Download & cache WRI Aqueduct GeoTIFF
# ===========================================================================

def load_aqueduct_raster():
    """Download Aqueduct TIF if needed; return rasterio DatasetReader."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")

    if not os.path.exists(AQ_TIF_PATH):
        size_mb = 82.6
        print(f"  Downloading Aqueduct TIF (~{size_mb:.0f} MB) …", flush=True)

        def _reporthook(blocks, block_size, total):
            downloaded = blocks * block_size
            if total > 0:
                pct = min(100, downloaded * 100 / total)
                print(f"\r    {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(AQ_TIF_URL, AQ_TIF_PATH, reporthook=_reporthook)
        print(f"\n  Saved → {AQ_TIF_PATH}", flush=True)
    else:
        print(f"  Using cached Aqueduct TIF: {AQ_TIF_PATH}", flush=True)

    import rasterio
    return rasterio.open(AQ_TIF_PATH)


def _classify_depth(depth_m):
    """Convert WRI Aqueduct depth (metres) to NOAH-equivalent class 0–3."""
    if depth_m is None or np.isnan(depth_m) or depth_m <= 0:
        return 0
    if depth_m <= 0.5:
        return 1
    if depth_m <= 1.5:
        return 2
    return 3


# ===========================================================================
# Grid / NOAH / water helpers (shared pattern with noah_ai4g_comparison.py)
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


def _load_water_mask(city, buf):
    import signal
    buf_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(epsg=4326).iloc[0]

    def _timeout(signum, frame):
        raise TimeoutError

    frames = []
    for tags in [{"natural": "water"}, {"landuse": "reservoir"}]:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(30)
        try:
            gdf = ox.features_from_polygon(buf_wgs, tags=tags)
            signal.alarm(0)
            polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not polys.empty:
                frames.append(polys[["geometry"]])
        except Exception:
            signal.alarm(0)
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], crs=UTM)
    water = pd.concat(frames, ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=4326).to_crs(epsg=UTM)
    water = gpd.clip(water[["geometry"]], buf)
    return water[~water.is_empty].copy()


def _apply_water_mask(points, water):
    if water.empty:
        return np.zeros(len(points), dtype=bool)
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], water[["geometry"]],
                       how="left", predicate="within")
    is_water = set(joined.dropna(subset=["index_right"]).index)
    return np.array([i in is_water for i in range(len(points))], dtype=bool)


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


def _sample_aqueduct(pts_wgs84, src):
    """
    Sample the Aqueduct raster at each point (WGS84 lon/lat).
    Returns float array of depth values (m); nodata → NaN.
    """
    import rasterio
    from rasterio.transform import rowcol

    lons = pts_wgs84.geometry.x.values
    lats = pts_wgs84.geometry.y.values

    depths = np.full(len(pts_wgs84), np.nan)
    nodata = src.nodata

    rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    h, w = src.height, src.width
    valid_mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    if valid_mask.any():
        # Read a small window covering all valid cells at once would be slow
        # for sparse points; use vectorised indexing on the full band instead.
        band = src.read(1)
        for idx in np.where(valid_mask)[0]:
            v = float(band[rows[idx], cols[idx]])
            if nodata is not None and v == nodata:
                depths[idx] = np.nan
            elif v < 0:
                depths[idx] = np.nan
            else:
                depths[idx] = v

    return depths


def _diff_bucket(diff):
    if diff <= -2: return "noah_much_higher"
    if diff == -1: return "noah_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "aq_higher"
    return "aq_much_higher"


def _consensus_bucket(noah_cls, aq_cls, water):
    if water:
        return "water"
    if noah_cls >= NOAH_ACTIVE_CLASS and aq_cls >= NOAH_ACTIVE_CLASS:
        return "confirmed"
    if noah_cls >= NOAH_ACTIVE_CLASS and aq_cls < NOAH_ACTIVE_CLASS:
        return "modelled"
    if noah_cls < NOAH_ACTIVE_CLASS and aq_cls >= NOAH_ACTIVE_CLASS:
        return "empirical"
    return "low"


# ===========================================================================
# Main
# ===========================================================================
print("=" * 72, flush=True)
print("NOAH 5-yr vs WRI Aqueduct Floods v2 (WATCH, RP-5, riverine)", flush=True)
print("=" * 72, flush=True)

# ── Step 1: Load Aqueduct raster ─────────────────────────────────────────
print("\n[1/4] Loading WRI Aqueduct raster…", flush=True)
aq_src = load_aqueduct_raster()
print(f"  CRS: {aq_src.crs}  |  shape: {aq_src.height}×{aq_src.width}  "
      f"|  nodata: {aq_src.nodata}", flush=True)

# ── Step 2: OSM water masks ───────────────────────────────────────────────
print("\n[2/4] Fetching OSM water bodies…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} polygon(s)", flush=True)

# ── Step 3: Per-city analysis ─────────────────────────────────────────────
print("\n[3/4] Building grids and comparing NOAH vs Aqueduct…", flush=True)
city_data    = {}
summary_rows = []
all_cells    = []

for city in CITIES:
    print(f"\n  {city['name']} ({city['region']})…", flush=True)
    pts, buf   = _build_grid(city)
    water      = water_by_city[city["slug"]]
    water_mask = _apply_water_mask(pts, water)

    # ── NOAH ──────────────────────────────────────────────────────────────
    noah_shp = _find_noah_shp(city["noah_province"])
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
        print(f"    WARN: no NOAH shapefile for {city['noah_province'] or 'N/A'} "
              "(city skipped for NOAH stats but kept for Aqueduct)")
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)

    noah_cls = _sample_noah(pts, noah)

    # ── Aqueduct: sample at WGS84 grid centres ────────────────────────────
    pts_wgs = pts.to_crs(epsg=4326)
    aq_depth = _sample_aqueduct(pts_wgs, aq_src)
    aq_cls   = np.array([_classify_depth(d) for d in aq_depth], dtype=int)
    print(f"    Aqueduct: {(aq_depth > 0).sum()} cells with depth > 0  |  "
          f"max depth = {np.nanmax(aq_depth):.2f} m", flush=True)

    # ── Consensus + difference ────────────────────────────────────────────
    consensus = [
        _consensus_bucket(int(noah_cls[i]), int(aq_cls[i]), bool(water_mask[i]))
        for i in range(len(pts))
    ]
    diff_buckets = [
        "water" if water_mask[i]
        else _diff_bucket(int(aq_cls[i]) - int(noah_cls[i]))
        for i in range(len(pts))
    ]

    # ── Stats ─────────────────────────────────────────────────────────────
    valid   = ~water_mask
    # For Butuan (no NOAH), correlation is not meaningful
    n_cls   = noah_cls[valid].astype(int)
    a_cls   = aq_cls[valid].astype(int)
    has_noah = (n_cls > 0).any()

    if has_noah:
        exact    = float(np.mean(n_cls == a_cls))
        within1  = float(np.mean(np.abs(n_cls - a_cls) <= 1))
        spearman = (pd.Series(n_cls).corr(pd.Series(a_cls), method="spearman")
                    if len(np.unique(n_cls)) > 1 and len(np.unique(a_cls)) > 1
                    else float("nan"))
    else:
        exact, within1, spearman = float("nan"), float("nan"), float("nan")

    cs         = pd.Series(consensus)
    conf_share = float((cs == "confirmed").mean())
    mod_share  = float((cs == "modelled").mean())
    emp_share  = float((cs == "empirical").mean())
    low_share  = float((cs == "low").mean())

    plot_df = pd.DataFrame({
        "city":        city["name"],
        "lon":         pts_wgs.geometry.x.values,
        "lat":         pts_wgs.geometry.y.values,
        "noah_cls":    noah_cls,
        "aq_depth":    aq_depth,
        "aq_cls":      aq_cls,
        "is_water":    water_mask,
        "consensus":   consensus,
        "diff_bucket": diff_buckets,
        "dist_m":      np.sqrt(
            (pts.geometry.x.values - pts.geometry.x.mean()) ** 2 +
            (pts.geometry.y.values - pts.geometry.y.mean()) ** 2
        ),
    })
    all_cells.append(plot_df)
    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    rho_str   = f"{spearman:.3f}" if pd.notna(spearman) else "—"
    exact_str = f"{exact:.3f}"   if pd.notna(exact)    else "—"
    print(f"    noah_cells={int((noah_cls>0).sum()):>5} | aq_cells={int((aq_cls>0).sum()):>5} | "
          f"exact={exact_str} | ρ={rho_str} | "
          f"conf={conf_share:.2f} mod={mod_share:.2f} emp={emp_share:.2f}", flush=True)

    city_data[city["slug"]] = {
        "city":        city,
        "plot_df":     plot_df,
        "extent":      extent,
        "n_noah_cells": int((noah_cls > 0).sum()),
        "n_aq_cells":   int((aq_cls > 0).sum()),
        "water_cells":  int(water_mask.sum()),
        "exact":        exact,
        "within_one":   within1,
        "spearman":     float(spearman) if pd.notna(spearman) else float("nan"),
        "conf_share":   conf_share,
        "mod_share":    mod_share,
        "emp_share":    emp_share,
        "low_share":    low_share,
        "aq_max_depth": float(np.nanmax(aq_depth)) if np.any(~np.isnan(aq_depth)) else 0.0,
    }
    summary_rows.append({
        "city":               city["name"],
        "region":             city["region"],
        "noah_province":      city["noah_province"] or "N/A",
        "radius_m":           city["radius_m"],
        "noah_hazard_cells":  int((noah_cls > 0).sum()),
        "aq_flooded_cells":   int((aq_cls > 0).sum()),
        "water_cells_masked": int(water_mask.sum()),
        "exact_accuracy":     exact,
        "within_one_accuracy": within1,
        "spearman":           float(spearman) if pd.notna(spearman) else float("nan"),
        "confirmed_share":    conf_share,
        "modelled_share":     mod_share,
        "empirical_share":    emp_share,
        "aq_max_depth_m":     city_data[city["slug"]]["aq_max_depth"],
    })

# Save outputs
summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(OUT_DIR, "noah_aqueduct_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}", flush=True)

cells_df   = pd.concat(all_cells, ignore_index=True)
cells_path = os.path.join(OUT_DIR, "noah_aqueduct_cells.parquet")
cells_df.to_parquet(cells_path, index=False)
print(f"  Saved → {cells_path}", flush=True)

# Close raster
aq_src.close()


# ===========================================================================
# Figure 1 — 10-city × 4-panel maps
# ===========================================================================
print("\n[4/4] Rendering figures…", flush=True)

PANEL_TITLES = [
    "",
    "NOAH 5-yr\n(modelled hazard)",
    "Aqueduct RP-5\n(riverine depth class)",
    "Consensus risk\n(NOAH + Aqueduct)",
    "Difference\n(Aqueduct class − NOAH class)",
]

depth_cmap  = plt.cm.Blues
depth_norm  = mcolors.Normalize(vmin=0, vmax=max(
    max(d["aq_max_depth"] for d in city_data.values()), 0.1
))

fig1, axes1 = plt.subplots(
    len(CITIES), 5,
    figsize=(21.0, 2.90 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.58, 1.0, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    "NOAH 5-yr Hazard vs WRI Aqueduct Floods v2  "
    "(WATCH historical, RP-5, riverine inundation depth)",
    fontsize=13, fontweight="bold", y=0.9995,
)
fig1.text(
    0.5, 0.976,
    "Aqueduct class: 0–0.5 m → Low | 0.5–1.5 m → Medium | >1.5 m → High  |  "
    "Consensus: NOAH ≥ Medium AND/OR Aqueduct ≥ Medium  |  "
    "Permanent water (OSM) masked in teal",
    ha="center", fontsize=8, color="#444444",
)

for ci, title in enumerate(PANEL_TITLES):
    axes1[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=5)

for ri, city in enumerate(CITIES):
    slug      = city["slug"]
    d         = city_data[slug]
    rho       = d["spearman"]
    rho_str   = f"{rho:.2f}"        if not np.isnan(rho)    else "—"
    exact_str = f"{d['exact']:.2f}" if pd.notna(d["exact"]) else "—"
    plot_df   = d["plot_df"]

    # --- Col 0: label ---
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"r={city['radius_m']//1000} km\n"
        f"noah_cells={d['n_noah_cells']:,}\n"
        f"aq_cells={d['n_aq_cells']:,}\n"
        f"exact={exact_str}\n"
        f"ρ={rho_str}\n\n"
        f"conf={d['conf_share']:.2f}\n"
        f"mod={d['mod_share']:.2f}\n"
        f"emp={d['emp_share']:.2f}",
        ha="center", va="center",
        fontsize=6.8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92,
                  boxstyle="round,pad=0.35"),
    )

    def _setup_ax(ax, city=city, d=d):
        ax.set_facecolor("#F1EDE5")
        ax.plot(city["lng"], city["lat"], "r*", markersize=4, zorder=5)
        ax.set_xlim(d["extent"][0], d["extent"][1])
        ax.set_ylim(d["extent"][2], d["extent"][3])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=5.0)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.grid(alpha=0.18)

    water_pts = plot_df[plot_df["is_water"]]

    # Col 1: NOAH
    ax = axes1[ri, 1]
    _setup_ax(ax)
    for klass in [0, 1, 2, 3]:
        sub = plot_df[plot_df["noah_cls"] == klass]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=4, marker="s",
                   c=NOAH_COLORS[klass], linewidths=0,
                   alpha=0.20 if klass == 0 else 0.88)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=4, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)

    # Col 2: Aqueduct depth class
    ax = axes1[ri, 2]
    _setup_ax(ax)
    for klass in [0, 1, 2, 3]:
        sub = plot_df[plot_df["aq_cls"] == klass]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=4, marker="s",
                   c=AQ_COLORS[klass], linewidths=0,
                   alpha=0.20 if klass == 0 else 0.88)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=4, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98,
            f"max={d['aq_max_depth']:.2f}m",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.12"))

    # Col 3: Consensus
    ax = axes1[ri, 3]
    _setup_ax(ax)
    for bucket in ["low", "modelled", "empirical", "confirmed", "water"]:
        sub = plot_df[plot_df["consensus"] == bucket]
        if sub.empty:
            continue
        alpha = 0.18 if bucket == "low" else 0.90
        ax.scatter(sub["lon"], sub["lat"], s=4, marker="s",
                   c=CONSENSUS_COLORS[bucket], linewidths=0, alpha=alpha)
    ax.text(0.02, 0.98,
            f"conf={d['conf_share']:.2f}\n"
            f"mod={d['mod_share']:.2f}\n"
            f"emp={d['emp_share']:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))

    # Col 4: Difference
    ax = axes1[ri, 4]
    _setup_ax(ax)
    for bucket in ["noah_much_higher", "noah_higher", "match",
                   "aq_higher", "aq_much_higher"]:
        sub = plot_df[plot_df["diff_bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=4, marker="s",
                   c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=4, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98, f"ρ={rho_str}",
            transform=ax.transAxes, va="top", fontsize=6, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))

# Legends
noah_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="NOAH Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="NOAH Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="NOAH High"),
]
aq_handles = [
    mpatches.Patch(color=AQ_COLORS[1], label="Aq. Low (0–0.5m)"),
    mpatches.Patch(color=AQ_COLORS[2], label="Aq. Medium (0.5–1.5m)"),
    mpatches.Patch(color=AQ_COLORS[3], label="Aq. High (>1.5m)"),
]
consensus_handles = [
    mpatches.Patch(color=CONSENSUS_COLORS["confirmed"], label="Confirmed risk (both)"),
    mpatches.Patch(color=CONSENSUS_COLORS["modelled"],  label="Modelled only (NOAH)"),
    mpatches.Patch(color=CONSENSUS_COLORS["empirical"], label="Empirical gap (Aq.)"),
    mpatches.Patch(color=CONSENSUS_COLORS["low"],       label="Low risk"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"], label="NOAH >> Aq."),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"],      label="NOAH > Aq."),
    mpatches.Patch(color=DIFF_COLORS["match"],            label="Match"),
    mpatches.Patch(color=DIFF_COLORS["aq_higher"],        label="Aq. > NOAH"),
    mpatches.Patch(color=DIFF_COLORS["aq_much_higher"],   label="Aq. >> NOAH"),
]
water_handle = mpatches.Patch(color=WATER_COLOR, label="Permanent water (OSM)")

all_handles = (noah_handles + aq_handles + consensus_handles
               + diff_handles + [water_handle])
fig1.legend(handles=all_handles, loc="lower center", ncol=8, fontsize=6.8,
            framealpha=0.92, bbox_to_anchor=(0.5, -0.005))

fig1.tight_layout(rect=[0, 0.025, 1, 0.970])
p1 = os.path.join(OUT_DIR, "noah_aqueduct_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# Figure 2 — Diagnostics
# ===========================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle("NOAH 5-yr vs WRI Aqueduct RP-5 — Diagnostic Statistics (10 cities)",
              fontsize=12, fontweight="bold")

city_names = [c["name"] for c in CITIES]
x          = np.arange(len(CITIES))
bar_w      = 0.35

# 2a: Spearman ρ
ax = axes2[0, 0]
rho_vals = [city_data[c["slug"]]["spearman"] for c in CITIES]
bar_colors = [CITY_COLORS[i] for i in range(len(CITIES))]
bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in rho_vals],
              color=bar_colors, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=30, ha="right", fontsize=7.5)
ax.set_title("Spearman ρ (NOAH class vs Aqueduct class)", fontweight="bold")
ax.set_ylabel("ρ")
ax.set_ylim(-0.3, 1.0)
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, rho_vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                "no NOAH", ha="center", va="bottom", fontsize=6, color="#888")

# 2b: Exact match & within-1
ax = axes2[0, 1]
exact_vals  = [city_data[c["slug"]]["exact"]      for c in CITIES]
within_vals = [city_data[c["slug"]]["within_one"] for c in CITIES]
ax.bar(x - bar_w / 2,
       [v if pd.notna(v) else 0 for v in exact_vals],
       bar_w, label="Exact match", color="#1976D2", alpha=0.85)
ax.bar(x + bar_w / 2,
       [v if pd.notna(v) else 0 for v in within_vals],
       bar_w, label="Within ±1",   color="#66BB6A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=30, ha="right", fontsize=7.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of cells")
ax.set_title("Class agreement (NOAH vs Aqueduct)", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2c: Consensus share (stacked)
ax = axes2[0, 2]
conf_vals = [city_data[c["slug"]]["conf_share"] for c in CITIES]
mod_vals  = [city_data[c["slug"]]["mod_share"]  for c in CITIES]
emp_vals  = [city_data[c["slug"]]["emp_share"]  for c in CITIES]
low_vals  = [city_data[c["slug"]]["low_share"]  for c in CITIES]
ax.bar(x, conf_vals, color=CONSENSUS_COLORS["confirmed"], alpha=0.85, label="Confirmed")
ax.bar(x, mod_vals,  color=CONSENSUS_COLORS["modelled"],  alpha=0.85, label="Modelled",
       bottom=conf_vals)
mod_b = [c + m for c, m in zip(conf_vals, mod_vals)]
ax.bar(x, emp_vals, color=CONSENSUS_COLORS["empirical"], alpha=0.85, label="Empirical gap",
       bottom=mod_b)
emp_b = [m + e for m, e in zip(mod_b, emp_vals)]
ax.bar(x, low_vals, color=CONSENSUS_COLORS["low"], alpha=0.85, label="Low risk",
       bottom=emp_b, edgecolor="#aaa", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=30, ha="right", fontsize=7.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of valid cells")
ax.set_title("Consensus risk share", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2d: NOAH cells vs Aqueduct cells scatter
ax = axes2[1, 0]
for ci, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    ax.scatter(d["n_noah_cells"], d["n_aq_cells"],
               color=CITY_COLORS[ci], s=80, zorder=3)
    ax.annotate(city["name"], (d["n_noah_cells"], d["n_aq_cells"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
max_cells = max(
    max(d["n_noah_cells"] for d in city_data.values()),
    max(d["n_aq_cells"]   for d in city_data.values()),
)
ax.plot([0, max_cells], [0, max_cells], "k--", linewidth=0.8, alpha=0.4, label="1:1")
ax.set_xlabel("NOAH hazard cells (Var > 0)")
ax.set_ylabel("Aqueduct flooded cells (depth > 0)")
ax.set_title("NOAH vs Aqueduct spatial coverage", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 2e: NOAH class vs mean Aqueduct depth
ax = axes2[1, 1]
for ci, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    pf = d["plot_df"]
    valid = ~pf["is_water"].values
    n_cls = pf["noah_cls"].values[valid]
    a_dep = np.where(np.isnan(pf["aq_depth"].values[valid]), 0,
                     pf["aq_depth"].values[valid])
    means = [a_dep[n_cls == k].mean() if (n_cls == k).any() else np.nan
             for k in [0, 1, 2, 3]]
    ax.plot([0, 1, 2, 3], means, marker="o", linewidth=2, markersize=5,
            color=CITY_COLORS[ci], label=city["name"])
ax.set_title("Mean Aqueduct depth per NOAH class", fontweight="bold")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["No hazard", "Low", "Medium", "High"], rotation=20, ha="right")
ax.set_ylabel("Mean Aqueduct depth (m)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=2)

# 2f: Summary table
ax = axes2[1, 2]
ax.axis("off")
col_labels = ["City", "NOAH\ncells", "Aq.\ncells", "Exact", "ρ", "Conf", "Mod", "Emp"]
row_data   = []
for c in CITIES:
    d = city_data[c["slug"]]
    rho = d["spearman"]
    exact_s = f"{d['exact']:.2f}" if pd.notna(d['exact']) else "—"
    rho_s   = f"{rho:.2f}" if not np.isnan(rho) else "—"
    row_data.append([
        c["name"],
        str(d["n_noah_cells"]),
        str(d["n_aq_cells"]),
        exact_s,
        rho_s,
        f"{d['conf_share']:.2f}",
        f"{d['mod_share']:.2f}",
        f"{d['emp_share']:.2f}",
    ])
tbl = ax.table(cellText=row_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.0)
tbl.scale(1.0, 1.35)
ax.set_title("Summary statistics", fontweight="bold", pad=4)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "noah_aqueduct_02_diagnostics.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
print(f"  {csv_path}", flush=True)
print(f"  {cells_path}", flush=True)
