"""
noah_ai4g_comparison.py
========================
Compare NOAH 5-year flood hazard maps against the AI for Good Lab
ai4g-flood-dataset (Sentinel-1 SAR, Oct 2014 – Sep 2024) across five
Philippine cities.

Data sources
------------
NOAH  — static modelled hazard (Var 1/2/3 = Low/Medium/High, 5-yr return period)
AI4G  — individual flood pixel detections from Sentinel-1 SAR, downloaded as
        per-tile Parquet files from "ai-for-good-lab/ai4g-flood-dataset"
        Paper: "Mapping global floods with 10 years of satellite radar data",
        Nature Communications 2025.  Tiles are 3°×3°, 20 m resolution.

Method
------
1. Compute the 3°×3° tile name(s) covering each city (e.g. N15E120).
2. Download the tile's Parquet via huggingface_hub; filter edge_false_positives=0.
3. Cache filtered Philippines data to data/ai4g/philippines_floods.parquet.
4. Snap detections to a 250 m UTM grid per city; count unique flood-event dates
   per cell (same treatment as PhilSA in noah_philsa_consensus.py).
5. Classify frequency into Low/Medium/High by tertile of ever-flooded cells.
6. Load NOAH 5-yr shapefile; sample max Var per grid cell.
7. Produce consensus risk maps and difference maps vs NOAH.

Tile naming: SW corner rounded to nearest 3° (e.g. lat 14.6 → 12, lon 120.9 → 120).

Outputs
-------
  output/noah_ai4g_01_maps.png        — 5-city × 5-panel maps
  output/noah_ai4g_02_diagnostics.png — statistics summary
  output/noah_ai4g_summary.csv
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point
import osmnx as ox

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(ROOT, "output", "noah_validation", "ai4g")
AI4G_DIR  = os.path.join(ROOT, "data", "ai4g")
NOAH_BASE = os.path.join(ROOT, "data", "noah", "5yr")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AI4G_DIR, exist_ok=True)

CACHE_PATH = os.path.join(AI4G_DIR, "philippines_floods.parquet")

UTM    = 32651    # WGS84 / UTM Zone 51N — covers the Philippines
GRID_M = 250

NOAH_ACTIVE_CLASS = 2   # Var ≥ 2 (Medium or High) counts as "NOAH active"

CITIES = [
    {"name": "Tuguegarao",     "slug": "tuguegarao",    "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan",             "region": "Cagayan Valley"},
    {"name": "Dagupan",        "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan",          "region": "Ilocos"},
    {"name": "Manila",         "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Cagayan de Oro", "slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental",    "region": "Mindanao"},
    {"name": "Cotabato",       "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao",         "region": "BARMM"},
]

# Colour palettes (consistent with other scripts in this project)
NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR = "#2CBBB4"

DIFF_COLORS = {
    "noah_much_higher": "#54278F",
    "noah_higher":      "#9467BD",
    "match":            "#2E7D32",
    "ai4g_higher":      "#FDAE61",
    "ai4g_much_higher": "#B2182B",
}

CONSENSUS_COLORS = {
    "confirmed": "#B71C1C",
    "modelled":  "#1565C0",
    "empirical": "#E65100",
    "low":       "#E8E8E8",
    "water":     WATER_COLOR,
}

CITY_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]


# ===========================================================================
# Step 1 — Download & cache AI4G Philippines parquet tiles
# ===========================================================================

def _tile_name(lat, lon):
    """Return the 3°-tile SW corner name for a given coordinate, e.g. 'N15E120'."""
    lat_base = int(np.floor(lat / 3)) * 3
    lon_base = int(np.floor(lon / 3)) * 3
    lat_str  = f"N{lat_base:02d}" if lat_base >= 0 else f"S{-lat_base:02d}"
    lon_str  = f"E{lon_base:03d}" if lon_base >= 0 else f"W{-lon_base:03d}"
    return f"{lat_str}{lon_str}"


def load_ai4g_philippines():
    """
    Download the per-tile Parquet files from HuggingFace for all tiles that
    cover the five study cities.  Returns a DataFrame with columns:
      date (datetime64), lat, lon.

    A local cache at data/ai4g/philippines_floods.parquet is written so
    subsequent runs skip the download.
    """
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached AI4G data from {CACHE_PATH} …", flush=True)
        df = pd.read_parquet(CACHE_PATH)
        print(f"  {len(df):,} detections | {df['date'].nunique()} unique dates "
              f"| {df['tile'].nunique()} tile(s)", flush=True)
        return df

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required: pip install huggingface_hub")

    # Unique tiles needed
    tiles_needed = set()
    for city in CITIES:
        tiles_needed.add(_tile_name(city["lat"], city["lng"]))
    print(f"  Tiles needed: {sorted(tiles_needed)}", flush=True)

    parts = []
    for tile in sorted(tiles_needed):
        lat_prefix = tile[:3]   # e.g. "N15"
        repo_path  = f"{lat_prefix}/{tile}/{tile}-post-processing.parquet"
        print(f"  Downloading {repo_path} …", end=" ", flush=True)
        try:
            local_path = hf_hub_download(
                repo_id="ai-for-good-lab/ai4g-flood-dataset",
                filename=repo_path,
                repo_type="dataset",
                local_dir=AI4G_DIR,
            )
            df_tile = pd.read_parquet(local_path)
            print(f"{len(df_tile):,} rows", flush=True)
        except Exception as e:
            print(f"WARN: {e}", flush=True)
            continue

        # Keep only non-false-positive detections
        if "edge_false_positives" in df_tile.columns:
            df_tile = df_tile[df_tile["edge_false_positives"] == 0]

        df_tile["tile"] = tile
        df_tile["date"] = pd.to_datetime(df_tile[["year", "month", "day"]])
        parts.append(df_tile[["date", "lat", "lon", "tile"]])

    if not parts:
        raise RuntimeError("No AI4G tiles downloaded successfully.")

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached → {CACHE_PATH}  ({len(df):,} rows after FP filter)", flush=True)
    return df


# ===========================================================================
# Grid / NOAH / water helpers
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


def _snap_to_grid(df_city, pts):
    """
    Snap AI4G point detections (lat/lon at ~20 m) to a 250 m UTM grid.

    Strategy: per event-date, any cell that contains ≥ 1 detection is
    marked flooded (binary).  Sum across dates → flood count per cell.

    Returns (flood_count array, n_event_dates).
    """
    if df_city.empty:
        return np.zeros(len(pts), dtype=float), 0

    # Convert detections to UTM
    gdf = gpd.GeoDataFrame(
        df_city,
        geometry=gpd.points_from_xy(df_city["lon"], df_city["lat"]),
        crs=4326,
    ).to_crs(epsg=UTM)

    # Expand each grid cell to a 250 m square so nearby points fall inside
    cell_polys = pts.copy()
    cell_polys["geometry"] = cell_polys.geometry.buffer(GRID_M / 2, cap_style=3)

    idx_map     = {(int(r.xi), int(r.yi)): i for i, r in pts.reset_index().iterrows()}
    flood_count = np.zeros(len(pts), dtype=float)
    n_dates     = 0

    for event_date, grp in gdf.groupby("date"):
        joined = gpd.sjoin(
            cell_polys[["xi", "yi", "geometry"]],
            grp[["geometry"]],
            how="left",
            predicate="contains",
        )
        hit_cells = set(
            (int(r.xi), int(r.yi))
            for _, r in joined.dropna(subset=["index_right"]).iterrows()
        )
        for cell in hit_cells:
            if cell in idx_map:
                flood_count[idx_map[cell]] += 1
        n_dates += 1

    return flood_count, n_dates


def _classify_tertile(score, water_mask):
    out = np.full(len(score), -1, dtype=int)
    valid = ~water_mask
    s   = score[valid]
    cls = np.zeros(len(s), dtype=int)
    flooded = s > 0
    if flooded.sum() > 0:
        t33, t67 = np.percentile(s[flooded], [33, 67])
        cls[flooded & (s <= t33)] = 1
        cls[flooded & (s > t33) & (s <= t67)] = 2
        cls[flooded & (s > t67)] = 3
    out[valid] = cls
    return out


def _freq_thresh(freq, water_mask):
    """67th percentile of flooded non-water cells (top tertile threshold)."""
    valid   = freq[~water_mask]
    flooded = valid[valid > 0]
    if len(flooded) == 0:
        return 0.01
    return float(np.percentile(flooded, 67))


def _diff_bucket(diff):
    if diff <= -2: return "noah_much_higher"
    if diff == -1: return "noah_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "ai4g_higher"
    return "ai4g_much_higher"


def _consensus_bucket(noah_cls, ai4g_freq, water, thresh):
    if water:
        return "water"
    active_noah = noah_cls >= NOAH_ACTIVE_CLASS
    active_ai4g = ai4g_freq >= thresh
    if active_noah and active_ai4g:   return "confirmed"
    if active_noah and not active_ai4g: return "modelled"
    if not active_noah and active_ai4g: return "empirical"
    return "low"


# ===========================================================================
# Main
# ===========================================================================
print("=" * 72, flush=True)
print("NOAH 5-yr vs AI4G Sentinel-1 SAR — flood comparison", flush=True)
print("=" * 72, flush=True)

# ── Step 1: Load / download AI4G ─────────────────────────────────────────
print("\n[1/4] Loading AI4G flood detections…", flush=True)
ai4g_df  = load_ai4g_philippines()
date_min = ai4g_df["date"].min().date()
date_max = ai4g_df["date"].max().date()
n_dates  = ai4g_df["date"].nunique()
print(f"  Date range: {date_min} → {date_max}  |  unique dates: {n_dates:,}", flush=True)

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
print("\n[3/4] Building grids and comparing NOAH vs AI4G…", flush=True)
city_data    = {}
summary_rows = []

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
        print(f"    WARN: no NOAH shapefile for {city['noah_province']}")
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)

    noah_cls = _sample_noah(pts, noah)

    # ── AI4G — bbox filter then grid snap ─────────────────────────────────
    b_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(4326).iloc[0].bounds
    df_city = ai4g_df[
        (ai4g_df["lat"] >= b_wgs[1]) & (ai4g_df["lat"] <= b_wgs[3]) &
        (ai4g_df["lon"] >= b_wgs[0]) & (ai4g_df["lon"] <= b_wgs[2])
    ].copy()
    print(f"    AI4G detections in city bbox: {len(df_city):,}", flush=True)

    flood_count, n_ai4g_dates = _snap_to_grid(df_city, pts)
    ai4g_freq   = flood_count / max(n_ai4g_dates, 1)
    ai4g_cls    = _classify_tertile(flood_count, water_mask)
    thresh      = _freq_thresh(ai4g_freq, water_mask)

    consensus = [
        _consensus_bucket(int(noah_cls[i]), float(ai4g_freq[i]),
                          bool(water_mask[i]), thresh)
        for i in range(len(pts))
    ]

    diff_buckets = [
        "water" if water_mask[i]
        else _diff_bucket(int(max(0, ai4g_cls[i])) - int(noah_cls[i]))
        for i in range(len(pts))
    ]

    # Stats
    valid    = ~water_mask
    n_cls    = noah_cls[valid]
    a_cls    = np.where(ai4g_cls[valid] == -1, 0, ai4g_cls[valid])
    exact    = float(np.mean(n_cls == a_cls))
    within1  = float(np.mean(np.abs(n_cls.astype(int) - a_cls.astype(int)) <= 1))
    spearman = pd.Series(n_cls).corr(pd.Series(flood_count[valid]), method="spearman")
    cs       = pd.Series(consensus)
    conf_share = float((cs == "confirmed").mean())
    mod_share  = float((cs == "modelled").mean())
    emp_share  = float((cs == "empirical").mean())
    low_share  = float((cs == "low").mean())
    freq_max   = float(ai4g_freq[valid].max()) if valid.any() else 1.0

    pts_wgs  = pts.to_crs(epsg=4326)
    plot_df  = pd.DataFrame({
        "lon":         pts_wgs.geometry.x.values,
        "lat":         pts_wgs.geometry.y.values,
        "noah_cls":    noah_cls,
        "ai4g_freq":   ai4g_freq,
        "ai4g_cls":    np.where(ai4g_cls == -1, 0, ai4g_cls),
        "is_water":    water_mask,
        "consensus":   consensus,
        "diff_bucket": diff_buckets,
    })
    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    rho_str = f"{spearman:.3f}" if pd.notna(spearman) else "nan"
    print(f"    ai4g_dates={n_ai4g_dates:>4} | noah_cells={int((noah_cls>0).sum()):>5} | "
          f"exact={exact:.3f} | ρ={rho_str} | "
          f"confirmed={conf_share:.2f} mod={mod_share:.2f} emp={emp_share:.2f}", flush=True)

    city_data[city["slug"]] = {
        "city":         city,
        "plot_df":      plot_df,
        "extent":       extent,
        "n_ai4g_dates": n_ai4g_dates,
        "n_noah_cells": int((noah_cls > 0).sum()),
        "water_cells":  int(water_mask.sum()),
        "exact":        exact,
        "within_one":   within1,
        "spearman":     float(spearman) if pd.notna(spearman) else float("nan"),
        "conf_share":   conf_share,
        "mod_share":    mod_share,
        "emp_share":    emp_share,
        "low_share":    low_share,
        "freq_thresh":  thresh,
        "freq_max":     freq_max,
        "mean_ai4g_by_noah": {
            k: float(flood_count[valid][n_cls == k].mean())
            if (n_cls == k).any() else float("nan")
            for k in [0, 1, 2, 3]
        },
    }
    summary_rows.append({
        "city":                city["name"],
        "ai4g_event_dates":    n_ai4g_dates,
        "ai4g_date_range":     f"{date_min} → {date_max}",
        "noah_hazard_cells":   int((noah_cls > 0).sum()),
        "water_cells_masked":  int(water_mask.sum()),
        "exact_accuracy":      exact,
        "within_one_accuracy": within1,
        "spearman":            float(spearman) if pd.notna(spearman) else float("nan"),
        "confirmed_share":     conf_share,
        "modelled_share":      mod_share,
        "empirical_share":     emp_share,
    })

summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(OUT_DIR, "noah_ai4g_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}", flush=True)


# ===========================================================================
# Figure 1 — 5-city × 5-panel maps
# ===========================================================================
print("\n[4/4] Rendering figures…", flush=True)

PANEL_TITLES = [
    "",
    "NOAH 5-yr\n(modelled hazard)",
    "AI4G flood frequency\n(Sentinel-1, fraction of dates)",
    "Consensus risk\n(NOAH + AI4G combined)",
    "Difference\n(AI4G tertile − NOAH class)",
]

freq_cmap = plt.cm.YlOrRd

fig1, axes1 = plt.subplots(
    len(CITIES), 5,
    figsize=(20.0, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.60, 1.0, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    f"NOAH 5-yr Hazard vs AI4G Sentinel-1 Flood Detections  "
    f"({date_min} → {date_max},  {n_dates:,} observation dates)",
    fontsize=13, fontweight="bold", y=0.999,
)
fig1.text(
    0.5, 0.974,
    f"Consensus: NOAH ≥ Medium (Var ≥ {NOAH_ACTIVE_CLASS})  AND/OR  AI4G frequent "
    f"= top tertile of flooded cells (67th pct, per city)  |  "
    f"Permanent water (OSM) masked in teal  |  Edge false positives excluded",
    ha="center", fontsize=8, color="#444444",
)

for ci, title in enumerate(PANEL_TITLES):
    axes1[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=5)

for ri, city in enumerate(CITIES):
    slug    = city["slug"]
    d       = city_data[slug]
    rho     = d["spearman"]
    rho_str = f"{rho:.2f}" if not np.isnan(rho) else "nan"
    plot_df = d["plot_df"]

    # --- Col 0: label ---
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"ai4g_dates={d['n_ai4g_dates']}\n"
        f"noah_cells={d['n_noah_cells']:,}\n"
        f"exact={d['exact']:.2f}\nwithin1={d['within_one']:.2f}\nρ={rho_str}\n\n"
        f"confirmed={d['conf_share']:.2f}\n"
        f"modelled={d['mod_share']:.2f}\n"
        f"empirical={d['emp_share']:.2f}",
        ha="center", va="center",
        fontsize=7.2, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92,
                  boxstyle="round,pad=0.35"),
    )

    def _setup_ax(ax, city=city, d=d):
        ax.set_facecolor("#F1EDE5")
        ax.plot(city["lng"], city["lat"], "r*", markersize=4, zorder=5)
        ax.set_xlim(d["extent"][0], d["extent"][1])
        ax.set_ylim(d["extent"][2], d["extent"][3])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=5.5)
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
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=NOAH_COLORS[klass], linewidths=0,
                   alpha=0.25 if klass == 0 else 0.88)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)

    # Col 2: AI4G frequency (continuous colourmap)
    ax = axes1[ri, 2]
    _setup_ax(ax)
    freq_norm = mcolors.Normalize(vmin=0, vmax=max(d["freq_max"], 0.01))
    zero   = plot_df[~plot_df["is_water"] & (plot_df["ai4g_freq"] == 0)]
    active = plot_df[~plot_df["is_water"] & (plot_df["ai4g_freq"] > 0)]
    if not zero.empty:
        ax.scatter(zero["lon"], zero["lat"], s=6, marker="s",
                   c="#F1EDE5", linewidths=0, alpha=0.28)
    if not active.empty:
        ax.scatter(active["lon"], active["lat"], s=6, marker="s",
                   c=freq_cmap(freq_norm(active["ai4g_freq"].values)),
                   linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98,
            f"max={d['freq_max']:.2f}\nthresh={d['freq_thresh']:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.12"))

    # Col 3: Consensus risk
    ax = axes1[ri, 3]
    _setup_ax(ax)
    for bucket in ["low", "modelled", "empirical", "confirmed", "water"]:
        sub = plot_df[plot_df["consensus"] == bucket]
        if sub.empty:
            continue
        alpha = 0.22 if bucket == "low" else 0.90
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
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
                   "ai4g_higher", "ai4g_much_higher"]:
        sub = plot_df[plot_df["diff_bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
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
consensus_handles = [
    mpatches.Patch(color=CONSENSUS_COLORS["confirmed"], label="Confirmed risk (both)"),
    mpatches.Patch(color=CONSENSUS_COLORS["modelled"],  label="Modelled only (NOAH)"),
    mpatches.Patch(color=CONSENSUS_COLORS["empirical"], label="Empirical gap (AI4G only)"),
    mpatches.Patch(color=CONSENSUS_COLORS["low"],       label="Low risk"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"], label="NOAH >> AI4G"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"],      label="NOAH > AI4G"),
    mpatches.Patch(color=DIFF_COLORS["match"],            label="Match"),
    mpatches.Patch(color=DIFF_COLORS["ai4g_higher"],      label="AI4G > NOAH"),
    mpatches.Patch(color=DIFF_COLORS["ai4g_much_higher"], label="AI4G >> NOAH"),
]
water_handle = mpatches.Patch(color=WATER_COLOR, label="Permanent water (OSM)")

sm = plt.cm.ScalarMappable(cmap=freq_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])

all_handles = noah_handles + consensus_handles + diff_handles + [water_handle]
fig1.legend(handles=all_handles, loc="lower center", ncol=7, fontsize=7,
            framealpha=0.92, bbox_to_anchor=(0.5, -0.008))

cbar_ax = fig1.add_axes([0.41, -0.022, 0.10, 0.012])
cb = fig1.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("AI4G flood freq.", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)

fig1.tight_layout(rect=[0, 0.03, 1, 0.968])
p1 = os.path.join(OUT_DIR, "noah_ai4g_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# Figure 2 — Diagnostics
# ===========================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle("NOAH 5-yr vs AI4G Sentinel-1 — Diagnostic Statistics",
              fontsize=12, fontweight="bold")

city_names = [c["name"] for c in CITIES]
x          = np.arange(len(CITIES))
bar_w      = 0.35

# 2a: Mean AI4G event count per NOAH class
ax = axes2[0, 0]
for i, city in enumerate(CITIES):
    vals = [city_data[city["slug"]]["mean_ai4g_by_noah"][k] for k in [0, 1, 2, 3]]
    ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=2, markersize=5,
            color=CITY_COLORS[i], label=city["name"])
ax.set_title("Mean AI4G event count per NOAH class", fontweight="bold")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["No hazard", "Low", "Medium", "High"], rotation=20, ha="right")
ax.set_ylabel("Mean flood-event count")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# 2b: Exact match & within-1
ax = axes2[0, 1]
exact_vals  = [city_data[c["slug"]]["exact"]      for c in CITIES]
within_vals = [city_data[c["slug"]]["within_one"] for c in CITIES]
ax.bar(x - bar_w / 2, exact_vals,  bar_w, label="Exact match", color="#1976D2", alpha=0.85)
ax.bar(x + bar_w / 2, within_vals, bar_w, label="Within ±1",   color="#66BB6A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of cells")
ax.set_title("Class agreement (NOAH vs AI4G tertile)", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2c: Spearman ρ
ax = axes2[0, 2]
rho_vals = [city_data[c["slug"]]["spearman"] for c in CITIES]
bars = ax.bar(x, rho_vals, color=CITY_COLORS, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_title("Spearman ρ (NOAH class vs AI4G flood count)", fontweight="bold")
ax.set_ylabel("ρ")
ax.set_ylim(-1, 1)
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, rho_vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if v >= 0 else -0.08),
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

# 2d: Consensus share (stacked bar)
ax = axes2[1, 0]
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
ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of valid cells")
ax.set_title("Consensus risk share", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2e: AI4G dates vs NOAH hazard cells
ax = axes2[1, 1]
for ci, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    ax.scatter(d["n_ai4g_dates"], d["n_noah_cells"],
               color=CITY_COLORS[ci], s=80, zorder=3)
    ax.annotate(city["name"], (d["n_ai4g_dates"], d["n_noah_cells"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel("AI4G observation dates in city buffer")
ax.set_ylabel("NOAH hazard cells (Var > 0)")
ax.set_title("NOAH hazard coverage vs AI4G temporal coverage", fontweight="bold")
ax.grid(alpha=0.3)

# 2f: Summary table
ax = axes2[1, 2]
ax.axis("off")
col_labels = ["City", "AI4G\ndates", "NOAH\ncells", "Exact", "ρ", "Conf", "Mod", "Emp"]
row_data   = []
for c in CITIES:
    d = city_data[c["slug"]]
    rho = d["spearman"]
    row_data.append([
        c["name"],
        str(d["n_ai4g_dates"]),
        str(d["n_noah_cells"]),
        f"{d['exact']:.2f}",
        f"{rho:.2f}" if not np.isnan(rho) else "nan",
        f"{d['conf_share']:.2f}",
        f"{d['mod_share']:.2f}",
        f"{d['emp_share']:.2f}",
    ])
tbl = ax.table(cellText=row_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.0, 1.4)
ax.set_title("Summary statistics", fontweight="bold", pad=4)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "noah_ai4g_02_diagnostics.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
print(f"  {csv_path}", flush=True)
