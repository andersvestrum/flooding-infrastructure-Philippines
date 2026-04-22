"""
noah_ai4g_5yr_windows.py
=========================
Compare NOAH 5-yr flood hazard against AI4G (Sentinel-1, 2014-2024) using
two windowing strategies — both averaging flood frequency across multiple
5-year windows to match NOAH's return-period framing.

Windows
-------
Non-overlapping (2 windows, aligned to Oct–Sep):
  W1  Oct 2014 – Sep 2019
  W2  Oct 2019 – Sep 2024

Overlapping, 1-year step (6 windows):
  W1  Oct 2014 – Sep 2019
  W2  Oct 2015 – Sep 2020
  W3  Oct 2016 – Sep 2021
  W4  Oct 2017 – Sep 2022
  W5  Oct 2018 – Sep 2023
  W6  Oct 2019 – Sep 2024

For each strategy the per-window flood frequency (fraction of observation
dates when a cell was flooded) is averaged across windows before comparison.

Efficiency: per city, a {date → set(cell_indices)} lookup table is built
once (one sjoin pass per date), then each window is computed by slicing
over dates — no repeated spatial joins.

Outputs
-------
  output/noah_ai4g_nonoverlap_maps.png   — Fig 1  (non-overlapping windows)
  output/noah_ai4g_overlap_maps.png      — Fig 2  (overlapping windows)
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
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(ROOT, "output", "noah_validation", "ai4g")
AI4G_DIR  = os.path.join(ROOT, "data", "ai4g")
NOAH_BASE = os.path.join(ROOT, "data", "noah", "5yr")
CACHE     = os.path.join(AI4G_DIR, "philippines_floods.parquet")
os.makedirs(OUT_DIR, exist_ok=True)

UTM    = 32651
GRID_M = 250
NOAH_ACTIVE_CLASS = 2

# 5-year windows  (start inclusive, end exclusive)
NON_OVERLAP_WINDOWS = [
    ("2014-10", "2019-10", "Oct 2014–Sep 2019"),
    ("2019-10", "2024-10", "Oct 2019–Sep 2024"),
]

OVERLAP_WINDOWS = [
    (f"{y}-10", f"{y+5}-10", f"Oct {y}–Sep {y+5}")
    for y in range(2014, 2020)       # 2014,2015,...,2019  →  6 windows
]

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

# Colours (consistent with other scripts)
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
CITY_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C",
    "#00796B", "#F57C00", "#37474F", "#880E4F", "#1B5E20",
]
WINDOW_COLORS = ["#1976D2", "#388E3C", "#7B1FA2", "#F57C00", "#D32F2F", "#00838F"]


# ===========================================================================
# Spatial helpers
# ===========================================================================

def _build_grid(city):
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
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


def _build_date_hits(df_city, pts):
    """
    One-time spatial join: builds a dict {date → frozenset(cell_indices)}.
    For each observation date, records which 250 m cells contained ≥1 detection.
    Subsequent window computations are pure dict lookups — no repeated sjoins.
    """
    if df_city.empty:
        return {}

    gdf = gpd.GeoDataFrame(
        df_city,
        geometry=gpd.points_from_xy(df_city["lon"], df_city["lat"]),
        crs=4326,
    ).to_crs(epsg=UTM)

    # Expand each grid cell to a square so nearby 20 m points fall inside
    cell_polys = pts.copy()
    cell_polys["geometry"] = cell_polys.geometry.buffer(GRID_M / 2, cap_style=3)
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in pts.reset_index().iterrows()}

    date_hits = {}
    for event_date, grp in gdf.groupby("date"):
        joined = gpd.sjoin(
            cell_polys[["xi", "yi", "geometry"]],
            grp[["geometry"]],
            how="left",
            predicate="contains",
        )
        hit = frozenset(
            idx_map[(int(r.xi), int(r.yi))]
            for _, r in joined.dropna(subset=["index_right"]).iterrows()
            if (int(r.xi), int(r.yi)) in idx_map
        )
        if hit:
            date_hits[pd.Timestamp(event_date)] = hit
    return date_hits


def _window_flood_count(date_hits, start_str, end_str, n_cells):
    """Slice date_hits to [start, end) and return (flood_count, n_dates)."""
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    flood_count = np.zeros(n_cells, dtype=float)
    n_dates = 0
    for date, cells in date_hits.items():
        if start <= date < end:
            for ci in cells:
                flood_count[ci] += 1
            n_dates += 1
    return flood_count, n_dates


# ===========================================================================
# Classification / consensus helpers
# ===========================================================================

def _freq_thresh(freq, water_mask):
    valid   = freq[~water_mask]
    flooded = valid[valid > 0]
    if len(flooded) == 0:
        return 0.01
    return float(np.percentile(flooded, 67))


def _classify_tertile(score, water_mask):
    out   = np.full(len(score), -1, dtype=int)
    valid = ~water_mask
    s     = score[valid]
    cls   = np.zeros(len(s), dtype=int)
    flooded = s > 0
    if flooded.sum() > 0:
        t33, t67 = np.percentile(s[flooded], [33, 67])
        cls[flooded & (s <= t33)] = 1
        cls[flooded & (s > t33) & (s <= t67)] = 2
        cls[flooded & (s > t67)] = 3
    out[valid] = cls
    return out


def _diff_bucket(diff):
    if diff <= -2: return "noah_much_higher"
    if diff == -1: return "noah_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "ai4g_higher"
    return "ai4g_much_higher"


def _consensus_bucket(noah_cls, ai4g_freq, water, thresh):
    if water:
        return "water"
    if noah_cls >= NOAH_ACTIVE_CLASS and ai4g_freq >= thresh: return "confirmed"
    if noah_cls >= NOAH_ACTIVE_CLASS:                         return "modelled"
    if ai4g_freq >= thresh:                                   return "empirical"
    return "low"


# ===========================================================================
# Figure helpers
# ===========================================================================

def _setup_ax(ax, city, extent):
    ax.set_facecolor("#F1EDE5")
    ax.plot(city["lng"], city["lat"], "r*", markersize=4, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5.5)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.grid(alpha=0.18)


def _scatter_freq(ax, plot_df, freq_col, freq_max, freq_thresh, freq_cmap):
    """Plot a continuous flood-frequency layer."""
    freq_norm = mcolors.Normalize(vmin=0, vmax=max(freq_max, 0.01))
    zero   = plot_df[~plot_df["is_water"] & (plot_df[freq_col] == 0)]
    active = plot_df[~plot_df["is_water"] & (plot_df[freq_col] > 0)]
    water  = plot_df[plot_df["is_water"]]
    if not zero.empty:
        ax.scatter(zero["lon"],  zero["lat"],  s=6, marker="s",
                   c="#F1EDE5", linewidths=0, alpha=0.28)
    if not active.empty:
        ax.scatter(active["lon"], active["lat"], s=6, marker="s",
                   c=freq_cmap(freq_norm(active[freq_col].values)),
                   linewidths=0, alpha=0.90)
    if not water.empty:
        ax.scatter(water["lon"],  water["lat"],  s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98, f"max={freq_max:.2f}\nthr={freq_thresh:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.12"))


def _scatter_noah(ax, plot_df):
    for klass in [0, 1, 2, 3]:
        sub = plot_df[plot_df["noah_cls"] == klass]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                       c=NOAH_COLORS[klass], linewidths=0,
                       alpha=0.25 if klass == 0 else 0.88)
    w = plot_df[plot_df["is_water"]]
    if not w.empty:
        ax.scatter(w["lon"], w["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)


def _scatter_consensus(ax, plot_df, conf, mod, emp):
    for bucket in ["low", "modelled", "empirical", "confirmed", "water"]:
        sub = plot_df[plot_df["consensus"] == bucket]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=CONSENSUS_COLORS[bucket], linewidths=0,
                   alpha=0.22 if bucket == "low" else 0.90)
    ax.text(0.02, 0.98, f"conf={conf:.2f}\nmod={mod:.2f}\nemp={emp:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))


def _scatter_diff(ax, plot_df, rho_str):
    for bucket in ["noah_much_higher", "noah_higher", "match",
                   "ai4g_higher", "ai4g_much_higher"]:
        sub = plot_df[plot_df["diff_bucket"] == bucket]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                       c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
    w = plot_df[plot_df["is_water"]]
    if not w.empty:
        ax.scatter(w["lon"], w["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98, f"ρ={rho_str}",
            transform=ax.transAxes, va="top", fontsize=6, family="monospace",
            bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"))


def _shared_legends(fig):
    """Attach NOAH, consensus, diff, and frequency-colorbar legends."""
    noah_h = [
        mpatches.Patch(color=NOAH_COLORS[1], label="NOAH Low"),
        mpatches.Patch(color=NOAH_COLORS[2], label="NOAH Medium"),
        mpatches.Patch(color=NOAH_COLORS[3], label="NOAH High"),
    ]
    cons_h = [
        mpatches.Patch(color=CONSENSUS_COLORS["confirmed"], label="Confirmed (both)"),
        mpatches.Patch(color=CONSENSUS_COLORS["modelled"],  label="Modelled only"),
        mpatches.Patch(color=CONSENSUS_COLORS["empirical"], label="Empirical gap"),
        mpatches.Patch(color=CONSENSUS_COLORS["low"],       label="Low risk"),
    ]
    diff_h = [
        mpatches.Patch(color=DIFF_COLORS["noah_much_higher"], label="NOAH >> AI4G"),
        mpatches.Patch(color=DIFF_COLORS["noah_higher"],      label="NOAH > AI4G"),
        mpatches.Patch(color=DIFF_COLORS["match"],            label="Match"),
        mpatches.Patch(color=DIFF_COLORS["ai4g_higher"],      label="AI4G > NOAH"),
        mpatches.Patch(color=DIFF_COLORS["ai4g_much_higher"], label="AI4G >> NOAH"),
    ]
    water_h = mpatches.Patch(color=WATER_COLOR, label="Permanent water")
    fig.legend(handles=noah_h + cons_h + diff_h + [water_h],
               loc="lower center", ncol=7, fontsize=7,
               framealpha=0.92, bbox_to_anchor=(0.5, -0.008))
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.41, -0.022, 0.10, 0.012])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("AI4G freq. (avg)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)


# ===========================================================================
# Main
# ===========================================================================
print("=" * 72, flush=True)
print("NOAH 5-yr vs AI4G — 5-year window analysis", flush=True)
print("=" * 72, flush=True)

# ── Load AI4G cache ──────────────────────────────────────────────────────
print(f"\nLoading AI4G cache from {CACHE} …", flush=True)
ai4g_df = pd.read_parquet(CACHE)
print(f"  {len(ai4g_df):,} detections | {ai4g_df['date'].nunique():,} unique dates", flush=True)

# ── OSM water masks (reuse from prior runs if possible — just recompute) ─
print("\nFetching OSM water bodies…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} polygon(s)", flush=True)

# ── Per-city precomputation ──────────────────────────────────────────────
print("\nPrecomputing grid hits and NOAH samples…", flush=True)
city_base = {}   # slug → {pts, buf, water_mask, noah_cls, extent, date_hits, pts_wgs}

for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, buf   = _build_grid(city)
    water      = water_by_city[city["slug"]]
    water_mask = _apply_water_mask(pts, water)

    # NOAH
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

    # AI4G detections filtered to city bbox
    b_wgs = gpd.GeoSeries([buf], crs=UTM).to_crs(4326).iloc[0].bounds
    df_city = ai4g_df[
        (ai4g_df["lat"] >= b_wgs[1]) & (ai4g_df["lat"] <= b_wgs[3]) &
        (ai4g_df["lon"] >= b_wgs[0]) & (ai4g_df["lon"] <= b_wgs[2])
    ].copy()
    print(f"    {len(df_city):,} detections — building date-hit lookup…", flush=True)

    date_hits = _build_date_hits(df_city, pts)
    print(f"    {len(date_hits)} unique hit-dates", flush=True)

    pts_wgs = pts.to_crs(epsg=4326)
    extent  = [float(pts_wgs.geometry.x.min()), float(pts_wgs.geometry.x.max()),
               float(pts_wgs.geometry.y.min()), float(pts_wgs.geometry.y.max())]

    city_base[city["slug"]] = {
        "pts":        pts,
        "pts_wgs":    pts_wgs,
        "water_mask": water_mask,
        "noah_cls":   noah_cls,
        "date_hits":  date_hits,
        "extent":     extent,
        "n_cells":    len(pts),
        "n_noah_cells": int((noah_cls > 0).sum()),
    }


# ===========================================================================
# Core function: compute averaged flood frequency over a list of windows
# ===========================================================================

def compute_window_avg(slug, windows):
    """
    Returns (avg_freq, per_window_freqs, n_dates_per_window) for a city.
    avg_freq[i] = mean of (flood_count[i] / n_dates) across windows.
    """
    base    = city_base[slug]
    n_cells = base["n_cells"]

    per_window_freqs = []
    n_dates_per_win  = []

    for start_str, end_str, label in windows:
        flood_count, n_dates = _window_flood_count(
            base["date_hits"], start_str, end_str, n_cells
        )
        freq = flood_count / max(n_dates, 1)
        per_window_freqs.append(freq)
        n_dates_per_win.append(n_dates)

    avg_freq = np.mean(per_window_freqs, axis=0)
    return avg_freq, per_window_freqs, n_dates_per_win


# ===========================================================================
# Build plot DataFrames for both strategies
# ===========================================================================

def build_city_plot_data(slug, windows, city):
    base       = city_base[slug]
    water_mask = base["water_mask"]
    noah_cls   = base["noah_cls"]
    pts_wgs    = base["pts_wgs"]

    avg_freq, per_win_freqs, n_dates_list = compute_window_avg(slug, windows)

    ai4g_cls = _classify_tertile(avg_freq, water_mask)
    thresh   = _freq_thresh(avg_freq, water_mask)

    consensus = [
        _consensus_bucket(int(noah_cls[i]), float(avg_freq[i]),
                          bool(water_mask[i]), thresh)
        for i in range(base["n_cells"])
    ]
    diff_buckets = [
        "water" if water_mask[i]
        else _diff_bucket(int(max(0, ai4g_cls[i])) - int(noah_cls[i]))
        for i in range(base["n_cells"])
    ]

    valid    = ~water_mask
    n_cls    = noah_cls[valid]
    a_cls    = np.where(ai4g_cls[valid] == -1, 0, ai4g_cls[valid])
    exact    = float(np.mean(n_cls == a_cls))
    within1  = float(np.mean(np.abs(n_cls.astype(int) - a_cls.astype(int)) <= 1))
    spearman = pd.Series(n_cls).corr(pd.Series(avg_freq[valid]), method="spearman")

    cs         = pd.Series(consensus)
    conf_share = float((cs == "confirmed").mean())
    mod_share  = float((cs == "modelled").mean())
    emp_share  = float((cs == "empirical").mean())
    freq_max   = float(avg_freq[valid].max()) if valid.any() else 1.0

    rho_str = f"{spearman:.2f}" if pd.notna(spearman) else "nan"

    plot_df = pd.DataFrame({
        "lon":         pts_wgs.geometry.x.values,
        "lat":         pts_wgs.geometry.y.values,
        "noah_cls":    noah_cls,
        "avg_freq":    avg_freq,
        "spread":      np.max(per_win_freqs, axis=0) - np.min(per_win_freqs, axis=0),
        "ai4g_cls":    np.where(ai4g_cls == -1, 0, ai4g_cls),
        "is_water":    water_mask,
        "consensus":   consensus,
        "diff_bucket": diff_buckets,
    })
    # Add per-window frequency columns
    for wi, (start_str, end_str, label) in enumerate(windows):
        plot_df[f"win_{wi}_freq"] = per_win_freqs[wi]

    return {
        "plot_df":      plot_df,
        "per_win_freqs": per_win_freqs,
        "n_dates_list": n_dates_list,
        "exact":        exact,
        "within_one":   within1,
        "spearman":     float(spearman) if pd.notna(spearman) else float("nan"),
        "rho_str":      rho_str,
        "conf_share":   conf_share,
        "mod_share":    mod_share,
        "emp_share":    emp_share,
        "freq_max":     freq_max,
        "freq_thresh":  thresh,
    }


# ===========================================================================
# Figure builder
# ===========================================================================

def make_figure(strategy_name, windows, out_path):
    """
    5-city figure.  Columns:
      0: labels
      1: NOAH 5-yr
      2: Window freq per window  (one panel if 2 windows, shows W1/W2/avg as
         annotations; for 6 windows shows W1-W6 as small multiples in one axis)
      3: Average flood freq
      4: Consensus vs NOAH
      5: Difference (AI4G tertile − NOAH class)
    """
    n_wins   = len(windows)
    freq_cmap = plt.cm.YlOrRd

    # Column layout: [label, NOAH, W1, W2, ..., avg, consensus, diff]
    # For 2 windows: [label, NOAH, W1, W2, avg, consensus, diff]  → 7 cols
    # For 6 windows: showing all 6 would be too wide; show W1, W6, avg, consensus, diff → 6 cols
    # Instead: always use 6 content columns but adapt:
    # Non-overlap (2 win): label | NOAH | W1 | W2 | Avg | Consensus | Diff
    # Overlap    (6 win):  label | NOAH | W1 (earliest) | W6 (latest) | Avg | Consensus | Diff
    #   + annotate "spread" on avg panel

    n_cols   = 7   # label + 6 content
    col_w    = [0.55] + [1.0] * 6
    fig, axes = plt.subplots(
        len(CITIES), n_cols,
        figsize=(24, 3.2 * len(CITIES)),
        gridspec_kw={"width_ratios": col_w},
    )
    fig.patch.set_facecolor("#F7F7F7")

    if n_wins == 2:
        win_a_label, win_b_label = windows[0][2], windows[1][2]
        panel_titles = [
            "",
            f"NOAH 5-yr\n(modelled hazard)",
            f"AI4G freq\n{win_a_label}",
            f"AI4G freq\n{win_b_label}",
            f"Average freq\n({n_wins} windows)",
            f"Consensus risk\n(NOAH + avg AI4G)",
            f"Difference\n(avg AI4G − NOAH)",
        ]
    else:  # 6 overlapping windows
        win_a_label, win_b_label = windows[0][2], windows[-1][2]
        panel_titles = [
            "",
            f"NOAH 5-yr\n(modelled hazard)",
            f"AI4G freq\nEarliest: {win_a_label}",
            f"AI4G freq\nLatest: {win_b_label}",
            f"Average freq\n({n_wins} windows)",
            f"Consensus risk\n(NOAH + avg AI4G)",
            f"Difference\n(avg AI4G − NOAH)",
        ]

    fig.suptitle(
        f"NOAH 5-yr vs AI4G Sentinel-1 — {strategy_name} 5-year windows",
        fontsize=13, fontweight="bold", y=0.999,
    )
    subtitle = (
        f"{n_wins} windows  |  per-window freq = (flooded dates) / (obs. dates in window)  |  "
        f"average = mean across windows  |  consensus threshold = 67th pct of flooded cells"
    )
    fig.text(0.5, 0.974, subtitle, ha="center", fontsize=8, color="#444444")

    for ci, title in enumerate(panel_titles):
        axes[0, ci].set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    for ri, city in enumerate(CITIES):
        slug = city["slug"]
        base = city_base[slug]
        d    = build_city_plot_data(slug, windows, city)
        plot_df = d["plot_df"]
        extent  = base["extent"]
        rho_str = d["rho_str"]

        # --- Col 0: label ---
        lax = axes[ri, 0]
        lax.axis("off")
        n_dates_str = "/".join(str(n) for n in d["n_dates_list"])
        lax.text(
            0.5, 0.5,
            f"{city['name']}\n{city['region']}\n"
            f"obs_dates\n{n_dates_str}\n"
            f"noah={base['n_noah_cells']:,}\n"
            f"exact={d['exact']:.2f}\nρ={rho_str}\n\n"
            f"conf={d['conf_share']:.2f}\n"
            f"mod={d['mod_share']:.2f}\n"
            f"emp={d['emp_share']:.2f}",
            ha="center", va="center",
            fontsize=6.8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.92,
                      boxstyle="round,pad=0.3"),
        )

        # Helper to set up axes quickly
        def sa(ax, city=city, extent=extent):
            _setup_ax(ax, city, extent)

        # --- Col 1: NOAH ---
        ax = axes[ri, 1]
        sa(ax)
        _scatter_noah(ax, plot_df)

        # --- Col 2: Earliest window frequency ---
        ax = axes[ri, 2]
        sa(ax)
        f0    = plot_df["win_0_freq"]
        fmax0 = float(f0[~plot_df["is_water"]].max()) if (~plot_df["is_water"]).any() else 0.01
        thr0  = _freq_thresh(f0.values, plot_df["is_water"].values)
        tmp   = plot_df.copy()
        tmp["_f"] = f0
        _scatter_freq(ax, tmp, "_f", fmax0, thr0, freq_cmap)

        # --- Col 3: Latest window frequency ---
        ax = axes[ri, 3]
        sa(ax)
        last_col = f"win_{n_wins-1}_freq"
        fn    = plot_df[last_col]
        fmaxn = float(fn[~plot_df["is_water"]].max()) if (~plot_df["is_water"]).any() else 0.01
        thrn  = _freq_thresh(fn.values, plot_df["is_water"].values)
        tmp2  = plot_df.copy()
        tmp2["_f"] = fn
        _scatter_freq(ax, tmp2, "_f", fmaxn, thrn, freq_cmap)

        # --- Col 4: Average frequency ---
        ax = axes[ri, 4]
        sa(ax)
        spread_max = float(plot_df["spread"][~plot_df["is_water"]].max()) if (~plot_df["is_water"]).any() else 0.0
        _scatter_freq(ax, plot_df, "avg_freq", d["freq_max"], d["freq_thresh"], freq_cmap)
        # Overlay spread info
        ax.text(0.02, 0.02,
                f"spread_max={spread_max:.2f}",
                transform=ax.transAxes, va="bottom", fontsize=5.0, family="monospace",
                bbox=dict(facecolor="white", alpha=0.78, boxstyle="round,pad=0.1"))

        # --- Col 5: Consensus ---
        ax = axes[ri, 5]
        sa(ax)
        _scatter_consensus(ax, plot_df, d["conf_share"], d["mod_share"], d["emp_share"])

        # --- Col 6: Difference ---
        ax = axes[ri, 6]
        sa(ax)
        _scatter_diff(ax, plot_df, rho_str)

    _shared_legends(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.968])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}", flush=True)


# ===========================================================================
# Run both strategies
# ===========================================================================
print("\n[Map 1] Non-overlapping windows…", flush=True)
p1 = os.path.join(OUT_DIR, "noah_ai4g_nonoverlap_maps.png")
make_figure("Non-overlapping", NON_OVERLAP_WINDOWS, p1)

print("\n[Map 2] Overlapping windows (1-year step)…", flush=True)
p2 = os.path.join(OUT_DIR, "noah_ai4g_overlap_maps.png")
make_figure("Overlapping", OVERLAP_WINDOWS, p2)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
