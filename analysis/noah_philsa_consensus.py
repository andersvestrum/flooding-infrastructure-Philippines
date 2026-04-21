"""
noah_philsa_consensus.py
========================
Redesigned NOAH vs PhilSA comparison with explicit data-source framing:

  NOAH    = modelled hazard envelope (terrain + hydrology, 5-yr return period)
  PhilSA  = empirical event frequency (SAR/optical, typhoon-driven, 2022-2026)

Figures
-------
  output/noah_philsa_consensus_01_maps.png
      5-city × 5-panel:
        [labels] | [NOAH 5-yr] | [PhilSA raw frequency] | [Consensus risk] | [Difference]

      Consensus risk quadrants (per grid cell):
        Confirmed risk  – NOAH ≥ Medium  AND PhilSA frequent  → dark red
        Modelled risk   – NOAH ≥ Medium  AND PhilSA rare      → blue
        Empirical gap   – NOAH < Medium  AND PhilSA frequent  → orange
        Low risk        – NOAH < Medium  AND PhilSA rare      → light grey

      Frequency threshold for "PhilSA frequent": flooded in ≥ 15 % of
      observation dates that intersect the city buffer.

  output/noah_philsa_consensus_02_timeline.png
      Event timeline: dot strip per city, coloured by sensor, with
      approximate typhoon season shading.

Visual fixes vs the original noah_philsa_01_maps.png
  • Water (OSM) is rendered in teal (#2CBBB4) — clearly distinct from
    NOAH-higher purple (#9467BD) in the difference panel.
  • The raw-frequency panel uses a continuous sequential colormap so
    the historical nature of PhilSA is immediately legible.
"""

import glob
import os
import re
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
import osmnx as ox

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "output", "noah_validation", "philsa")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
NOAH_BASE  = os.path.join(ROOT, "data", "noah", "5yr")
os.makedirs(OUT_DIR, exist_ok=True)

UTM          = 32651
GRID_M       = 250
SAT_PRIORITY = ["s1", "s2", "rcm", "alos2", "k5", "gf3", "iceye", "nv1", "l9"]

# Threshold for "PhilSA frequent": top tertile of flooded cells (per city)
# PHILSA_FREQ_THRESH is computed per city dynamically — see _philsa_freq_thresh()
NOAH_ACTIVE_CLASS  = 2     # Var ≥ 2 (Medium or High) → "NOAH active"

CITIES = [
    {"name": "Tuguegarao",    "slug": "tuguegarao",    "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "noah_province": "Cagayan",             "region": "Cagayan Valley"},
    {"name": "Dagupan",       "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "noah_province": "Pangasinan",          "region": "Ilocos"},
    {"name": "Manila",        "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "noah_province": "Metropolitan Manila", "region": "NCR"},
    {"name": "Cagayan de Oro","slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "noah_province": "Misamis Oriental",    "region": "Mindanao"},
    {"name": "Cotabato",      "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "noah_province": "Maguindanao",         "region": "BARMM"},
]

# Colour palettes
NOAH_COLORS  = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR  = "#2CBBB4"   # teal — visually distinct from NOAH purple

# Difference panel: NOAH-higher shades changed to purple to separate from teal water
DIFF_COLORS = {
    "noah_much_higher":   "#54278F",   # deep purple
    "noah_higher":        "#9467BD",   # medium purple  ← was blue #6BAED6
    "match":              "#2E7D32",   # green
    "philsa_higher":      "#FDAE61",   # orange
    "philsa_much_higher": "#B2182B",   # dark red
}

# Consensus risk colours
CONSENSUS_COLORS = {
    "confirmed":  "#B71C1C",   # dark red  – both sources flag risk
    "modelled":   "#1565C0",   # dark blue – NOAH only, not yet observed
    "empirical":  "#E65100",   # burnt orange – PhilSA only, model gap
    "low":        "#E8E8E8",   # light grey – low risk
    "water":      WATER_COLOR,
}

SENSOR_COLORS = {
    "s1": "#1976D2", "s2": "#66BB6A", "rcm": "#8E24AA",
    "alos2": "#F57F17", "k5": "#00838F", "tdx": "#5D4037",
    "iceye": "#E53935", "saocom": "#FB8C00", "gf3": "#546E7A",
    "nv1": "#AD1457", "l9": "#558B2F",
}
CITY_COLORS  = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]


# ===========================================================================
# Data helpers (same logic as noah_philsa_comparison.py)
# ===========================================================================

def _sat_rank(sat):
    try:
        return SAT_PRIORITY.index(sat.lower())
    except ValueError:
        return len(SAT_PRIORITY)


def load_all_philsa():
    PAT = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    inventory = {}
    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str, _, sensor, _ = m.groups()
        inventory.setdefault(date_str, {}).setdefault(sensor, []).append(
            os.path.join(PHILSA_DIR, fname)
        )

    parts = []
    for date_str, sensor_map in sorted(inventory.items()):
        event_date    = pd.to_datetime(date_str, format="%Y%m%d")
        chosen_sensor = min(sensor_map.keys(), key=_sat_rank)
        event_gdfs    = []
        for fpath in sensor_map[chosen_sensor]:
            try:
                with zipfile.ZipFile(fpath) as zf:
                    shps = [n for n in zf.namelist() if n.lower().endswith(".shp")]
                    if not shps:
                        continue
                    gdf = gpd.read_file(f"zip://{fpath}!{shps[0]}")
                if gdf is None or len(gdf) == 0:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
                event_gdfs.append(gdf[["geometry"]])
            except Exception as e:
                print(f"    WARN {os.path.basename(fpath)}: {e}")
        if not event_gdfs:
            continue
        event = pd.concat(event_gdfs, ignore_index=True)
        event = gpd.GeoDataFrame(event, geometry="geometry", crs=4326)
        event["event_date"] = event_date
        event["sensor"]     = chosen_sensor
        event["event_key"]  = f"{date_str}_{chosen_sensor}"
        parts.append(event)

    all_p = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_p, geometry="geometry", crs=4326)


def _find_noah_shp(province):
    camel = province.replace(" ", "")
    for folder in [province, camel, camel.lower()]:
        base = os.path.join(NOAH_BASE, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base, "*.shp"))
            if shps:
                return shps[0]
    return None


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


def _rasterize_polygons(points, polys):
    counts = np.zeros(len(points), dtype=float)
    if polys.empty:
        return counts
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], polys[["geometry"]],
                       how="left", predicate="within")
    hit = (joined.dropna(subset=["index_right"])
               .groupby(["xi", "yi"]).size().reset_index(name="n"))
    idx_map = {(int(r.xi), int(r.yi)): i for i, r in points.reset_index().iterrows()}
    for _, row in hit.iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map:
            counts[idx_map[k]] = float(row["n"])
    return counts


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


def _diff_bucket(diff):
    if diff <= -2: return "noah_much_higher"
    if diff == -1: return "noah_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "philsa_higher"
    return "philsa_much_higher"


def _philsa_freq_thresh(philsa_freq, water_mask):
    """Per-city threshold: 67th percentile of flooded (freq > 0) non-water cells."""
    valid = philsa_freq[~water_mask]
    flooded = valid[valid > 0]
    if len(flooded) == 0:
        return 0.01
    return float(np.percentile(flooded, 67))


def _consensus_bucket(noah_cls, philsa_freq, water, thresh):
    """4-category consensus risk per cell."""
    if water:
        return "water"
    active_noah   = noah_cls >= NOAH_ACTIVE_CLASS
    active_philsa = philsa_freq >= thresh
    if active_noah and active_philsa:
        return "confirmed"
    if active_noah and not active_philsa:
        return "modelled"
    if not active_noah and active_philsa:
        return "empirical"
    return "low"


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 72, flush=True)
print("NOAH vs PhilSA — Consensus Risk Maps", flush=True)
print("=" * 72, flush=True)

print("\n[1/4] Loading PhilSA…", flush=True)
philsa_all = load_all_philsa()
philsa_utm = philsa_all.to_crs(epsg=UTM)
n_events   = philsa_all["event_key"].nunique()
date_min   = philsa_all["event_date"].min().date()
date_max   = philsa_all["event_date"].max().date()
print(f"  {len(philsa_all):,} polygons | {n_events} observation dates | "
      f"{date_min} → {date_max}", flush=True)

print("\n[2/4] OSM water masks…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} polygons", flush=True)

print("\n[3/4] Building grids…", flush=True)
city_data = {}

for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, buf = _build_grid(city)
    water    = water_by_city[city["slug"]]
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
        print(f"    WARN: no NOAH for {city['noah_province']}")
        noah = gpd.GeoDataFrame(columns=["Var", "geometry"], crs=UTM)

    noah_cls = _sample_noah(pts, noah)

    # PhilSA — count per event
    philsa_city = gpd.clip(philsa_utm[philsa_utm.intersects(buf)].copy(), buf)
    philsa_city = philsa_city[~philsa_city.is_empty].copy()

    philsa_count    = np.zeros(len(pts), dtype=float)
    n_philsa_events = 0
    event_dates_city = []
    event_sensors_city = []

    for ekey, grp in philsa_city.groupby("event_key"):
        c = _rasterize_polygons(pts, grp)
        philsa_count += (c > 0).astype(float)
        n_philsa_events += 1
        row = grp.iloc[0]
        event_dates_city.append(row["event_date"])
        event_sensors_city.append(row["sensor"])

    # Frequency as fraction of city-intersecting events
    philsa_freq = philsa_count / max(n_philsa_events, 1)

    # Tertile classification (for difference panel)
    philsa_cls = _classify_tertile(philsa_count, water_mask)

    # Consensus — threshold = 67th pct of flooded non-water cells (top tertile)
    freq_thresh = _philsa_freq_thresh(philsa_freq, water_mask)
    consensus = [
        _consensus_bucket(int(noah_cls[i]), float(philsa_freq[i]), bool(water_mask[i]), freq_thresh)
        for i in range(len(pts))
    ]

    # Difference (PHILSA tertile − NOAH class)
    diff_buckets = [
        "water" if water_mask[i]
        else _diff_bucket(int(max(0, philsa_cls[i])) - int(noah_cls[i]))
        for i in range(len(pts))
    ]

    pts_wgs = pts.to_crs(epsg=4326)
    plot_df = pd.DataFrame({
        "lon":        pts_wgs.geometry.x.values,
        "lat":        pts_wgs.geometry.y.values,
        "noah_cls":   noah_cls,
        "philsa_freq": philsa_freq,
        "philsa_cls": np.where(philsa_cls == -1, 0, philsa_cls),
        "is_water":   water_mask,
        "consensus":  consensus,
        "diff_bucket": diff_buckets,
    })

    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    # Stats
    valid    = ~water_mask
    n_cls    = noah_cls[valid]
    p_cls_s  = np.where(philsa_cls[valid] == -1, 0, philsa_cls[valid])
    exact    = float(np.mean(n_cls == p_cls_s))
    within1  = float(np.mean(np.abs(n_cls.astype(int) - p_cls_s.astype(int)) <= 1))
    spearman = pd.Series(n_cls).corr(pd.Series(philsa_count[valid]), method="spearman")

    # Consensus shares
    cs = pd.Series(consensus)
    conf_share = float((cs == "confirmed").mean())
    mod_share  = float((cs == "modelled").mean())
    emp_share  = float((cs == "empirical").mean())
    low_share  = float((cs == "low").mean())

    rho_str = f"{spearman:.3f}" if pd.notna(spearman) else "nan"
    print(f"    events={n_philsa_events:>3} | exact={exact:.3f} | rho={rho_str} | "
          f"thresh={freq_thresh:.3f} | confirmed={conf_share:.2f} "
          f"modelled={mod_share:.2f} empirical={emp_share:.2f}", flush=True)

    city_data[city["slug"]] = {
        "city":             city,
        "plot_df":          plot_df,
        "extent":           extent,
        "n_philsa_events":  n_philsa_events,
        "n_noah_cells":     int((noah_cls > 0).sum()),
        "water_cells":      int(water_mask.sum()),
        "exact":            exact,
        "within_one":       within1,
        "spearman":         float(spearman) if pd.notna(spearman) else float("nan"),
        "conf_share":       conf_share,
        "mod_share":        mod_share,
        "emp_share":        emp_share,
        "low_share":        low_share,
        "freq_thresh":      freq_thresh,
        "freq_max":         float(philsa_freq[~water_mask].max()) if (~water_mask).any() else 1.0,
        "event_dates":      event_dates_city,
        "event_sensors":    event_sensors_city,
    }


# ===========================================================================
# Figure 1 — 5-city × 5-panel maps
# ===========================================================================
print("\n[4/4] Rendering figures…", flush=True)

PANEL_TITLES = [
    "",
    "NOAH 5-yr\n(modelled hazard)",
    "PhilSA flood frequency\n(empirical, fraction of events)",
    "Consensus risk\n(NOAH + PhilSA combined)",
    "Difference\n(PhilSA tertile − NOAH class)",
]

fig1, axes1 = plt.subplots(
    len(CITIES), 5,
    figsize=(20.0, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.60, 1.0, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    f"NOAH 5-yr Hazard vs PhilSA Satellite Flood Extents  "
    f"({date_min} → {date_max},  {n_events} PhilSA observation dates)",
    fontsize=13, fontweight="bold", y=0.999,
)
fig1.text(
    0.5, 0.974,
    f"Consensus: NOAH ≥ Medium (Var ≥ {NOAH_ACTIVE_CLASS})  AND/OR  PhilSA frequent "
    f"= top tertile of flooded cells (67th pct, varies per city)  |  "
    f"Permanent water (OSM) masked in teal",
    ha="center", fontsize=8, color="#444444",
)

for ci, title in enumerate(PANEL_TITLES):
    axes1[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=5)

# Frequency colormap (norm computed per-city for better contrast)
freq_cmap = plt.cm.YlOrRd

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d    = city_data[slug]
    rho  = d["spearman"]
    rho_str = f"{rho:.2f}" if not np.isnan(rho) else "nan"
    plot_df  = d["plot_df"]

    # --- Col 0: label ---
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"philsa_dates={d['n_philsa_events']}\n"
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

    def _setup_ax(ax):
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

    # --- Col 1: NOAH ---
    ax = axes1[ri, 1]
    _setup_ax(ax)
    for klass in [0, 1, 2, 3]:
        sub = plot_df[plot_df["noah_cls"] == klass]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=NOAH_COLORS[klass], linewidths=0,
                   alpha=0.25 if klass == 0 else 0.88)
    water_pts = plot_df[plot_df["is_water"]]
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)

    # --- Col 2: PhilSA frequency (continuous, autoscaled per city) ---
    ax = axes1[ri, 2]
    _setup_ax(ax)
    city_freq_max  = max(d["freq_max"], 0.01)
    city_freq_norm = mcolors.Normalize(vmin=0, vmax=city_freq_max)
    # Non-water, non-zero frequency
    active = plot_df[~plot_df["is_water"] & (plot_df["philsa_freq"] > 0)]
    zero   = plot_df[~plot_df["is_water"] & (plot_df["philsa_freq"] == 0)]
    if not zero.empty:
        ax.scatter(zero["lon"], zero["lat"], s=6, marker="s",
                   c="#F1EDE5", linewidths=0, alpha=0.30)
    if not active.empty:
        ax.scatter(active["lon"], active["lat"], s=6, marker="s",
                   c=freq_cmap(city_freq_norm(active["philsa_freq"].values)),
                   linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(0.02, 0.98,
            f"max={city_freq_max:.2f}\nthresh={d['freq_thresh']:.2f}\n→ top tertile",
            transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
            bbox=dict(facecolor="white", alpha=0.80, boxstyle="round,pad=0.12"))

    # --- Col 3: Consensus risk ---
    ax = axes1[ri, 3]
    _setup_ax(ax)
    order = ["low", "modelled", "empirical", "confirmed", "water"]
    for bucket in order:
        sub = plot_df[plot_df["consensus"] == bucket]
        if sub.empty:
            continue
        alpha = 0.22 if bucket == "low" else 0.90
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=CONSENSUS_COLORS[bucket], linewidths=0, alpha=alpha)
    ax.text(
        0.02, 0.98,
        f"conf={d['conf_share']:.2f}\n"
        f"mod={d['mod_share']:.2f}\n"
        f"emp={d['emp_share']:.2f}",
        transform=ax.transAxes, va="top", fontsize=5.5, family="monospace",
        bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"),
    )

    # --- Col 4: Difference ---
    ax = axes1[ri, 4]
    _setup_ax(ax)
    for bucket in ["noah_much_higher", "noah_higher", "match",
                   "philsa_higher", "philsa_much_higher"]:
        sub = plot_df[plot_df["diff_bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=6, marker="s",
                   c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
    if not water_pts.empty:
        ax.scatter(water_pts["lon"], water_pts["lat"], s=6, marker="s",
                   c=WATER_COLOR, linewidths=0, alpha=0.90)
    ax.text(
        0.02, 0.98,
        f"ρ={rho_str}",
        transform=ax.transAxes, va="top", fontsize=6, family="monospace",
        bbox=dict(facecolor="white", alpha=0.82, boxstyle="round,pad=0.12"),
    )

# --- Legends ---
noah_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="NOAH Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="NOAH Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="NOAH High"),
]
consensus_handles = [
    mpatches.Patch(color=CONSENSUS_COLORS["confirmed"],  label="Confirmed risk (both)"),
    mpatches.Patch(color=CONSENSUS_COLORS["modelled"],   label="Modelled only (NOAH)"),
    mpatches.Patch(color=CONSENSUS_COLORS["empirical"],  label="Empirical gap (PhilSA)"),
    mpatches.Patch(color=CONSENSUS_COLORS["low"],        label="Low risk"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"],   label="NOAH >> PhilSA"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"],        label="NOAH > PhilSA"),
    mpatches.Patch(color=DIFF_COLORS["match"],              label="Match"),
    mpatches.Patch(color=DIFF_COLORS["philsa_higher"],      label="PhilSA > NOAH"),
    mpatches.Patch(color=DIFF_COLORS["philsa_much_higher"], label="PhilSA >> NOAH"),
]
water_handle = mpatches.Patch(color=WATER_COLOR, label="Permanent water (OSM)")

# Colorbar for frequency panel (representative 0→max scale)
sm = plt.cm.ScalarMappable(cmap=freq_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])

all_handles = noah_handles + consensus_handles + diff_handles + [water_handle]
legend = fig1.legend(
    handles=all_handles,
    loc="lower center", ncol=7, fontsize=7,
    framealpha=0.92, bbox_to_anchor=(0.5, -0.008),
)

# Add freq colorbar inline
cbar_ax = fig1.add_axes([0.41, -0.022, 0.10, 0.012])
cb = fig1.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("PhilSA flood freq.", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)

fig1.tight_layout(rect=[0, 0.03, 1, 0.968])
p1 = os.path.join(OUT_DIR, "noah_philsa_consensus_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# Figure 2 — Event timeline
# ===========================================================================
fig2, axes2 = plt.subplots(
    len(CITIES), 1,
    figsize=(14, 1.6 * len(CITIES)),
    sharex=True,
)
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle(
    "PhilSA observation dates by city buffer  "
    "(coloured by sensor — sampling is typhoon-driven, not uniform)",
    fontsize=11, fontweight="bold",
)

# Typhoon season shading: June–November each year
all_dates = pd.to_datetime(philsa_all["event_date"].unique())
year_min  = all_dates.min().year
year_max  = all_dates.max().year

for ri, city in enumerate(CITIES):
    ax  = axes2[ri]
    d   = city_data[city["slug"]]
    ax.set_facecolor("#FAFAFA")
    ax.set_yticks([])
    ax.set_ylabel(city["name"], fontsize=8, rotation=0, labelpad=60, va="center")

    # Shade typhoon season (Jun–Nov)
    for yr in range(year_min, year_max + 1):
        s_start = pd.Timestamp(f"{yr}-06-01")
        s_end   = pd.Timestamp(f"{yr}-11-30")
        ax.axvspan(s_start, s_end, color="#FFF3E0", alpha=0.55, zorder=0)

    # Plot each event as a vertical line + dot
    for date, sensor in zip(d["event_dates"], d["event_sensors"]):
        col = SENSOR_COLORS.get(sensor, "#888888")
        ax.axvline(pd.Timestamp(date), color=col, linewidth=1.8, alpha=0.85, zorder=2)
        ax.scatter([pd.Timestamp(date)], [0.5], color=col, s=30, zorder=3, linewidths=0)

    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", alpha=0.2)

    # City event count
    ax.text(0.01, 0.80, f"n = {d['n_philsa_events']} dates",
            transform=ax.transAxes, fontsize=7, va="top", color="#333")

axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes2[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes2[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

# Sensor legend
sensor_handles = [
    mpatches.Patch(color=v, label=k.upper())
    for k, v in SENSOR_COLORS.items()
    if k in philsa_all["sensor"].unique()
]
season_handle = mpatches.Patch(color="#FFF3E0", alpha=0.55, label="Typhoon season (Jun–Nov)")
fig2.legend(handles=sensor_handles + [season_handle],
            loc="lower center", ncol=8, fontsize=7.5, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.04))

fig2.tight_layout(rect=[0, 0.04, 1, 0.96])
p2 = os.path.join(OUT_DIR, "noah_philsa_consensus_02_timeline.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
