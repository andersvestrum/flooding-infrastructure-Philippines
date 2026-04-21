"""
philsa_groundsource_comparison.py
===================================
Compare PhilSA satellite-derived flood extents (Sentinel-1 SAR and
other sensors, from HDX) against Google Groundsource flood observations
across five Philippine cities.

Loading rules
-------------
* File naming: YYYYMMDD_HHMM_fld_{sensor}_shp[_{region}].zip
* Same date + complementary regions  → concatenate (e.g. 20241027 cagayan-isabela +
  oriental-mindoro + region-3 are all part of the same event).
* Same date + overlapping footprints (different sensors on the same day) →
  keep only the highest-priority sensor per day (s1 > rcm > alos2 > iceye >
  nv1 > l9 > gf3 > k5 > s2).  This is assessed nationally: if the same
  sensor priority applies across the country we don't split by region.

Method
------
1. Rasterise every resolved PhilSA event onto a 250 m UTM grid per city:
   each grid cell gets a count of how many observation dates showed flooding.
2. Build a smoothed Groundsource support surface on the same grid
   (Gaussian σ = 750 m, log-normalised) summed across all years.
3. Classify PhilSA flood frequency into Low / Medium / High by tertile
   thresholds (cells that were ever flooded are split evenly).
4. Area-match Groundsource to the same Low / Medium / High class shares
   (excluding permanent-water-masked cells).
5. Difference = GS class − PhilSA class.

Outputs
-------
  output/philsa_groundsource_01_maps.png        — 5-city × 4-panel figure
  output/philsa_groundsource_02_diagnostics.png — statistics summary
  output/philsa_groundsource_summary.csv
"""

import glob
import os
import re
import warnings
import zipfile

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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "output", "noah_validation", "philsa")
PARQUET    = os.path.join(ROOT, "data", "google_gemini_flood", "groundsource_2026.parquet")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
os.makedirs(OUT_DIR, exist_ok=True)

UTM       = 32651
GRID_M    = 250
SIGMA_M   = 750
SIGMA_CELLS = SIGMA_M / GRID_M

# Satellite priority (lower index = higher priority)
SAT_PRIORITY = ["s1", "s2", "rcm", "alos2", "k5", "gf3", "iceye", "nv1", "l9"]

CITIES = [
    {"name": "Tuguegarao",    "slug": "tuguegarao",    "lat": 17.6158, "lng": 121.7229,
     "radius_m": 10_000, "region": "Cagayan Valley"},
    {"name": "Dagupan",       "slug": "dagupan",        "lat": 16.0431, "lng": 120.3333,
     "radius_m": 12_000, "region": "Ilocos"},
    {"name": "Manila",        "slug": "manila",         "lat": 14.5995, "lng": 120.9842,
     "radius_m": 20_000, "region": "NCR"},
    {"name": "Cagayan de Oro","slug": "cagayan_de_oro", "lat": 8.4772,  "lng": 124.6459,
     "radius_m": 12_000, "region": "Mindanao"},
    {"name": "Cotabato",      "slug": "cotabato",       "lat": 7.2236,  "lng": 124.2464,
     "radius_m": 10_000, "region": "BARMM"},
]

# Colour schemes (mirrors the existing NOAH-GS scripts)
CLASS_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR  = "#A8D5E2"
DIFF_COLORS  = {
    "philsa_much_higher": "#08306B",
    "philsa_higher":      "#6BAED6",
    "match":              "#2E7D32",
    "gs_higher":          "#FDAE61",
    "gs_much_higher":     "#B2182B",
}


# ===========================================================================
# 1.  Load & resolve PhilSA zips
# ===========================================================================

def _sat_rank(sat: str) -> int:
    try:
        return SAT_PRIORITY.index(sat.lower())
    except ValueError:
        return len(SAT_PRIORITY)  # unknown → lowest priority


def load_all_philsa() -> gpd.GeoDataFrame:
    """
    Parse every zip in PHILSA_DIR, resolve sensor conflicts, concatenate
    complementary regions, and return a single GeoDataFrame with columns:
      geometry, event_date (datetime), sensor, event_key (YYYYMMDD_sensor)
    """
    PAT = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    # Inventory: {date_str → {sensor → [fpath, ...]}}
    inventory: dict[str, dict[str, list[str]]] = {}

    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str, _, sensor, _ = m.groups()
        inventory.setdefault(date_str, {}).setdefault(sensor, []).append(
            os.path.join(PHILSA_DIR, fname)
        )

    parts = []
    resolved_log = []

    for date_str, sensor_map in sorted(inventory.items()):
        event_date = pd.to_datetime(date_str, format="%Y%m%d")

        # Pick the highest-priority sensor available for this date
        chosen_sensor = min(sensor_map.keys(), key=_sat_rank)
        files_to_load = sensor_map[chosen_sensor]
        skipped = [s for s in sensor_map if s != chosen_sensor]

        resolved_log.append(
            f"  {date_str}  sensor={chosen_sensor:<6}  files={len(files_to_load)}"
            + (f"  (skipped: {', '.join(skipped)})" if skipped else "")
        )

        event_gdfs = []
        for fpath in files_to_load:
            fname = os.path.basename(fpath)
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
                print(f"    WARN {fname}: {e}")

        if not event_gdfs:
            continue

        event = pd.concat(event_gdfs, ignore_index=True)
        event = gpd.GeoDataFrame(event, geometry="geometry", crs=4326)
        event["event_date"] = event_date
        event["sensor"]     = chosen_sensor
        event["event_key"]  = f"{date_str}_{chosen_sensor}"
        parts.append(event)

    print(f"  Resolved {len(parts)} PhilSA event-dates from {PHILSA_DIR}/")
    for line in resolved_log:
        print(line)

    if not parts:
        raise RuntimeError(f"No PhilSA shapefiles found in {PHILSA_DIR}")

    all_philsa = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_philsa, geometry="geometry", crs=4326)


# ===========================================================================
# 2.  Grid helpers  (same logic as existing scripts)
# ===========================================================================

def _build_grid(city: dict):
    centre = (
        gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
        .to_crs(epsg=UTM).iloc[0]
    )
    r = city["radius_m"]
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
    buf = centre.buffer(r)
    return pts, xx, yy, inside, buf


def _load_water_mask(city: dict, buf):
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


def _apply_water_mask(points: gpd.GeoDataFrame, water: gpd.GeoDataFrame) -> np.ndarray:
    if water.empty:
        return np.zeros(len(points), dtype=bool)
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], water[["geometry"]],
                       how="left", predicate="within")
    is_water_idx = set(joined.dropna(subset=["index_right"]).index)
    return np.array([i in is_water_idx for i in range(len(points))], dtype=bool)


def _rasterize_polygons(points: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> np.ndarray:
    """Count how many polygons in `polys` contain each grid point."""
    counts = np.zeros(len(points), dtype=float)
    if polys.empty:
        return counts
    joined = gpd.sjoin(points[["xi", "yi", "geometry"]], polys[["geometry"]],
                       how="left", predicate="within")
    hit = (joined.dropna(subset=["index_right"])
               .groupby(["xi", "yi"]).size().reset_index(name="n"))
    idx_map = {(int(r.xi), int(r.yi)): i
               for i, r in points.reset_index().iterrows()}
    for _, row in hit.iterrows():
        k = (int(row["xi"]), int(row["yi"]))
        if k in idx_map:
            counts[idx_map[k]] = float(row["n"])
    return counts


def _gs_counts(points: gpd.GeoDataFrame, gs: gpd.GeoDataFrame) -> np.ndarray:
    return _rasterize_polygons(points, gs)


def _classify_tertile(score: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    """
    Classify non-water cells into 0/1/2/3 using tertile thresholds on
    cells that were ever flooded (score > 0).
    Returns array of same length; -1 = water.
    """
    out = np.full(len(score), -1, dtype=int)
    valid = ~water_mask
    s = score[valid]
    cls = np.zeros(len(s), dtype=int)
    flooded = s > 0
    if flooded.sum() > 0:
        t33, t67 = np.percentile(s[flooded], [33, 67])
        cls[flooded & (s <= t33)] = 1
        cls[flooded & (s > t33) & (s <= t67)] = 2
        cls[flooded & (s > t67)] = 3
    out[valid] = cls
    return out


def _area_match_gs(gs_score: np.ndarray, philsa_cls: np.ndarray,
                   water_mask: np.ndarray) -> np.ndarray:
    """
    Area-match Groundsource score to the same Low/Med/High cell counts
    as the PhilSA classification (non-water cells only).
    """
    out = np.full(len(gs_score), -1, dtype=int)
    valid = ~water_mask
    s = gs_score[valid]
    p = philsa_cls[valid]

    n_high = int((p == 3).sum())
    n_med  = int((p == 2).sum())
    n_low  = int((p == 1).sum())

    cls   = np.zeros(len(s), dtype=int)
    order = np.argsort(s)[::-1]
    if n_high > 0:
        cls[order[:n_high]] = 3
    if n_med > 0:
        cls[order[n_high:n_high + n_med]] = 2
    if n_low > 0:
        cls[order[n_high + n_med:n_high + n_med + n_low]] = 1

    out[valid] = cls
    return out


def _diff_bucket(diff: int) -> str:
    if diff <= -2: return "philsa_much_higher"
    if diff == -1: return "philsa_higher"
    if diff ==  0: return "match"
    if diff ==  1: return "gs_higher"
    return "gs_much_higher"


# ===========================================================================
# 3.  Main
# ===========================================================================

print("=" * 72, flush=True)
print("PhilSA Satellite vs Google Groundsource — flood comparison", flush=True)
print("=" * 72, flush=True)

# ── Load PhilSA ──────────────────────────────────────────────────────────────
print("\n[1/4] Loading PhilSA satellite shapefiles…", flush=True)
philsa_all = load_all_philsa()
philsa_utm = philsa_all.to_crs(epsg=UTM)
n_events   = philsa_all["event_key"].nunique()
date_min   = philsa_all["event_date"].min().date()
date_max   = philsa_all["event_date"].max().date()
print(f"  {len(philsa_all):,} flood polygons | {n_events} observation dates | "
      f"{date_min} → {date_max}", flush=True)

# ── Load Groundsource ────────────────────────────────────────────────────────
print("\n[2/4] Loading Google Groundsource parquet…", flush=True)
raw  = gpd.read_parquet(PARQUET, columns=["start_date", "geometry"])
phil = raw.cx[116:127, 4:22].copy()
phil["start_date"] = pd.to_datetime(phil["start_date"])
phil["year"]       = phil["start_date"].dt.year
phil_utm           = phil.to_crs(epsg=UTM)
print(f"  {len(phil):,} Groundsource records ({phil['year'].min()}–{phil['year'].max()})",
      flush=True)

# ── OSM water masks ──────────────────────────────────────────────────────────
print("\n[3/4] Fetching OSM water bodies…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} water polygon(s)", flush=True)

# ── Per-city analysis ─────────────────────────────────────────────────────────
print("\n[4/4] Building grids & classifying…", flush=True)
city_data    = {}
summary_rows = []

for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, xx, yy, inside, buf = _build_grid(city)
    water     = water_by_city[city["slug"]]
    water_mask = _apply_water_mask(pts, water)

    # ── PhilSA: flood-observation count per cell ──────────────────────────
    # Clip to city buffer, split by event-date, rasterize each, sum
    philsa_city = gpd.clip(philsa_utm[philsa_utm.intersects(buf)].copy(), buf)
    philsa_city = philsa_city[~philsa_city.is_empty].copy()

    philsa_count = np.zeros(len(pts), dtype=float)
    n_philsa_events = 0
    for ekey, grp in philsa_city.groupby("event_key"):
        c = _rasterize_polygons(pts, grp)
        philsa_count += (c > 0).astype(float)  # binary per event
        n_philsa_events += 1

    # Classify PhilSA into 0/1/2/3 by tertile
    philsa_cls = _classify_tertile(philsa_count, water_mask)

    # ── Groundsource: smoothed log-count across all years ────────────────
    gs_city = phil_utm[phil_utm.intersects(buf)].copy()
    gs_city = gpd.clip(gs_city, buf)
    gs_city = gs_city[~gs_city.is_empty].copy()

    raw_count = _gs_counts(pts, gs_city)
    raw_grid  = np.zeros_like(xx, dtype=float)
    raw_grid[pts["yi"].astype(int), pts["xi"].astype(int)] = raw_count
    smooth    = gaussian_filter(raw_grid, sigma=SIGMA_CELLS)
    gs_score  = np.log1p(smooth[inside])
    mx = gs_score.max()
    if mx > 0:
        gs_score = gs_score / mx

    # Area-match Groundsource to PhilSA class shares
    gs_cls = _area_match_gs(gs_score, philsa_cls, water_mask)

    # ── Metrics ───────────────────────────────────────────────────────────
    valid  = ~water_mask
    p_v    = philsa_cls[valid]
    g_v    = gs_cls[valid]

    exact      = float(np.mean(p_v == g_v))
    within_one = float(np.mean(np.abs(p_v - g_v) <= 1))
    spearman   = pd.Series(p_v).corr(pd.Series(gs_score[valid]), method="spearman")
    gs_higher_share     = float(np.mean(g_v > p_v))
    match_share         = float(np.mean(g_v == p_v))
    philsa_higher_share = float(np.mean(g_v < p_v))

    pts_wgs = pts.to_crs(epsg=4326)
    plot_df = pd.DataFrame({
        "lon":         pts_wgs.geometry.x,
        "lat":         pts_wgs.geometry.y,
        "philsa_cls":  philsa_cls,
        "gs_cls":      gs_cls,
        "is_water":    water_mask,
        "diff_bucket": [
            "water" if philsa_cls[i] == -1
            else _diff_bucket(int(gs_cls[i] - philsa_cls[i]))
            for i in range(len(philsa_cls))
        ],
    })

    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    rho_str = f"{spearman:.3f}" if pd.notna(spearman) else "nan"
    print(f"    philsa_events={n_philsa_events:>3} | gs_events={len(gs_city):>5} | "
          f"water={water_mask.sum():>4} cells | exact={exact:.3f} | rho={rho_str}",
          flush=True)

    city_data[city["slug"]] = {
        "city":                city,
        "plot_df":             plot_df,
        "extent":              extent,
        "n_philsa_events":     n_philsa_events,
        "n_gs_events":         len(gs_city),
        "water_cells":         int(water_mask.sum()),
        "exact":               exact,
        "within_one":          within_one,
        "spearman":            float(spearman) if pd.notna(spearman) else float("nan"),
        "gs_higher_share":     gs_higher_share,
        "match_share":         match_share,
        "philsa_higher_share": philsa_higher_share,
    }
    summary_rows.append({
        "city":                city["name"],
        "slug":                city["slug"],
        "philsa_event_dates":  n_philsa_events,
        "philsa_date_range":   f"{date_min} → {date_max}",
        "gs_events_total":     len(gs_city),
        "water_cells_masked":  int(water_mask.sum()),
        "exact_accuracy":      exact,
        "within_one_accuracy": within_one,
        "spearman":            float(spearman) if pd.notna(spearman) else float("nan"),
        "gs_higher_share":     gs_higher_share,
        "match_share":         match_share,
        "philsa_higher_share": philsa_higher_share,
    })

summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(OUT_DIR, "philsa_groundsource_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}", flush=True)


# ===========================================================================
# 4.  Figure 1 — 5-city × 4-panel maps
# ===========================================================================
print("\nRendering Figure 1 (maps)…", flush=True)

fig1, axes1 = plt.subplots(
    len(CITIES), 4,
    figsize=(15.8, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.72, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    f"PhilSA Satellite vs Google Groundsource — Flood Frequency "
    f"({date_min} → {date_max}, {n_events} observation dates)",
    fontsize=13, fontweight="bold", y=0.997,
)
fig1.text(
    0.5, 0.969,
    f"PhilSA: binary flood presence per observation date, classified by tertile. "
    f"Groundsource: smoothed log-count (Gaussian σ={SIGMA_M} m, {GRID_M} m grid), "
    "area-matched to PhilSA class shares. Permanent water (OSM) masked in blue.",
    ha="center", fontsize=8.2, color="#444444",
)

headers = ["", "PhilSA classes", "GS aggregated", "Difference (GS − PhilSA)"]
for i, h in enumerate(headers):
    axes1[0, i].set_title(h, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d    = city_data[slug]
    rho  = d["spearman"]
    rho_str = f"{rho:.2f}" if not np.isnan(rho) else "nan"

    # ── Label column ──────────────────────────────────────────────────────
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"philsa_dates={d['n_philsa_events']}\n"
        f"gs_events={d['n_gs_events']:,}\n"
        f"water={d['water_cells']:,} cells\n"
        f"exact={d['exact']:.2f}\nrho={rho_str}",
        ha="center", va="center",
        fontsize=7.8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92, boxstyle="round,pad=0.35"),
    )

    plot_df = d["plot_df"]

    for ci, (field, title) in enumerate([
        ("philsa_cls",  "PhilSA classes"),
        ("gs_cls",      "GS aggregated"),
        ("diff_bucket", "Difference (GS − PhilSA)"),
    ]):
        ax = axes1[ri, ci + 1]
        ax.set_facecolor("#F1EDE5")

        if field in ("philsa_cls", "gs_cls"):
            for klass in [0, 1, 2, 3]:
                sub = plot_df[plot_df[field] == klass]
                if sub.empty:
                    continue
                alpha = 0.28 if klass == 0 else 0.90
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=CLASS_COLORS[klass], linewidths=0, alpha=alpha)
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)

        else:  # diff_bucket
            for bucket in ["philsa_much_higher", "philsa_higher", "match",
                           "gs_higher", "gs_much_higher"]:
                sub = plot_df[plot_df[field] == bucket]
                if sub.empty:
                    continue
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=DIFF_COLORS[bucket], linewidths=0, alpha=0.90)
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)
            ax.text(
                0.02, 0.98,
                f"rho={rho_str}\n"
                f"GS>PhilSA={d['gs_higher_share']:.2f}\n"
                f"match={d['match_share']:.2f}\n"
                f"PhilSA>GS={d['philsa_higher_share']:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.0, family="monospace",
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

# Legend
class_handles = [
    mpatches.Patch(color=CLASS_COLORS[1], label="Low"),
    mpatches.Patch(color=CLASS_COLORS[2], label="Medium"),
    mpatches.Patch(color=CLASS_COLORS[3], label="High"),
    mpatches.Patch(color=WATER_COLOR,     label="Permanent water (masked)"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["philsa_much_higher"], label="PhilSA much higher"),
    mpatches.Patch(color=DIFF_COLORS["philsa_higher"],      label="PhilSA higher"),
    mpatches.Patch(color=DIFF_COLORS["match"],              label="Match"),
    mpatches.Patch(color=DIFF_COLORS["gs_higher"],          label="GS higher"),
    mpatches.Patch(color=DIFF_COLORS["gs_much_higher"],     label="GS much higher"),
]
fig1.legend(
    handles=class_handles + diff_handles,
    loc="lower center", ncol=9, fontsize=7.5,
    framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
)
fig1.tight_layout(rect=[0, 0.02, 1, 0.965])

p1 = os.path.join(OUT_DIR, "philsa_groundsource_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# 5.  Figure 2 — Diagnostics / statistics summary
# ===========================================================================
print("Rendering Figure 2 (diagnostics)…", flush=True)

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle("PhilSA vs Groundsource — Diagnostic Statistics", fontsize=12,
              fontweight="bold")

city_names  = [c["name"] for c in CITIES]
x           = np.arange(len(CITIES))
bar_w       = 0.25
colors_city = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C"]

# 2a: Exact match & within-1
ax = axes2[0, 0]
exact_vals  = [city_data[c["slug"]]["exact"]      for c in CITIES]
within_vals = [city_data[c["slug"]]["within_one"] for c in CITIES]
ax.bar(x - bar_w / 2, exact_vals,  bar_w, label="Exact match",  color="#1976D2", alpha=0.85)
ax.bar(x + bar_w / 2, within_vals, bar_w, label="Within ±1",    color="#66BB6A", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("Fraction of cells"); ax.set_title("Class agreement")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# 2b: Spearman ρ
ax = axes2[0, 1]
rho_vals = [city_data[c["slug"]]["spearman"] for c in CITIES]
bars = ax.bar(x, rho_vals, color=colors_city, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_title("Spearman ρ (GS score vs PhilSA class)")
ax.set_ylabel("ρ"); ax.set_ylim(-1, 1)
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, rho_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v:.2f}" if not np.isnan(v) else "nan",
            ha="center", va="bottom", fontsize=7.5)

# 2c: Share breakdown (GS higher / match / PhilSA higher)
ax = axes2[0, 2]
gs_high   = [city_data[c["slug"]]["gs_higher_share"]     for c in CITIES]
match_s   = [city_data[c["slug"]]["match_share"]          for c in CITIES]
phi_high  = [city_data[c["slug"]]["philsa_higher_share"]  for c in CITIES]
ax.bar(x,            gs_high,  bar_w * 3, label="GS higher",     color="#B2182B", alpha=0.85)
ax.bar(x, match_s,   bar_w * 3, label="Match",          color="#2E7D32", alpha=0.85,
       bottom=gs_high)
phi_bottom = [g + m for g, m in zip(gs_high, match_s)]
ax.bar(x, phi_high,  bar_w * 3, label="PhilSA higher",  color="#08306B", alpha=0.85,
       bottom=phi_bottom)
ax.set_xticks(x); ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("Fraction of valid cells")
ax.set_title("Cell-level class share"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# 2d: PhilSA event dates vs GS event count scatter
ax = axes2[1, 0]
for ci, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    ax.scatter(d["n_philsa_events"], d["n_gs_events"],
               color=colors_city[ci], s=80, zorder=3,
               label=city["name"])
    ax.annotate(city["name"], (d["n_philsa_events"], d["n_gs_events"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel("PhilSA observation dates in city buffer")
ax.set_ylabel("Groundsource events in city buffer")
ax.set_title("Event count comparison")
ax.grid(alpha=0.3)

# 2e: Temporal coverage — PhilSA events per year
ax = axes2[1, 1]
philsa_all["year"] = philsa_all["event_date"].dt.year
yr_counts = philsa_all.drop_duplicates("event_key").groupby(
    philsa_all.drop_duplicates("event_key")["event_date"].dt.year
).size()
ax.bar(yr_counts.index, yr_counts.values, color="#1976D2", alpha=0.85, edgecolor="white")
ax.set_xlabel("Year"); ax.set_ylabel("Number of observation dates")
ax.set_title(f"PhilSA observations per year ({n_events} total)")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)

# 2f: Sensor breakdown
ax = axes2[1, 2]
sensor_counts = (philsa_all.drop_duplicates("event_key")
                 .groupby("sensor").size().sort_values(ascending=False))
bar_colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_counts)))
ax.bar(sensor_counts.index, sensor_counts.values, color=bar_colors, alpha=0.85, edgecolor="white")
ax.set_xlabel("Sensor"); ax.set_ylabel("Observation dates")
ax.set_title("PhilSA sensor breakdown")
ax.grid(axis="y", alpha=0.3)
for i, (sensor, cnt) in enumerate(sensor_counts.items()):
    ax.text(i, cnt + 0.2, str(cnt), ha="center", va="bottom", fontsize=8)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "philsa_groundsource_02_diagnostics.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
print(f"  {csv_path}", flush=True)
