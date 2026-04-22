"""
noah_philsa_allfiles_comparison.py
==================================
Compare NOAH 5-year flood hazard maps against all available PhilSA
satellite-derived flood extent files (all sensors, all regions, no
same-date sensor collapsing) across five Philippine cities.

NOAH is a static modelled hazard product (Low / Medium / High classes).
PhilSA is observational: binary flood presence per satellite acquisition
date, aggregated here as a flood-frequency count per 250 m grid cell and
classified into Low / Medium / High by tertile of flooded cells.

Every PhilSA shapefile is treated as its own binary flood observation.
That means same-date files from different sensors or regions are all kept.

Method
------
1. Rasterise every PhilSA shapefile onto a 250 m UTM grid per city.
2. Sum binary flood presence across files → flood-frequency score.
3. Classify PhilSA frequency into Low / Medium / High by tertile of
   cells that were ever flooded (class 0 = never flooded).
4. Load NOAH 5-yr shapefile; sample max hazard class per grid cell.
5. Difference = PhilSA class − NOAH class.

Outputs
-------
  output/noah_philsa_allfiles_01_maps.png         — 5-city × 4-panel maps
  output/noah_philsa_allfiles_02_diagnostics.png  — statistics summary
  output/noah_philsa_allfiles_summary.csv
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
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
NOAH_BASE  = os.path.join(ROOT, "data", "noah", "5yr")
os.makedirs(OUT_DIR, exist_ok=True)

UTM         = 32651
GRID_M      = 250
SAT_PRIORITY = ["s1", "s2", "rcm", "alos2", "k5", "gf3", "iceye", "nv1", "l9"]

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

NOAH_COLORS = {0: "#F1EDE5", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_COLOR = "#A8D5E2"
DIFF_COLORS = {
    "noah_much_higher":   "#08306B",
    "noah_higher":        "#6BAED6",
    "match":              "#2E7D32",
    "philsa_higher":      "#FDAE61",
    "philsa_much_higher": "#B2182B",
}
CITY_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#B71C1C",
    "#00796B", "#F57C00", "#37474F", "#880E4F", "#1B5E20",
]


# ===========================================================================
# Helpers
# ===========================================================================

def load_all_philsa():
    PAT = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    parts = []
    for fname in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        date_str, _, sensor, _ = m.groups()
        fpath = os.path.join(PHILSA_DIR, fname)
        event_date    = pd.to_datetime(date_str, format="%Y%m%d")
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
        except Exception as e:
            print(f"    WARN {fname}: {e}")
            continue
        event = gdf[["geometry"]].copy()
        event["event_date"] = event_date
        event["sensor"]     = sensor
        event["event_key"]  = os.path.splitext(fname)[0]
        event["source_file"] = fname
        parts.append(event)

    all_p = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(all_p, geometry="geometry", crs=4326)


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
    return pts, xx, yy, inside, centre.buffer(r)


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
    """Return max NOAH Var per grid cell (0 = no hazard)."""
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
    """Classify non-water cells into 0/1/2/3 by tertile of flooded cells."""
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


# ===========================================================================
# Main
# ===========================================================================

print("=" * 72, flush=True)
print("NOAH 5-yr vs PhilSA Satellite (all files) — flood comparison", flush=True)
print("=" * 72, flush=True)

# ── Load PhilSA ──────────────────────────────────────────────────────────────
print("\n[1/4] Loading PhilSA satellite shapefiles…", flush=True)
philsa_all  = load_all_philsa()
philsa_utm  = philsa_all.to_crs(epsg=UTM)
n_events    = philsa_all["event_key"].nunique()
date_min    = philsa_all["event_date"].min().date()
date_max    = philsa_all["event_date"].max().date()
print(f"  {len(philsa_all):,} polygons | {n_events} PhilSA files | "
      f"{date_min} → {date_max}", flush=True)

# ── OSM water masks ──────────────────────────────────────────────────────────
print("\n[2/4] Fetching OSM water bodies…", flush=True)
water_by_city = {}
for city in CITIES:
    print(f"  {city['name']}…", end=" ", flush=True)
    centre = (gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326)
              .to_crs(epsg=UTM).iloc[0])
    buf = centre.buffer(city["radius_m"])
    water_by_city[city["slug"]] = _load_water_mask(city, buf)
    print(f"{len(water_by_city[city['slug']])} water polygon(s)", flush=True)

# ── Per-city analysis ─────────────────────────────────────────────────────────
print("\n[3/4] Building grids & classifying…", flush=True)
city_data    = {}
summary_rows = []

for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, xx, yy, inside, buf = _build_grid(city)
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

    noah_cls = _sample_noah(pts, noah)  # 0/1/2/3 per cell

    # ── PhilSA ────────────────────────────────────────────────────────────
    philsa_city = gpd.clip(philsa_utm[philsa_utm.intersects(buf)].copy(), buf)
    philsa_city = philsa_city[~philsa_city.is_empty].copy()

    philsa_count    = np.zeros(len(pts), dtype=float)
    n_philsa_events = 0
    for ekey, grp in philsa_city.groupby("event_key"):
        c = _rasterize_polygons(pts, grp)
        philsa_count += (c > 0).astype(float)
        n_philsa_events += 1

    philsa_cls = _classify_tertile(philsa_count, water_mask)  # -1/0/1/2/3

    # ── Metrics ───────────────────────────────────────────────────────────
    # Compare only non-water cells where NOAH has data (Var > 0 or any NOAH)
    valid   = ~water_mask
    n_cls   = noah_cls[valid]
    p_cls   = philsa_cls[valid]

    # For cells where PhilSA is -1 (water-only in PhilSA classification) treat as 0
    p_cls_safe = np.where(p_cls == -1, 0, p_cls)

    exact      = float(np.mean(n_cls == p_cls_safe))
    within_one = float(np.mean(np.abs(n_cls.astype(int) - p_cls_safe.astype(int)) <= 1))
    spearman   = pd.Series(n_cls).corr(pd.Series(philsa_count[valid]), method="spearman")

    philsa_higher_share = float(np.mean(p_cls_safe > n_cls))
    match_share         = float(np.mean(p_cls_safe == n_cls))
    noah_higher_share   = float(np.mean(n_cls > p_cls_safe))

    philsa_freq = philsa_count / max(n_philsa_events, 1)

    pts_wgs = pts.to_crs(epsg=4326)
    plot_df = pd.DataFrame({
        "lon":        pts_wgs.geometry.x,
        "lat":        pts_wgs.geometry.y,
        "noah_cls":   noah_cls,
        "philsa_cls": np.where(philsa_cls == -1, 0, philsa_cls),
        "philsa_freq": philsa_freq,
        "is_water":   water_mask,
        "diff_bucket": [
            "water" if water_mask[i]
            else _diff_bucket(
                int(max(0, philsa_cls[i])) - int(noah_cls[i])
            )
            for i in range(len(noah_cls))
        ],
    })

    extent = [float(plot_df["lon"].min()), float(plot_df["lon"].max()),
              float(plot_df["lat"].min()), float(plot_df["lat"].max())]

    rho_str = f"{spearman:.3f}" if pd.notna(spearman) else "nan"
    n_noah_cells = int((noah_cls > 0).sum())
    print(f"    philsa_events={n_philsa_events:>3} | noah_cells={n_noah_cells:>5} | "
          f"water={water_mask.sum():>4} | exact={exact:.3f} | rho={rho_str}", flush=True)

    city_data[city["slug"]] = {
        "city":                city,
        "plot_df":             plot_df,
        "extent":              extent,
        "n_philsa_events":     n_philsa_events,
        "n_noah_cells":        n_noah_cells,
        "water_cells":         int(water_mask.sum()),
        "exact":               exact,
        "within_one":          within_one,
        "spearman":            float(spearman) if pd.notna(spearman) else float("nan"),
        "philsa_higher_share": philsa_higher_share,
        "match_share":         match_share,
        "noah_higher_share":   noah_higher_share,
        # Mean PhilSA frequency per NOAH class (for diagnostic plot)
        "mean_philsa_by_noah": {
            k: float(philsa_count[valid][n_cls == k].mean())
            if (n_cls == k).any() else float("nan")
            for k in [0, 1, 2, 3]
        },
    }
    summary_rows.append({
        "city":                  city["name"],
        "slug":                  city["slug"],
        "philsa_files":          n_philsa_events,
        "philsa_date_range":     f"{date_min} → {date_max}",
        "noah_hazard_cells":     n_noah_cells,
        "water_cells_masked":    int(water_mask.sum()),
        "exact_accuracy":        exact,
        "within_one_accuracy":   within_one,
        "spearman":              float(spearman) if pd.notna(spearman) else float("nan"),
        "philsa_higher_share":   philsa_higher_share,
        "match_share":           match_share,
        "noah_higher_share":     noah_higher_share,
    })

summary_df = pd.DataFrame(summary_rows)
csv_path   = os.path.join(OUT_DIR, "noah_philsa_allfiles_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}", flush=True)

# Per-cell parquet (used by noah_source_scatter.py)
all_cell_frames = []
for city in CITIES:
    d  = city_data[city["slug"]]
    df = d["plot_df"].copy()
    df["city"] = city["name"]
    all_cell_frames.append(df[["city", "lon", "lat", "noah_cls", "philsa_freq", "is_water"]])
cells_df   = pd.concat(all_cell_frames, ignore_index=True)
cells_path = os.path.join(OUT_DIR, "noah_philsa_allfiles_cells.parquet")
cells_df.to_parquet(cells_path, index=False)
print(f"  Saved → {cells_path}", flush=True)


# ===========================================================================
# Figure 1 — maps
# ===========================================================================
print("\nRendering Figure 1 (maps)…", flush=True)

fig1, axes1 = plt.subplots(
    len(CITIES), 4,
    figsize=(15.8, 3.25 * len(CITIES)),
    gridspec_kw={"width_ratios": [0.72, 1.0, 1.0, 1.0]},
)
fig1.patch.set_facecolor("#F7F7F7")
fig1.suptitle(
    f"NOAH 5-yr Hazard vs PhilSA Satellite Flood Extents, All Files "
    f"({date_min} → {date_max}, {n_events} PhilSA files)",
    fontsize=13, fontweight="bold", y=0.997,
)
fig1.text(
    0.5, 0.969,
    f"PhilSA: binary flood presence per source file classified by tertile "
    f"({GRID_M} m grid). Permanent water (OSM) masked in blue.",
    ha="center", fontsize=8.2, color="#444444",
)

headers = ["", "NOAH 5-yr classes", "PhilSA classes", "Difference (PhilSA − NOAH)"]
for i, h in enumerate(headers):
    axes1[0, i].set_title(h, fontsize=10, fontweight="bold", pad=6)

for ri, city in enumerate(CITIES):
    slug = city["slug"]
    d    = city_data[slug]
    rho  = d["spearman"]
    rho_str = f"{rho:.2f}" if not np.isnan(rho) else "nan"

    # Label column
    lax = axes1[ri, 0]
    lax.axis("off")
    lax.text(
        0.5, 0.5,
        f"{city['name']}\n{city['region']}\n"
        f"philsa_files={d['n_philsa_events']}\n"
        f"noah_cells={d['n_noah_cells']:,}\n"
        f"water={d['water_cells']:,} cells\n"
        f"exact={d['exact']:.2f}\nwithin1={d['within_one']:.2f}\nrho={rho_str}",
        ha="center", va="center",
        fontsize=7.8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.92,
                  boxstyle="round,pad=0.35"),
    )

    plot_df = d["plot_df"]

    for ci, field in enumerate(["noah_cls", "philsa_cls", "diff_bucket"]):
        ax = axes1[ri, ci + 1]
        ax.set_facecolor("#F1EDE5")

        if field in ("noah_cls", "philsa_cls"):
            for klass in [0, 1, 2, 3]:
                sub = plot_df[plot_df[field] == klass]
                if sub.empty:
                    continue
                alpha = 0.28 if klass == 0 else 0.90
                ax.scatter(sub["lon"], sub["lat"], s=8, marker="s",
                           c=NOAH_COLORS[klass], linewidths=0, alpha=alpha)
            water_pts = plot_df[plot_df["is_water"]]
            if not water_pts.empty:
                ax.scatter(water_pts["lon"], water_pts["lat"], s=8, marker="s",
                           c=WATER_COLOR, linewidths=0, alpha=0.85)

        else:  # diff_bucket
            for bucket in ["noah_much_higher", "noah_higher", "match",
                           "philsa_higher", "philsa_much_higher"]:
                sub = plot_df[plot_df["diff_bucket"] == bucket]
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
                f"PhilSA>NOAH={d['philsa_higher_share']:.2f}\n"
                f"match={d['match_share']:.2f}\n"
                f"NOAH>PhilSA={d['noah_higher_share']:.2f}",
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

class_handles = [
    mpatches.Patch(color=NOAH_COLORS[1], label="Low"),
    mpatches.Patch(color=NOAH_COLORS[2], label="Medium"),
    mpatches.Patch(color=NOAH_COLORS[3], label="High"),
    mpatches.Patch(color=WATER_COLOR,    label="Permanent water (masked)"),
]
diff_handles = [
    mpatches.Patch(color=DIFF_COLORS["noah_much_higher"],   label="NOAH much higher"),
    mpatches.Patch(color=DIFF_COLORS["noah_higher"],        label="NOAH higher"),
    mpatches.Patch(color=DIFF_COLORS["match"],              label="Match"),
    mpatches.Patch(color=DIFF_COLORS["philsa_higher"],      label="PhilSA higher"),
    mpatches.Patch(color=DIFF_COLORS["philsa_much_higher"], label="PhilSA much higher"),
]
fig1.legend(
    handles=class_handles + diff_handles,
    loc="lower center", ncol=9, fontsize=7.5,
    framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
)
fig1.tight_layout(rect=[0, 0.02, 1, 0.965])

p1 = os.path.join(OUT_DIR, "noah_philsa_allfiles_01_maps.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {p1}", flush=True)


# ===========================================================================
# Figure 2 — Diagnostics
# ===========================================================================
print("Rendering Figure 2 (diagnostics)…", flush=True)

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.patch.set_facecolor("#F7F7F7")
fig2.suptitle("NOAH 5-yr vs PhilSA Satellite (all files) — Diagnostic Statistics",
              fontsize=12, fontweight="bold")

city_names = [c["name"] for c in CITIES]
x          = np.arange(len(CITIES))
bar_w      = 0.35

# 2a: Mean PhilSA frequency per NOAH class
ax = axes2[0, 0]
for i, city in enumerate(CITIES):
    vals = [city_data[city["slug"]]["mean_philsa_by_noah"][k] for k in [0, 1, 2, 3]]
    ax.plot([0, 1, 2, 3], vals, marker="o", linewidth=2, markersize=5,
            color=CITY_COLORS[i], label=city["name"])
ax.set_title("Mean PhilSA frequency by NOAH class", fontweight="bold")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["No hazard", "Low", "Medium", "High"], rotation=20, ha="right")
ax.set_ylabel("Mean flood-observation count")
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
ax.set_title("Class agreement", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2c: Spearman ρ
ax = axes2[0, 2]
rho_vals = [city_data[c["slug"]]["spearman"] for c in CITIES]
bars = ax.bar(x, rho_vals, color=CITY_COLORS, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_title("Spearman ρ (NOAH class vs PhilSA frequency)", fontweight="bold")
ax.set_ylabel("ρ")
ax.set_ylim(-1, 1)
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, rho_vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if v >= 0 else -0.08),
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

# 2d: Stacked share breakdown
ax = axes2[1, 0]
phi_h  = [city_data[c["slug"]]["philsa_higher_share"] for c in CITIES]
match  = [city_data[c["slug"]]["match_share"]          for c in CITIES]
noah_h = [city_data[c["slug"]]["noah_higher_share"]    for c in CITIES]
ax.bar(x, phi_h,  color=DIFF_COLORS["philsa_much_higher"], alpha=0.85,
       label="PhilSA higher")
ax.bar(x, match,  color=DIFF_COLORS["match"],              alpha=0.85,
       label="Match",          bottom=phi_h)
phi_m_bottom = [p + m for p, m in zip(phi_h, match)]
ax.bar(x, noah_h, color=DIFF_COLORS["noah_much_higher"],   alpha=0.85,
       label="NOAH higher",    bottom=phi_m_bottom)
ax.set_xticks(x)
ax.set_xticklabels(city_names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of valid cells")
ax.set_title("Cell-level class share", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 2e: NOAH hazard area vs PhilSA coverage
ax = axes2[1, 1]
for ci, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    # NOAH cells with hazard vs PhilSA event count
    ax.scatter(d["n_philsa_events"], d["n_noah_cells"],
               color=CITY_COLORS[ci], s=80, zorder=3)
    ax.annotate(city["name"], (d["n_philsa_events"], d["n_noah_cells"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel("PhilSA files in city buffer")
ax.set_ylabel("NOAH hazard cells (Var > 0)")
ax.set_title("NOAH hazard area vs PhilSA coverage", fontweight="bold")
ax.grid(alpha=0.3)

# 2f: Sensor breakdown of PhilSA events
ax = axes2[1, 2]
sensor_counts = (philsa_all.drop_duplicates("event_key")
                 .groupby("sensor").size().sort_values(ascending=False))
bar_colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_counts)))
ax.bar(sensor_counts.index, sensor_counts.values, color=bar_colors,
       alpha=0.85, edgecolor="white")
ax.set_xlabel("Sensor")
ax.set_ylabel("PhilSA files")
ax.set_title(f"PhilSA sensor breakdown ({n_events} total files)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, (sensor, cnt) in enumerate(sensor_counts.items()):
    ax.text(i, cnt + 0.2, str(cnt), ha="center", va="bottom", fontsize=8)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "noah_philsa_allfiles_02_diagnostics.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {p2}", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE", flush=True)
print(f"  {p1}", flush=True)
print(f"  {p2}", flush=True)
print(f"  {csv_path}", flush=True)
