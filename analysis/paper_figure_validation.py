"""
paper_figure_validation.py
==========================
Publication-quality figure: NOAH 5-yr hazard vs satellite flood observations.

Layout  : 5 cities × 3 panels
          (a) NOAH 5-yr hazard classes
          (b) Observed flood frequency  (PhilSA SAR + GFD/MODIS combined)
          (c) Bias map  (Confirmed / Modelled-only / Empirical-gap / Low)

Design  : 7-inch wide (fits double-column journal), 300 dpi
          Minimum 7 pt body text, 9 pt labels
          Teal permanent water, purple NOAH-higher (no blue ambiguity)
          CSI / POD / FAR stats in each bias panel

Output  : output/paper_fig_noah_validation.pdf
          output/paper_fig_noah_validation.png   (300 dpi)
"""

import glob, os, re, warnings, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(ROOT, "output", "paper_figures")
PHILSA_DIR = os.path.join(ROOT, "data", "philsa_satellite_flood")
GFD_DIR    = os.path.join(ROOT, "data", "gfd")
NOAH_BASE  = os.path.join(ROOT, "data", "noah", "5yr")

UTM          = 32651
GRID_M       = 250
SAT_PRIORITY = ["s1","s2","rcm","alos2","k5","gf3","iceye","nv1","l9","gfd"]
NOAH_ACTIVE  = 2      # Var ≥ 2 → "NOAH active" (Medium or High)

CITIES = [
    {"name": "Tuguegarao",     "region": "Cagayan Valley", "slug": "tuguegarao",
     "lat": 17.6158, "lng": 121.7229, "radius_m": 10_000,
     "noah_province": "Cagayan"},
    {"name": "Dagupan",        "region": "Ilocos",          "slug": "dagupan",
     "lat": 16.0431, "lng": 120.3333, "radius_m": 12_000,
     "noah_province": "Pangasinan"},
    {"name": "Manila",         "region": "NCR",             "slug": "manila",
     "lat": 14.5995, "lng": 120.9842, "radius_m": 20_000,
     "noah_province": "Metropolitan Manila"},
    {"name": "Cagayan de Oro", "region": "Mindanao",        "slug": "cagayan_de_oro",
     "lat":  8.4772, "lng": 124.6459, "radius_m": 12_000,
     "noah_province": "Misamis Oriental"},
    {"name": "Cotabato",       "region": "BARMM",           "slug": "cotabato",
     "lat":  7.2236, "lng": 124.2464, "radius_m": 10_000,
     "noah_province": "Maguindanao"},
]

# ── colours ──────────────────────────────────────────────────────────────────
NOAH_C = {0: "#F5F0EA", 1: "#FFD54F", 2: "#EF6C00", 3: "#B71C1C"}
WATER_C = "#2CBBB4"                        # teal – permanent water
BIAS_C  = {
    "confirmed": "#C62828",               # dark red   – both sources flag risk
    "modelled":  "#1565C0",               # dark blue  – NOAH only (over-prediction)
    "empirical": "#E65100",               # burnt orange – obs only (model gap)
    "low":       "#EFEFEF",               # light grey – low risk
    "water":     WATER_C,
}
FREQ_CMAP = plt.cm.YlOrRd

# ── data helpers (identical logic to existing scripts) ────────────────────────

def _sat_rank(s):
    try:    return SAT_PRIORITY.index(s.lower())
    except: return len(SAT_PRIORITY)

def _load_philsa():
    PAT = re.compile(r"^(\d{8})_(\d{4})_fld_(\w+)_shp(.*)\.zip$")
    inv = {}
    for f in sorted(os.listdir(PHILSA_DIR)):
        m = PAT.match(f)
        if not m: continue
        d, _, sen, _ = m.groups()
        inv.setdefault(d, {}).setdefault(sen, []).append(os.path.join(PHILSA_DIR, f))
    parts = []
    for d, smap in sorted(inv.items()):
        sen = min(smap, key=_sat_rank)
        gdfs = []
        for fp in smap[sen]:
            try:
                with zipfile.ZipFile(fp) as z:
                    shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
                    if not shps: continue
                    g = gpd.read_file(f"zip://{fp}!{shps[0]}")
                if g is None or len(g)==0: continue
                if g.crs is None: g = g.set_crs(4326)
                elif g.crs.to_epsg()!=4326: g = g.to_crs(4326)
                g = g[g.geometry.notna() & g.geometry.is_valid]
                gdfs.append(g[["geometry"]])
            except: pass
        if not gdfs: continue
        ev = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)
        ev["event_date"] = pd.to_datetime(d, format="%Y%m%d")
        ev["source"]     = "philsa"
        ev["event_key"]  = f"{d}_{sen}"
        parts.append(ev)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry","event_date","source","event_key"], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)

def _load_gfd():
    PAT = re.compile(r"^(\d{8})_(.+)_fld_gfd_shp\.zip$")
    parts = []
    if not os.path.isdir(GFD_DIR): return gpd.GeoDataFrame(
        columns=["geometry","event_date","source","event_key"], crs=4326)
    for f in sorted(os.listdir(GFD_DIR)):
        m = PAT.match(f)
        if not m: continue
        d, eid = m.groups()
        fp = os.path.join(GFD_DIR, f)
        try:
            with zipfile.ZipFile(fp) as z:
                shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
                if not shps: continue
                g = gpd.read_file(f"zip://{fp}!{shps[0]}")
            if g is None or len(g)==0: continue
            if g.crs is None: g = g.set_crs(4326)
            elif g.crs.to_epsg()!=4326: g = g.to_crs(4326)
            g = g[g.geometry.notna() & g.geometry.is_valid][["geometry"]].copy()
            g["event_date"] = pd.to_datetime(d, format="%Y%m%d")
            g["source"]     = "gfd"
            g["event_key"]  = f"{d}_gfd_{eid}"
            parts.append(g)
        except: pass
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry","event_date","source","event_key"], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)

def _build_grid(city):
    c = gpd.GeoSeries([Point(city["lng"], city["lat"])], crs=4326).to_crs(UTM).iloc[0]
    r = city["radius_m"]
    xs = np.arange(c.x-r, c.x+r+GRID_M, GRID_M)
    ys = np.arange(c.y-r, c.y+r+GRID_M, GRID_M)
    xx, yy = np.meshgrid(xs, ys)
    ins = (xx-c.x)**2 + (yy-c.y)**2 <= r**2
    yi, xi = np.where(ins)
    pts = gpd.GeoDataFrame({"xi":xi,"yi":yi},
        geometry=[Point(xx[y,x], yy[y,x]) for y,x in zip(yi,xi)], crs=UTM)
    return pts, c.buffer(r)

def _water_mask(city, buf):
    import signal
    bw = gpd.GeoSeries([buf], crs=UTM).to_crs(4326).iloc[0]
    def _t(s,f): raise TimeoutError
    frames=[]
    for tags in [{"natural":"water"},{"landuse":"reservoir"}]:
        signal.signal(signal.SIGALRM,_t); signal.alarm(25)
        try:
            g = ox.features_from_polygon(bw, tags=tags); signal.alarm(0)
            p = g[g.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
            if not p.empty: frames.append(p[["geometry"]])
        except: signal.alarm(0)
    if not frames: return gpd.GeoDataFrame(columns=["geometry"], crs=UTM)
    w = pd.concat(frames, ignore_index=True)
    w = gpd.GeoDataFrame(w, crs=4326).to_crs(UTM)
    w = gpd.clip(w[["geometry"]], buf)
    return w[~w.is_empty].copy()

def _apply_water(pts, water):
    if water.empty: return np.zeros(len(pts), bool)
    j = gpd.sjoin(pts[["xi","yi","geometry"]], water[["geometry"]],
                  how="left", predicate="within")
    wset = set(j.dropna(subset=["index_right"]).index)
    return np.array([i in wset for i in range(len(pts))], bool)

def _sample_noah(pts, noah):
    out = np.zeros(len(pts), int)
    if noah.empty: return out
    j = gpd.sjoin(pts[["xi","yi","geometry"]], noah[["Var","geometry"]],
                  how="left", predicate="intersects")
    mv = j.groupby(["xi","yi"])["Var"].max().reset_index()
    im = {(int(r.xi),int(r.yi)):i for i,r in pts.reset_index().iterrows()}
    for _,row in mv.iterrows():
        k=(int(row["xi"]),int(row["yi"]))
        if k in im and not np.isnan(row["Var"]): out[im[k]]=int(row["Var"])
    return out

def _rasterize(pts, polys):
    c = np.zeros(len(pts), float)
    if polys.empty: return c
    j = gpd.sjoin(pts[["xi","yi","geometry"]], polys[["geometry"]],
                  how="left", predicate="within")
    h = j.dropna(subset=["index_right"]).groupby(["xi","yi"]).size().reset_index(name="n")
    im = {(int(r.xi),int(r.yi)):i for i,r in pts.reset_index().iterrows()}
    for _,row in h.iterrows():
        k=(int(row["xi"]),int(row["yi"]))
        if k in im: c[im[k]]=float(row["n"])
    return c

def _freq_thresh(freq, wmask):
    v = freq[~wmask]; f = v[v>0]
    return float(np.percentile(f,67)) if len(f)>0 else 0.01

def _bias_cat(noah_cls, freq, water, thresh):
    if water:            return "water"
    hi_n = noah_cls >= NOAH_ACTIVE
    hi_f = freq     >= thresh
    if hi_n and hi_f:   return "confirmed"
    if hi_n:            return "modelled"
    if hi_f:            return "empirical"
    return "low"

def _find_noah(prov):
    for folder in [prov, prov.replace(" ",""), prov.replace(" ","").lower()]:
        base = os.path.join(NOAH_BASE, folder)
        if os.path.isdir(base):
            shps = glob.glob(os.path.join(base,"*.shp"))
            if shps: return shps[0]
    return None

# ── load & process ────────────────────────────────────────────────────────────
print("Loading observations…", flush=True)
obs_all = gpd.GeoDataFrame(
    pd.concat([_load_philsa(), _load_gfd()], ignore_index=True), crs=4326)
obs_utm = obs_all.to_crs(UTM)
n_philsa = obs_all[obs_all.source=="philsa"]["event_key"].nunique()
n_gfd    = obs_all[obs_all.source=="gfd"]["event_key"].nunique()
date_min = obs_all.event_date.min().date()
date_max = obs_all.event_date.max().date()
print(f"  PhilSA={n_philsa}  GFD={n_gfd}  total={n_philsa+n_gfd}", flush=True)

print("Building city grids…", flush=True)
city_data = {}
for city in CITIES:
    print(f"  {city['name']}…", flush=True)
    pts, buf = _build_grid(city)
    water    = _water_mask(city, buf)
    wmask    = _apply_water(pts, water)

    # NOAH
    ns = _find_noah(city["noah_province"])
    if ns:
        noah = gpd.read_file(ns)
        if noah.crs is None: noah = noah.set_crs(4326)
        if noah.crs.to_epsg()!=UTM: noah = noah.to_crs(UTM)
        noah["Var"] = pd.to_numeric(noah["Var"],errors="coerce").fillna(0).astype(int)
        noah = gpd.clip(noah[["Var","geometry"]], buf)
        noah = noah[~noah.is_empty]
    else:
        noah = gpd.GeoDataFrame(columns=["Var","geometry"], crs=UTM)
    ncls = _sample_noah(pts, noah)

    # Observations
    oc = gpd.clip(obs_utm[obs_utm.intersects(buf)].copy(), buf)
    oc = oc[~oc.is_empty]
    freq = np.zeros(len(pts), float)
    n_ev = 0
    for _, grp in oc.groupby("event_key"):
        freq += (_rasterize(pts, grp) > 0).astype(float)
        n_ev += 1
    freq_frac = freq / max(n_ev, 1)
    thresh    = _freq_thresh(freq_frac, wmask)

    # Bias categories
    cats = [_bias_cat(int(ncls[i]), float(freq_frac[i]), bool(wmask[i]), thresh)
            for i in range(len(pts))]

    # Metrics
    valid = ~wmask
    n_conf = sum(1 for c in cats if c=="confirmed")
    n_mod  = sum(1 for c in cats if c=="modelled")
    n_emp  = sum(1 for c in cats if c=="empirical")
    denom_csi = n_conf + n_mod + n_emp
    denom_pod = n_conf + n_emp
    denom_far = n_conf + n_mod
    csi  = n_conf / denom_csi if denom_csi > 0 else 0.0
    pod  = n_conf / denom_pod if denom_pod > 0 else 0.0
    far  = n_mod  / denom_far if denom_far > 0 else 0.0
    bias = denom_far / denom_pod if denom_pod > 0 else 1.0
    print(f"    n_ev={n_ev} | CSI={csi:.2f} POD={pod:.2f} FAR={far:.2f} bias={bias:.2f}",
          flush=True)

    pts_wgs = pts.to_crs(4326)
    city_data[city["slug"]] = dict(
        lon=pts_wgs.geometry.x.values, lat=pts_wgs.geometry.y.values,
        ncls=ncls, freq=freq_frac, cats=cats, wmask=wmask,
        freq_max=float(freq_frac[valid].max()) if valid.any() else 0.01,
        thresh=thresh, n_ev=n_ev,
        n_philsa=oc[oc.source=="philsa"]["event_key"].nunique() if len(oc) else 0,
        n_gfd   =oc[oc.source=="gfd"   ]["event_key"].nunique() if len(oc) else 0,
        csi=csi, pod=pod, far=far, bias=bias,
        extent=[float(pts_wgs.geometry.x.min()), float(pts_wgs.geometry.x.max()),
                float(pts_wgs.geometry.y.min()), float(pts_wgs.geometry.y.max())],
        city=city,
    )

# ── figure ────────────────────────────────────────────────────────────────────
print("Rendering paper figure…", flush=True)

NCITIES = len(CITIES)
FIG_W   = 7.0          # inches – fits double-column journal
ROW_H   = 1.90         # inches per city row
TOP_PAD = 0.55         # title + subtitle
LEG_H   = 0.55         # legend strip at bottom
FIG_H   = TOP_PAD + NCITIES * ROW_H + LEG_H

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

# Outer grid: title area | map area | legend area
outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[TOP_PAD, NCITIES*ROW_H, LEG_H],
    hspace=0.0,
    top=1.0, bottom=0.0, left=0.0, right=1.0,
)

# Map grid: NCITIES rows × 4 cols (label | NOAH | freq | bias)
map_gs = gridspec.GridSpecFromSubplotSpec(
    NCITIES, 4, subplot_spec=outer[1],
    width_ratios=[0.22, 1, 1, 1],
    wspace=0.04, hspace=0.08,
)

# ── column headers ────────────────────────────────────────────────────────────
hdr_ax = fig.add_subplot(outer[0])
hdr_ax.axis("off")
hdr_ax.text(0.5, 0.92,
    "NOAH 5-yr Hazard vs Satellite Flood Observations — Five Philippine Cities",
    ha="center", va="top", fontsize=9, fontweight="bold",
    transform=hdr_ax.transAxes)
hdr_ax.text(0.5, 0.52,
    f"Observations: PhilSA SAR (n={n_philsa}) + GFD/MODIS (n={n_gfd}),  "
    f"{date_min} – {date_max}  |  Grid: 250 m UTM  |  "
    f"Permanent water (OSM) masked",
    ha="center", va="top", fontsize=6.5, color="#444",
    transform=hdr_ax.transAxes)

# Column title positions (approximate, using a dummy row)
col_titles = ["", "(a) NOAH 5-yr hazard", "(b) Observed frequency", "(c) Bias"]
col_x      = [0.13, 0.36, 0.62, 0.87]
for x, t in zip(col_x[1:], col_titles[1:]):
    hdr_ax.text(x, 0.12, t, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", transform=hdr_ax.transAxes)

# ── per-city rows ─────────────────────────────────────────────────────────────
for ri, city in enumerate(CITIES):
    d = city_data[city["slug"]]
    lon, lat = d["lon"], d["lat"]
    ext = d["extent"]

    def _ax(ci, ri=ri):
        ax = fig.add_subplot(map_gs[ri, ci])
        ax.set_facecolor("#F8F5F0")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal", adjustable="box")
        # Only show coordinate ticks on bottom row & leftmost map
        if ri < NCITIES-1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(2))
            ax.tick_params(axis="x", labelsize=5.5, rotation=30)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(2))
        ax.tick_params(axis="y", labelsize=5.5)
        ax.ticklabel_format(style="plain", useOffset=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        return ax

    # Col 0 – city label
    lax = fig.add_subplot(map_gs[ri, 0])
    lax.axis("off")
    lax.text(0.95, 0.55,
             f"{city['name']}\n{city['region']}\n\n"
             f"n = {d['n_ev']}\n"
             f"({d['n_philsa']} SAR\n"
             f" {d['n_gfd']} GFD)",
             ha="right", va="center", fontsize=6.0,
             transform=lax.transAxes, linespacing=1.4,
             color="#222")

    # Col 1 – NOAH
    ax1 = _ax(1)
    wmask = d["wmask"]
    for k in [0,1,2,3]:
        mask = d["ncls"]==k
        if mask.any():
            ax1.scatter(lon[mask], lat[mask], s=1.2, marker="s",
                        c=NOAH_C[k], linewidths=0,
                        alpha=0.18 if k==0 else 0.85, rasterized=True)
    if wmask.any():
        ax1.scatter(lon[wmask], lat[wmask], s=1.2, marker="s",
                    c=WATER_C, linewidths=0, alpha=0.90, rasterized=True)
    ax1.plot(city["lng"], city["lat"], "*", color="#333",
             markersize=3, markeredgewidth=0.3, zorder=5)

    # Col 2 – Observed frequency (continuous, per-city scaled)
    ax2 = _ax(2)
    fnorm  = mcolors.Normalize(vmin=0, vmax=max(d["freq_max"], 0.01))
    zero   = (~wmask) & (d["freq"]==0)
    active = (~wmask) & (d["freq"] >0)
    if zero.any():
        ax2.scatter(lon[zero], lat[zero], s=1.2, marker="s",
                    c="#F8F5F0", linewidths=0, alpha=0.30, rasterized=True)
    if active.any():
        ax2.scatter(lon[active], lat[active], s=1.2, marker="s",
                    c=FREQ_CMAP(fnorm(d["freq"][active])),
                    linewidths=0, alpha=0.90, rasterized=True)
    if wmask.any():
        ax2.scatter(lon[wmask], lat[wmask], s=1.2, marker="s",
                    c=WATER_C, linewidths=0, alpha=0.90, rasterized=True)
    ax2.plot(city["lng"], city["lat"], "*", color="#333",
             markersize=3, markeredgewidth=0.3, zorder=5)
    # Small max-freq label
    ax2.text(0.97, 0.03, f"max={d['freq_max']:.2f}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=5.0, color="#555")

    # Col 3 – Bias
    ax3 = _ax(3)
    cats = np.array(d["cats"])
    order = ["low","modelled","empirical","confirmed","water"]
    for cat in order:
        mask = (cats==cat) if cat!="water" else wmask
        if mask.any():
            ax3.scatter(lon[mask], lat[mask], s=1.2, marker="s",
                        c=BIAS_C[cat], linewidths=0,
                        alpha=0.15 if cat=="low" else 0.88,
                        rasterized=True)
    ax3.plot(city["lng"], city["lat"], "*", color="#333",
             markersize=3, markeredgewidth=0.3, zorder=5)

    # Stats box
    stats_txt = (f"CSI={d['csi']:.2f}\n"
                 f"POD={d['pod']:.2f}\n"
                 f"FAR={d['far']:.2f}\n"
                 f"bias={d['bias']:.2f}")
    ax3.text(0.03, 0.97, stats_txt,
             transform=ax3.transAxes, va="top", ha="left",
             fontsize=5.2, family="monospace", linespacing=1.35,
             bbox=dict(facecolor="white", edgecolor="#bbb",
                       alpha=0.88, boxstyle="round,pad=0.25", linewidth=0.5))

# ── legend ────────────────────────────────────────────────────────────────────
leg_ax = fig.add_subplot(outer[2])
leg_ax.axis("off")

# Row 1: NOAH classes
noah_handles = [
    mpatches.Patch(facecolor=NOAH_C[1], label="Low",    edgecolor="#888", linewidth=0.4),
    mpatches.Patch(facecolor=NOAH_C[2], label="Medium", edgecolor="#888", linewidth=0.4),
    mpatches.Patch(facecolor=NOAH_C[3], label="High",   edgecolor="#888", linewidth=0.4),
]
# Row 2: Bias categories
bias_handles = [
    mpatches.Patch(facecolor=BIAS_C["confirmed"],  label="Confirmed risk",    edgecolor="#888", linewidth=0.4),
    mpatches.Patch(facecolor=BIAS_C["modelled"],   label="Modelled only (NOAH over-predicts)", edgecolor="#888", linewidth=0.4),
    mpatches.Patch(facecolor=BIAS_C["empirical"],  label="Empirical gap (model misses)",       edgecolor="#888", linewidth=0.4),
    mpatches.Patch(facecolor=WATER_C,              label="Permanent water",   edgecolor="#888", linewidth=0.4),
]

# Freq colorbar
sm = plt.cm.ScalarMappable(cmap=FREQ_CMAP, norm=mcolors.Normalize(0,1))
sm.set_array([])

# Place two legend rows manually
y1, y2 = 0.95, 0.42
leg_ax.legend(handles=noah_handles,
              loc="upper left", bbox_to_anchor=(0.0, y1),
              ncol=3, fontsize=6.2, frameon=False,
              handlelength=1.2, handleheight=0.9,
              columnspacing=1.0,
              title="NOAH 5-yr hazard class", title_fontsize=6.5)
leg_ax.legend(handles=bias_handles,
              loc="upper left", bbox_to_anchor=(0.0, y2),
              ncol=4, fontsize=6.2, frameon=False,
              handlelength=1.2, handleheight=0.9,
              columnspacing=0.9,
              title="(c) Bias categories", title_fontsize=6.5)

# Can't have two calls to .legend() — use add_artist for first
from matplotlib.legend import Legend
l1 = Legend(leg_ax, noah_handles,
            [h.get_label() for h in noah_handles],
            loc="upper left", bbox_to_anchor=(0.0, y1),
            ncol=3, fontsize=6.2, frameon=False,
            handlelength=1.2, handleheight=0.9,
            columnspacing=1.0,
            title="NOAH 5-yr hazard class", title_fontsize=6.5)
l1._legend_title_box._text.set_fontweight("bold")
leg_ax.add_artist(l1)

l2 = Legend(leg_ax, bias_handles,
            [h.get_label() for h in bias_handles],
            loc="upper left", bbox_to_anchor=(0.0, y2),
            ncol=4, fontsize=6.2, frameon=False,
            handlelength=1.2, handleheight=0.9,
            columnspacing=0.9,
            title="(c) Bias categories  —  CSI = TP/(TP+FP+FN),  POD = TP/(TP+FN),  "
                  "FAR = FP/(TP+FP),  bias = (TP+FP)/(TP+FN)",
            title_fontsize=5.8)
l2._legend_title_box._text.set_fontweight("bold")
leg_ax.add_artist(l2)

# Colorbar for freq panel
cbar_ax = fig.add_axes([0.72, 0.015, 0.12, 0.018])
cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Obs. flood freq.  (0 → city max)", fontsize=5.5, labelpad=2)
cb.ax.tick_params(labelsize=5.0, length=2)
cb.outline.set_linewidth(0.4)

# ── save ──────────────────────────────────────────────────────────────────────
for ext_fmt in ["png", "pdf"]:
    out = os.path.join(OUT_DIR, f"paper_fig_noah_validation.{ext_fmt}")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", format=ext_fmt)
    kb = os.path.getsize(out)/1024
    print(f"  Saved → {out}  ({kb:.0f} KB)", flush=True)

plt.close(fig)
print("Done.", flush=True)
