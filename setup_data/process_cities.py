"""
Batch-process all study cities: download POIs from OSM and clip NOAH flood
hazard data, producing clean GeoPackage files for each city.

Output per city (in data/processed/<city_slug>/):
    pois.gpkg             Standardised POI table
    flood_5yr.gpkg        5-year flood hazard polygons clipped to the city
    flood_100yr.gpkg      100-year flood hazard polygons clipped to the city
    summary.csv           POI counts and flood area stats

Also generates a POI map per city in output/<city_slug>_pois.png.

Usage:
    python setup_data/process_cities.py                 # all cities
    python setup_data/process_cities.py tuguegarao      # one city
    python setup_data/process_cities.py manila dagupan   # several cities
"""

import glob
import os
import sys
import traceback

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from setup_data.download_osm import download_pois

# ---------------------------------------------------------------------------
# City configuration
# ---------------------------------------------------------------------------

CITIES = {
    "manila": {
        "display_name": "Manila",
        "gadm_name_1": "MetropolitanManila",
        "gadm_name_2": ["Manila", "City of Manila", "CityOfManila"],
        "noah_province": "MetropolitanManila",
    },
    "san_fernando": {
        "display_name": "San Fernando",
        "gadm_name_1": "Pampanga",
        "gadm_name_2": ["City of San Fernando", "San Fernando", "SanFernandoCity"],
        "noah_province": "Pampanga",
    },
    "dagupan": {
        "display_name": "Dagupan",
        "gadm_name_1": "Pangasinan",
        "gadm_name_2": ["Dagupan City", "Dagupan", "DagupanCity"],
        "noah_province": "Pangasinan",
    },
    "cagayan_de_oro": {
        "display_name": "Cagayan de Oro",
        "gadm_name_1": "MisamisOriental",
        "gadm_name_2": ["Cagayan de Oro", "Cagayan de Oro City", "CagayanDeOro"],
        "noah_province": "Misamis Oriental",
    },
    "naga": {
        "display_name": "Naga",
        "gadm_name_1": "CamarinesSur",
        "gadm_name_2": ["City of Naga", "Naga City", "Naga"],
        "noah_province": "Camarines Sur",
    },
    "butuan": {
        "display_name": "Butuan",
        "gadm_name_1": "AgusandelNorte",
        "gadm_name_2": ["Butuan City", "Butuan", "ButuanCity", "City of Butuan"],
        "noah_province": "Agusan del Norte",
    },
    "tuguegarao": {
        "display_name": "Tuguegarao",
        "gadm_name_1": "Cagayan",
        "gadm_name_2": ["TuguegaraoCity", "Tuguegarao City", "Tuguegarao"],
        "noah_province": "Cagayan",
    },
    "cotabato": {
        "display_name": "Cotabato",
        "gadm_name_1": "Maguindanao",
        "gadm_name_2": ["Cotabato City", "Cotabato", "CotabatoCity"],
        "noah_province": "Maguindanao",
    },
    "daet": {
        "display_name": "Daet",
        "gadm_name_1": "CamarinesNorte",
        "gadm_name_2": ["Daet"],
        "noah_province": "Camarines Norte",
    },
    "ilagan": {
        "display_name": "Ilagan",
        "gadm_name_1": "Isabela",
        "gadm_name_2": ["Ilagan", "City of Ilagan", "IlaganCity"],
        "noah_province": "Isabela",
    },
}

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json"
NOAH_DIR = os.path.join(ROOT, "data", "noah")
OUTPUT_DIR = os.path.join(ROOT, "data", "processed")
MAP_DIR = os.path.join(ROOT, "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gadm():
    """Load GADM level-2 boundaries for the Philippines."""
    print("Loading GADM municipality boundaries...")
    return gpd.read_file(GADM_URL)


def find_municipality(ph_admin, name_1, candidates):
    """
    Try each candidate NAME_2 value against GADM for the given province.
    Falls back to case-insensitive substring matching.
    """
    province = ph_admin[ph_admin["NAME_1"] == name_1]
    if province.empty:
        province = ph_admin[ph_admin["NAME_1"].str.lower() == name_1.lower()]
    if province.empty:
        print(f"  [WARN] Province '{name_1}' not found in GADM.")
        available = ph_admin["NAME_1"].unique().tolist()
        print(f"         Available NAME_1 values containing partial match:")
        for a in sorted(available):
            if name_1.lower().split()[0] in a.lower():
                print(f"           - {a}")
        return None, None

    for candidate in candidates:
        match = province[province["NAME_2"] == candidate]
        if not match.empty:
            return match, candidate
        match = province[province["NAME_2"].str.lower() == candidate.lower()]
        if not match.empty:
            return match, match.iloc[0]["NAME_2"]

    for candidate in candidates:
        token = candidate.lower().replace("city of ", "").replace(" city", "").strip()
        match = province[province["NAME_2"].str.lower().str.contains(token, na=False)]
        if not match.empty:
            return match.head(1), match.iloc[0]["NAME_2"]

    available = province["NAME_2"].sort_values().tolist()
    print(f"  [WARN] Municipality not found in '{name_1}'. Tried: {candidates}")
    print(f"         Available NAME_2 values:")
    for m in available:
        print(f"           - {m}")
    return None, None


def find_flood_shapefile(province, return_period):
    """
    Find the .shp file for a given province and return period.
    Checks several folder name variants because NOAH zips use inconsistent
    naming (e.g., 'Camarines Norte' vs 'CamarinesNorte' vs 'camarinesnorte').
    """
    # Build candidate folder names: spaced, CamelCase (no spaces), all-lowercase
    camel = province.replace(" ", "").replace("del", "del")
    candidates = [
        province,                      # "Camarines Norte"
        camel,                         # "CamarinesNorte"
        camel.lower(),                 # "camarinesnorte"
        province.replace(" ", ""),     # "CamarinesNorte" (same as camel but covers edge cases)
    ]
    for folder in dict.fromkeys(candidates):  # deduplicate, preserve order
        base = os.path.join(NOAH_DIR, return_period, folder)
        if os.path.isdir(base):
            shp_files = glob.glob(os.path.join(base, "*.shp"))
            if shp_files:
                return shp_files[0]
    return None


def make_poi_map(city_slug, cfg, muni, pois_df):
    """Generate and save a POI map for the city."""
    fig, ax = plt.subplots(figsize=(12, 12))
    muni.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, zorder=1)

    category_colors = {
        "Hospitals & Health Centers": "#E41A1C",
        "Schools": "#377EB8",
        "Markets & Grocery": "#4DAF4A",
        "Pharmacies": "#984EA3",
        "Barangay Halls / Gov't": "#FF7F00",
    }
    category_markers = {
        "Hospitals & Health Centers": "+",
        "Schools": "s",
        "Markets & Grocery": "^",
        "Pharmacies": "D",
        "Barangay Halls / Gov't": "P",
    }

    legend_elements = []
    for cat in pois_df["category"].unique():
        subset = pois_df[pois_df["category"] == cat]
        color = category_colors.get(cat, "#333333")
        marker = category_markers.get(cat, "o")
        subset.plot(
            ax=ax, color=color, marker=marker, markersize=30,
            edgecolor="white", linewidth=0.3, zorder=5,
        )
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="none",
                   markerfacecolor=color, markeredgecolor=color,
                   markersize=8, label=f"{cat} ({len(subset)})")
        )

    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              frameon=True, fancybox=True, framealpha=0.9)
    ax.set_title(f"Critical Facilities — {cfg['display_name']}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    os.makedirs(MAP_DIR, exist_ok=True)
    path = os.path.join(MAP_DIR, f"{city_slug}_pois.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Map saved: {path}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_city(city_slug, cfg, ph_admin):
    """Process a single city: POIs + flood hazard data."""
    display = cfg["display_name"]
    print(f"\n{'='*60}")
    print(f"  Processing: {display}")
    print(f"{'='*60}")

    # --- Municipality boundary ---
    muni, matched_name = find_municipality(
        ph_admin, cfg["gadm_name_1"], cfg["gadm_name_2"]
    )
    if muni is None:
        print(f"  [SKIP] Could not find municipality for {display}.")
        return False

    print(f"  GADM match: NAME_1='{cfg['gadm_name_1']}', NAME_2='{matched_name}'")
    muni_polygon = muni.geometry.union_all()

    # --- POIs from OSM ---
    pois_df, _ = download_pois(muni_polygon)

    if pois_df.empty:
        print(f"  [WARN] No POIs found for {display}.")

    # --- Flood data (clip to city boundary) ---
    province = cfg["noah_province"]

    flood_5yr_clipped = None
    flood_5yr_path = find_flood_shapefile(province, "5yr")
    if flood_5yr_path:
        print(f"  Loading 5yr flood: {os.path.basename(flood_5yr_path)}")
        flood_5yr = gpd.read_file(flood_5yr_path)
        flood_5yr_clipped = gpd.clip(flood_5yr, muni)
    else:
        print(f"  [INFO] No 5yr flood data for {province}")

    flood_100yr_clipped = None
    flood_100yr_path = find_flood_shapefile(province, "100yr")
    if flood_100yr_path:
        print(f"  Loading 100yr flood: {os.path.basename(flood_100yr_path)}")
        flood_100yr = gpd.read_file(flood_100yr_path)
        flood_100yr_clipped = gpd.clip(flood_100yr, muni)
    else:
        print(f"  [INFO] No 100yr flood data for {province}")

    # --- Save outputs ---
    city_dir = os.path.join(OUTPUT_DIR, city_slug)
    os.makedirs(city_dir, exist_ok=True)

    if not pois_df.empty:
        pois_save = _clean_for_gpkg(pois_df)
        pois_save.to_file(os.path.join(city_dir, "pois.gpkg"), driver="GPKG")
        print(f"  Saved pois.gpkg ({len(pois_save)} rows)")

    if flood_5yr_clipped is not None and not flood_5yr_clipped.empty:
        flood_5yr_clipped.to_file(
            os.path.join(city_dir, "flood_5yr.gpkg"), driver="GPKG"
        )
        print(f"  Saved flood_5yr.gpkg ({len(flood_5yr_clipped)} polygons)")

    if flood_100yr_clipped is not None and not flood_100yr_clipped.empty:
        flood_100yr_clipped.to_file(
            os.path.join(city_dir, "flood_100yr.gpkg"), driver="GPKG"
        )
        print(f"  Saved flood_100yr.gpkg ({len(flood_100yr_clipped)} polygons)")

    # --- Summary CSV ---
    summary = {
        "city": display,
        "province": cfg["gadm_name_1"],
        "gadm_name_2": matched_name,
        "total_pois": len(pois_df),
    }
    for cat in pois_df["category"].unique() if not pois_df.empty else []:
        summary[f"poi_{cat}"] = len(pois_df[pois_df["category"] == cat])

    if flood_5yr_clipped is not None and not flood_5yr_clipped.empty:
        summary["flood_5yr_polygons"] = len(flood_5yr_clipped)
    if flood_100yr_clipped is not None and not flood_100yr_clipped.empty:
        summary["flood_100yr_polygons"] = len(flood_100yr_clipped)

    pd.DataFrame([summary]).to_csv(
        os.path.join(city_dir, "summary.csv"), index=False
    )
    print(f"  Saved summary.csv")

    # --- POI map ---
    if not pois_df.empty:
        make_poi_map(city_slug, cfg, muni, pois_df)

    print(f"  Done: {display}")
    return True


def _clean_for_gpkg(gdf):
    """Drop columns that can't serialize to GeoPackage (lists, dicts, etc.)."""
    out = gdf.copy()
    drop = []
    for col in out.columns:
        if col == "geometry":
            continue
        if out[col].dtype == object:
            sample = out[col].dropna().head(5)
            if any(isinstance(v, (list, dict, set, tuple)) for v in sample):
                drop.append(col)
    if drop:
        out = out.drop(columns=drop)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    requested = [a.lower() for a in sys.argv[1:]]
    if requested:
        cities_to_run = {k: v for k, v in CITIES.items() if k in requested}
        unknown = set(requested) - set(cities_to_run.keys())
        if unknown:
            print(f"Unknown city slugs: {unknown}")
            print(f"Available: {list(CITIES.keys())}")
            return
    else:
        cities_to_run = CITIES

    ph_admin = load_gadm()

    results = {}
    for slug, cfg in cities_to_run.items():
        try:
            ok = process_city(slug, cfg, ph_admin)
            results[slug] = "OK" if ok else "SKIPPED"
        except Exception as e:
            print(f"\n  [ERROR] {cfg['display_name']}: {e}")
            traceback.print_exc()
            results[slug] = f"ERROR: {e}"

    # Final report
    print(f"\n\n{'='*60}")
    print(f"  BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    for slug, status in results.items():
        print(f"  {CITIES[slug]['display_name']:20s} {status}")
    print()
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Maps directory:   {MAP_DIR}")


if __name__ == "__main__":
    main()
