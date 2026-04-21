"""
download_gfd.py
===============
Download Global Flood Database (GFD) flood events for the Philippines
from Google Earth Engine (Tellman et al. 2021, Nature).

Source
------
  GEE collection : GLOBAL_FLOOD_DB/MODIS_EVENTS/V1
  Coverage       : 2000–2018, 913 global events, 250 m MODIS
  Band used      : 'flooded' (1 = flooded, 0 = not flooded)

Setup (run once)
----------------
  python3 -c "import ee; ee.Authenticate()"
  # complete browser sign-in, then run this script

Output
------
  data/gfd/<YYYYMMDD>_<event_id>_fld_gfd_shp.zip
  Each zip contains a shapefile with columns:
    DN         int    1 (flooded)
    ImageDate  date   event start date
    DataSource str    "GFD/MODIS"
    event_id   str    GEE image id
  CRS: EPSG:4326

Run
---
  python3 setup_data/download_gfd.py
"""

import io
import os
import struct
import sys
import zipfile
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import label as ndlabel
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT, "data", "gfd")
os.makedirs(OUT_DIR, exist_ok=True)

# Philippines bounding box (with generous buffer)
PHL_BBOX = [116.5, 4.5, 127.5, 21.5]   # [lon_min, lat_min, lon_max, lat_max]

# City buffers used to filter events (keep only events that intersect any city)
CITY_BUFFERS = [
    {"name": "Tuguegarao",    "lat": 17.6158, "lng": 121.7229, "r_deg": 0.15},
    {"name": "Dagupan",       "lat": 16.0431, "lng": 120.3333, "r_deg": 0.15},
    {"name": "Manila",        "lat": 14.5995, "lng": 120.9842, "r_deg": 0.25},
    {"name": "Cagayan de Oro","lat":  8.4772, "lng": 124.6459, "r_deg": 0.15},
    {"name": "Cotabato",      "lat":  7.2236, "lng": 124.2464, "r_deg": 0.15},
]

CITY_BOXES = [
    box(c["lng"] - c["r_deg"], c["lat"] - c["r_deg"],
        c["lng"] + c["r_deg"], c["lat"] + c["r_deg"])
    for c in CITY_BUFFERS
]
CITY_UNION = unary_union(CITY_BOXES)

# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------
def _init_ee():
    import ee
    try:
        ee.Initialize(project="project-4a1e5493-d3e7-4a90-989")
    except Exception:
        try:
            ee.Initialize()
        except Exception as ex:
            print(f"ERROR: GEE init failed — {ex}")
            print("Run:  python3 -c \"import ee; ee.Authenticate()\"  then retry.")
            sys.exit(1)
    return ee


# ---------------------------------------------------------------------------
# Download a single event raster over PHL bbox via getDownloadURL
# ---------------------------------------------------------------------------
def _download_city_array(ee_mod, img, city_bbox):
    """Download flooded band for a single city bbox. Returns (arr, transform)."""
    import tempfile
    from PIL import Image as PILImage

    lon_min, lat_min, lon_max, lat_max = city_bbox
    region = ee_mod.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    url = img.select("flooded").getDownloadURL({
        "region": region,
        "scale":  250,
        "format": "GEO_TIFF",
        "crs":    "EPSG:4326",
    })
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
        tf.write(resp.content)
        tpath = tf.name

    try:
        img_pil = PILImage.open(tpath)
        arr     = np.array(img_pil)
        tag     = img_pil.tag_v2
        px      = tag.get(33550, (0.002245, 0.002245, 0))
        tp      = tag.get(33922, (0, 0, 0, lon_min, lat_max, 0))
        transform = (float(tp[3]), float(px[0]), float(tp[4]), -float(px[1]))
    finally:
        os.unlink(tpath)

    return arr, transform


# ---------------------------------------------------------------------------
# Vectorise a binary flood raster → GeoDataFrame
# ---------------------------------------------------------------------------
def _vectorise(arr, transform, min_px=4):
    """
    arr       : 2D numpy array (0/1)
    transform : (lon_west, dx, lat_north, dy_negative)
    Returns GeoDataFrame with Polygon geometries in EPSG:4326.
    """
    lon_west, dx, lat_north, dy = transform
    # Connected-component label
    labeled, n_feat = ndlabel(arr > 0)
    polys = []
    for fid in range(1, n_feat + 1):
        rows, cols = np.where(labeled == fid)
        if len(rows) < min_px:
            continue
        lons = lon_west + cols * dx
        lats = lat_north + rows * dy          # dy is negative
        # Build convex hull of pixel corners
        pts = []
        half_x, half_y = dx / 2, abs(dy) / 2
        for lon, lat in zip(lons, lats):
            pts += [(lon - half_x, lat - half_y),
                    (lon + half_x, lat - half_y),
                    (lon + half_x, lat + half_y),
                    (lon - half_x, lat + half_y)]
        from shapely.geometry import MultiPoint
        hull = MultiPoint(pts).convex_hull
        if hull.is_valid and hull.area > 0:
            polys.append(hull)

    if not polys:
        return gpd.GeoDataFrame(columns=["geometry"], crs=4326)
    return gpd.GeoDataFrame(geometry=polys, crs=4326)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ee = _init_ee()
    import ee as ee_mod

    print("=" * 68)
    print("Global Flood Database → Philippines shapefiles")
    print("=" * 68)

    col = ee_mod.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

    # Filter to Philippines bbox
    phl_region = ee_mod.Geometry.Rectangle(PHL_BBOX)
    col_phl    = col.filterBounds(phl_region)

    info  = col_phl.getInfo()
    feats = info["features"]
    print(f"  {len(feats)} GFD events intersect Philippines bbox", flush=True)

    saved = skipped = errors = 0

    for feat in feats:
        props   = feat["properties"]
        img_id  = feat["id"]
        short   = img_id.split("/")[-1]

        # Date: GFD stores 'system:time_start' in ms
        ts_ms   = props.get("system:time_start", 0)
        dt      = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y%m%d")
        out_zip  = os.path.join(OUT_DIR, f"{date_str}_{short}_fld_gfd_shp.zip")

        if os.path.exists(out_zip):
            skipped += 1
            continue

        print(f"  {date_str} {short} … ", end="", flush=True)

        try:
            img = ee_mod.Image(img_id)

            # Download one small tile per city (avoids 50 MB GEE limit)
            city_gdfs = []
            for c in CITY_BUFFERS:
                pad = c["r_deg"] + 0.05
                bbox = [c["lng"]-pad, c["lat"]-pad, c["lng"]+pad, c["lat"]+pad]
                try:
                    arr, transform = _download_city_array(ee_mod, img, bbox)
                    if arr is not None and arr.sum() > 0:
                        g = _vectorise(arr, transform)
                        if not g.empty:
                            city_gdfs.append(g)
                except Exception as ce:
                    pass   # city had no data or request failed

            if not city_gdfs:
                print("no flood pixels in any city")
                skipped += 1
                continue

            gdf_city = pd.concat(city_gdfs, ignore_index=True)
            gdf_city = gpd.GeoDataFrame(gdf_city, geometry="geometry", crs=4326)
            gdf_city = gdf_city[gdf_city.geometry.intersects(CITY_UNION)].copy()
            if gdf_city.empty:
                print("no overlap with city buffers")
                skipped += 1
                continue

            gdf_city["DN"]         = 1
            gdf_city["ImageDate"]  = dt.date()
            gdf_city["DataSource"] = "GFD/MODIS"
            gdf_city["event_id"]   = short

            # Save to zip
            import tempfile, shutil
            tmp = tempfile.mkdtemp()
            shp_path = os.path.join(tmp, "flood.shp")
            gdf_city[["DN", "ImageDate", "DataSource", "event_id", "geometry"]].to_file(
                shp_path, driver="ESRI Shapefile"
            )
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in os.listdir(tmp):
                    zf.write(os.path.join(tmp, f), f)
            shutil.rmtree(tmp)

            kb = os.path.getsize(out_zip) / 1024
            print(f"{len(gdf_city)} polys → {kb:.0f} KB", flush=True)
            saved += 1

        except Exception as ex:
            print(f"ERROR: {ex}", flush=True)
            errors += 1

    print(f"\nDone: {saved} saved | {skipped} skipped | {errors} errors")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
