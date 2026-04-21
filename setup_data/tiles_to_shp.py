"""
Convert PhilSA ``*_maps`` tile folders into zipped shapefiles.

This script is for PhilSA flood map products that were downloaded as PNG tiles
instead of shapefiles. It reads the lower-left tile coordinates directly from
each tile's embedded legend, extracts the red flood mask inside the map frame,
polygonizes it, and writes a zipped shapefile next to the source folder using
the same naming convention as the existing PhilSA files:

    data/philsa_satellite_flood/<event>_shp.zip

Example:
    python setup_data/tiles_to_shp.py --overwrite
    python setup_data/tiles_to_shp.py --folder 20240725_0600_fld_s1_maps

Notes
-----
- The OCR step requires ``rapidocr-onnxruntime`` in the active Python env.
- The printed map layout is standardized across these products, so the map
  frame crop and 0.5° x 0.5° tile span are shared constants.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as nd_label
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit(
        "Missing dependency: rapidocr-onnxruntime\n"
        "Install it in the Python environment you use for this script, e.g.\n"
        "  pip install rapidocr-onnxruntime\n"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
PHILSA_DIR = ROOT / "data" / "philsa_satellite_flood"

# Standardized printed tile layout
FRAME_LEFT = 55
FRAME_RIGHT = 2426
FRAME_TOP = 58
FRAME_BOTTOM = 2419

FRAME_W = FRAME_RIGHT - FRAME_LEFT
FRAME_H = FRAME_BOTTOM - FRAME_TOP

TILE_LON_SPAN = 0.5
TILE_LAT_SPAN = 0.5
PX_PER_DEG_LON = FRAME_W / TILE_LON_SPAN
PX_PER_DEG_LAT = FRAME_H / TILE_LAT_SPAN

COORD_RE = re.compile(
    r"lower\s*left\s*corner\s*coordinates[:\s]*"
    r"(\d{1,2})\D+(\d{2})\D*[ns]\D+"
    r"(\d{1,3})\D+(\d{2})\D*[ew]",
    re.IGNORECASE,
)
COORD_FALLBACK_RE = re.compile(
    r"(\d{1,2})\D+(\d{2})\D*[ns]\D+(\d{1,3})\D+(\d{2})\D*[ew]",
    re.IGNORECASE,
)
EVENT_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{4})_fld_(?P<sensor>[a-z0-9]+)_maps$",
    re.IGNORECASE,
)

SENSOR_LABELS = {
    "saocom": "SAOCOM 1A",
    "tdx": "TanDEM-X",
    "s1": "Sentinel-1",
    "s2": "Sentinel-2",
    "rcm": "RADARSAT CM",
    "alos2": "ALOS-2",
    "iceye": "ICEYE",
    "gf3": "Gaofen-3",
    "nv1": "NovaSAR-1",
    "k5": "KOMPSAT-5",
    "l9": "Landsat 9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        action="append",
        default=[],
        help="Specific *_maps folder name(s) under data/philsa_satellite_flood.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_shp.zip outputs.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=4,
        help="Minimum connected flood pixels to keep per component.",
    )
    parser.add_argument(
        "--simplify-tol",
        type=float,
        default=0.00005,
        help="Simplification tolerance in degrees after polygonization.",
    )
    return parser.parse_args()


def sensor_label(sensor_code: str) -> str:
    return SENSOR_LABELS.get(sensor_code.lower(), sensor_code.upper())


def extract_flood_mask(arr: np.ndarray) -> np.ndarray:
    """
    Extract the red flood mask from the map frame.

    These PhilSA products use bright red filled polygons for flooded areas.
    """
    frame = arr[FRAME_TOP:FRAME_BOTTOM, FRAME_LEFT:FRAME_RIGHT, :3].astype(np.int16)
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    return (r > 160) & (g < 90) & (b < 90)


def _iter_polygonal(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    if isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            yield from _iter_polygonal(part)


def _row_runs(indices: np.ndarray):
    if indices.size == 0:
        return
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx != prev + 1:
            yield start, prev + 1
            start = idx
        prev = idx
    yield start, prev + 1


def mask_to_polygons(
    mask: np.ndarray,
    lon_west: float,
    lat_south: float,
    min_pixels: int = 4,
    simplify_tol: float = 0.00005,
):
    """
    Polygonize a binary flood mask using row runs of flood pixels.

    The row-run approach keeps the geometry much closer to the underlying raster
    than a convex hull, while still staying tractable for many tiles.
    """
    labeled, n_features = nd_label(mask.astype(np.uint8))
    if n_features == 0:
        return []

    polys = []
    for comp_id in range(1, n_features + 1):
        comp = labeled == comp_id
        if int(comp.sum()) < min_pixels:
            continue

        rects = []
        rows = np.flatnonzero(comp.any(axis=1))
        for y in rows:
            xs = np.flatnonzero(comp[y])
            for x0, x1 in _row_runs(xs):
                lon_min = lon_west + (x0 / PX_PER_DEG_LON)
                lon_max = lon_west + (x1 / PX_PER_DEG_LON)
                lat_max = lat_south + TILE_LAT_SPAN - (y / PX_PER_DEG_LAT)
                lat_min = lat_south + TILE_LAT_SPAN - ((y + 1) / PX_PER_DEG_LAT)
                rects.append(box(lon_min, lat_min, lon_max, lat_max))

        if not rects:
            continue

        geom = unary_union(rects)
        geom = make_valid(geom)
        if simplify_tol > 0:
            geom = geom.simplify(simplify_tol, preserve_topology=True)
            geom = make_valid(geom)

        for poly in _iter_polygonal(geom):
            if not poly.is_empty and poly.area > 0:
                polys.append(poly)

    return polys


def ocr_lower_left(tile_path: Path, ocr: RapidOCR) -> tuple[float, float]:
    """
    Read lower-left corner coordinates from the tile legend.

    Returns (lon_west, lat_south) in decimal degrees.
    """
    img = Image.open(tile_path).convert("RGB")
    w, h = img.size
    crop_specs = [
        (int(w * 0.69), int(h * 0.30), w, int(h * 0.63)),
        (int(w * 0.67), int(h * 0.22), w, int(h * 0.72)),
    ]

    texts = []
    for left, top, right, bottom in crop_specs:
        crop = img.crop((left, top, right, bottom))
        result, _ = ocr(np.asarray(crop))
        if result:
            texts.extend(item[1] for item in result)
        joined = " ".join(texts)
        match = COORD_RE.search(joined) or COORD_FALLBACK_RE.search(joined)
        if match:
            lat_deg, lat_min, lon_deg, lon_min = map(int, match.groups())
            lat = lat_deg + (lat_min / 60.0)
            lon = lon_deg + (lon_min / 60.0)
            return lon, lat

    # Slow fallback: OCR the whole image if the legend crop failed.
    result, _ = ocr(np.asarray(img))
    if result:
        joined = " ".join(item[1] for item in result)
        match = COORD_RE.search(joined) or COORD_FALLBACK_RE.search(joined)
        if match:
            lat_deg, lat_min, lon_deg, lon_min = map(int, match.groups())
            lat = lat_deg + (lat_min / 60.0)
            lon = lon_deg + (lon_min / 60.0)
            return lon, lat

    raise ValueError(f"Could not OCR lower-left coordinates from {tile_path.name}")


def parse_event_metadata(map_dir: Path) -> dict:
    match = EVENT_RE.match(map_dir.name)
    if not match:
        raise ValueError(f"Unexpected map folder name: {map_dir.name}")

    date_str = match.group("date")
    time_str = match.group("time")
    sensor = match.group("sensor").lower()
    event_key = map_dir.name.replace("_maps", "")
    event_date = pd.to_datetime(date_str, format="%Y%m%d")

    return {
        "event_key": event_key,
        "event_date": event_date,
        "event_time": time_str,
        "sensor_code": sensor,
        "sensor_label": sensor_label(sensor),
        "out_zip": PHILSA_DIR / f"{event_key}_shp.zip",
        "shp_stem": event_key,
    }


def convert_folder(
    map_dir: Path,
    ocr: RapidOCR,
    overwrite: bool = False,
    min_pixels: int = 4,
    simplify_tol: float = 0.00005,
) -> dict | None:
    meta = parse_event_metadata(map_dir)
    out_zip = meta["out_zip"]

    if out_zip.exists() and not overwrite:
        print(f"SKIP {map_dir.name} -> {out_zip.name} already exists")
        return None

    tiles = sorted(map_dir.glob("Tile_No_*.png"))
    if not tiles:
        print(f"WARN {map_dir.name}: no PNG tiles found")
        return None

    print(f"\n[{map_dir.name}]")
    print(f"  tiles={len(tiles)}")

    parts = []
    for tile_path in tiles:
        tile_num = int(tile_path.stem.replace("Tile_No_", ""))
        lon_west, lat_south = ocr_lower_left(tile_path, ocr)

        img = Image.open(tile_path).convert("RGB")
        arr = np.asarray(img)
        mask = extract_flood_mask(arr)
        flood_px = int(mask.sum())
        if flood_px == 0:
            print(f"  {tile_path.name}: no flood pixels")
            continue

        polys = mask_to_polygons(
            mask,
            lon_west=lon_west,
            lat_south=lat_south,
            min_pixels=min_pixels,
            simplify_tol=simplify_tol,
        )
        if not polys:
            print(f"  {tile_path.name}: flood_px={flood_px:,} but no polygons")
            continue

        gdf_tile = gpd.GeoDataFrame(
            {
                "Tile": [tile_num] * len(polys),
                "FloodPx": [flood_px] * len(polys),
                "LonWest": [lon_west] * len(polys),
                "LatSouth": [lat_south] * len(polys),
                "SrcMap": [map_dir.name] * len(polys),
            },
            geometry=polys,
            crs=4326,
        )
        area_km2 = float(gdf_tile.to_crs(epsg=32651).geometry.area.sum() / 1e6)
        print(
            f"  {tile_path.name}: ll=({lat_south:.2f}N, {lon_west:.2f}E) | "
            f"flood_px={flood_px:,} | polys={len(polys):,} | area={area_km2:.2f} km^2"
        )
        parts.append(gdf_tile)

    if not parts:
        print(f"  WARN {map_dir.name}: no polygons extracted")
        return None

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    merged["DN"] = 1
    merged["ImageDate"] = meta["event_date"].date()
    merged["DataSource"] = meta["sensor_label"]

    cols = [
        "DN",
        "ImageDate",
        "DataSource",
        "Tile",
        "FloodPx",
        "LonWest",
        "LatSouth",
        "SrcMap",
        "geometry",
    ]
    merged = merged[cols]

    total_area_km2 = float(merged.to_crs(epsg=32651).geometry.area.sum() / 1e6)
    bounds = np.round(merged.total_bounds, 4).tolist()

    with tempfile.TemporaryDirectory(prefix="philsa_tiles_") as tmp:
        shp_stem = meta["shp_stem"]
        shp_path = Path(tmp) / f"{shp_stem}.shp"
        merged.to_file(shp_path, driver="ESRI Shapefile")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                part = Path(tmp) / f"{shp_stem}{ext}"
                if part.exists():
                    zf.write(part, arcname=part.name)

    print(
        f"  saved={out_zip.name} | polygons={len(merged):,} | "
        f"area={total_area_km2:.2f} km^2 | bounds={bounds}"
    )

    return {
        "folder": map_dir.name,
        "out_zip": str(out_zip),
        "tiles": len(tiles),
        "polygons": int(len(merged)),
        "area_km2": total_area_km2,
        "bounds": bounds,
    }


def main() -> int:
    args = parse_args()
    if args.folder:
        map_dirs = [PHILSA_DIR / name for name in args.folder]
    else:
        map_dirs = sorted(PHILSA_DIR.glob("*_maps"))

    missing = [p for p in map_dirs if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR missing folder: {p}")
        return 1

    if not map_dirs:
        print(f"No *_maps folders found under {PHILSA_DIR}")
        return 1

    print("=" * 72)
    print("  PhilSA tile folders -> shapefile zips")
    print("=" * 72)
    print(f"Input root: {PHILSA_DIR}")
    print(f"Folders: {len(map_dirs)}")

    ocr = RapidOCR()
    results = []
    for map_dir in map_dirs:
        result = convert_folder(
            map_dir,
            ocr=ocr,
            overwrite=args.overwrite,
            min_pixels=args.min_pixels,
            simplify_tol=args.simplify_tol,
        )
        if result:
            results.append(result)

    print("\nSummary")
    print("-" * 72)
    if not results:
        print("No outputs written.")
        return 0

    for row in results:
        print(
            f"{row['folder']}: polygons={row['polygons']:,} | "
            f"area={row['area_km2']:.2f} km^2 | {Path(row['out_zip']).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
