"""
Download and process OpenStreetMap data into clean, analysis-ready GeoDataFrames.

Provides reusable functions for:
- Downloading road networks for a given area
- Downloading and cleaning POIs (Points of Interest) with standardized categories
- Preparing data for spatial/network analysis (flood exposure, accessibility, etc.)

Usage:
    from setup_data.download_osm import download_roads, download_pois, snap_pois_to_network

    roads, G = download_roads(polygon)
    pois, _  = download_pois(polygon)
    pois     = snap_pois_to_network(G, pois)

Or run standalone to verify the pipeline:
    python -m setup_data.download_osm
"""

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

# ---------------------------------------------------------------------------
# POI category definitions
# ---------------------------------------------------------------------------

# Maps human-readable category names to the OSM tags that identify them.
# Each category can have multiple OSM key-value pairs. Values are lists so
# that several OSM tag values roll up into one logical category.
DEFAULT_POI_CATEGORIES = {
    "Hospitals & Health Centers": {
        "amenity": ["hospital", "clinic", "doctors", "health_post"],
    },
    "Schools": {
        "amenity": ["school", "university", "college"],
    },
    "Markets & Grocery": {
        "shop": ["supermarket", "grocery", "convenience"],
        "amenity": ["marketplace"],
    },
    "Pharmacies": {
        "amenity": ["pharmacy"],
    },
    "Barangay Halls / Gov't": {
        "office": ["government"],
        "amenity": ["townhall"],
    },
}

# Optional normalization map: collapses fine-grained OSM categories into
# broader groups for analysis. Keys are the human-readable category names
# above; values are the standardized group names.
DEFAULT_CATEGORY_NORMALIZATION = {
    "Hospitals & Health Centers": "healthcare",
    "Pharmacies": "healthcare",
    "Schools": "education",
    "Markets & Grocery": "essential_services",
    "Barangay Halls / Gov't": "government",
}

# Default UTM zone for the Philippines (Zone 51N, EPSG:32651).
# Used when projecting to metres for centroid computation and length measurement.
PH_UTM_EPSG = 32651


# ---------------------------------------------------------------------------
# Road network
# ---------------------------------------------------------------------------

def download_roads(polygon, network_type="drive", crs="EPSG:4326"):
    """
    Download the drivable road network inside *polygon* from OpenStreetMap.

    Parameters
    ----------
    polygon : shapely Polygon or MultiPolygon
        Area of interest.
    network_type : str
        OSMnx network type (``"drive"``, ``"walk"``, ``"bike"``, ``"all"``).
    crs : str
        Target CRS for the returned GeoDataFrame.

    Returns
    -------
    roads : GeoDataFrame
        Edge geometries (LineStrings) with OSM attributes.
    G : networkx.MultiDiGraph
        The full graph object (useful for network analysis).
    """
    polygon = _ensure_polygon(polygon)
    print("Downloading road network from OpenStreetMap...")
    G = ox.graph_from_polygon(polygon, network_type=network_type)
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
    if roads.crs and str(roads.crs) != crs:
        roads = roads.to_crs(crs)
    print(f"  Road segments downloaded: {len(roads)}")
    return roads, G


# ---------------------------------------------------------------------------
# Points of Interest
# ---------------------------------------------------------------------------

def download_pois(
    polygon,
    categories=None,
    normalize=True,
    normalization_map=None,
    utm_epsg=PH_UTM_EPSG,
    target_crs="EPSG:4326",
):
    """
    Download POIs from OpenStreetMap inside *polygon*, clean them, and return
    a single analysis-ready GeoDataFrame.

    Processing steps
    ~~~~~~~~~~~~~~~~
    1. Query OSM for each category's tags via ``osmnx.features_from_polygon``.
    2. Convert polygon/multipolygon geometries to point centroids.
    3. Drop rows with missing or empty geometries.
    4. Remove duplicate features (same ``osmid``).
    5. Clip to the exact *polygon* boundary.
    6. Assign a unique ``poi_id`` and a ``category`` column.
    7. Optionally normalize categories into broader groups.
    8. Return a tidy GeoDataFrame with a consistent schema.

    Parameters
    ----------
    polygon : shapely Polygon or MultiPolygon
        Area of interest.
    categories : dict or None
        Category definitions in the same format as ``DEFAULT_POI_CATEGORIES``.
        If *None*, the built-in defaults are used.
    normalize : bool
        Whether to add a ``category_group`` column using *normalization_map*.
    normalization_map : dict or None
        Mapping from category name → group name. Uses defaults if *None*.
    utm_epsg : int
        EPSG code for a projected CRS used during centroid computation.
    target_crs : str
        CRS for the returned GeoDataFrame.

    Returns
    -------
    pois : GeoDataFrame
        Cleaned POIs with columns:
        ``poi_id | name | category | category_group | lon | lat | geometry``
    per_category : dict[str, GeoDataFrame]
        The same POIs split by original category name (convenient for plotting).
    """
    polygon = _ensure_polygon(polygon)
    categories = categories or DEFAULT_POI_CATEGORIES
    normalization_map = normalization_map or DEFAULT_CATEGORY_NORMALIZATION

    print("Downloading points of interest from OpenStreetMap...")

    per_category: dict[str, gpd.GeoDataFrame] = {}
    all_frames = []

    for category_name, tags in categories.items():
        frames = _query_osm_tags(polygon, tags)

        if not frames:
            print(f"  {category_name}: 0 features")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = _geometries_to_points(combined, utm_epsg, target_crs)
        combined = _drop_bad_geometry(combined)
        combined = _deduplicate(combined)
        combined = _clip_to_boundary(combined, polygon, target_crs)

        if combined.empty:
            print(f"  {category_name}: 0 features (after cleaning)")
            continue

        combined["category"] = category_name
        per_category[category_name] = combined
        all_frames.append(combined)
        print(f"  {category_name}: {len(combined)} features")

    if not all_frames:
        empty = gpd.GeoDataFrame(
            columns=["poi_id", "name", "category", "category_group",
                      "lon", "lat", "geometry"],
            geometry="geometry",
            crs=target_crs,
        )
        return empty, per_category

    pois = pd.concat(all_frames, ignore_index=True)
    pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=target_crs)

    # Assign unique POI IDs
    pois = pois.reset_index(drop=True)
    pois.insert(0, "poi_id", range(1, len(pois) + 1))

    # Extract name, lon, lat into explicit columns
    if "name" not in pois.columns:
        pois["name"] = None
    pois["lon"] = pois.geometry.x
    pois["lat"] = pois.geometry.y

    # Normalize categories into broader groups
    if normalize:
        pois["category_group"] = pois["category"].map(normalization_map)

    # Select output columns (keep extras from OSM as well)
    front_cols = ["poi_id", "name", "category"]
    if normalize:
        front_cols.append("category_group")
    front_cols += ["lon", "lat", "geometry"]
    other_cols = [c for c in pois.columns if c not in front_cols]
    pois = pois[front_cols + other_cols]

    print(f"  Total usable POIs: {len(pois)}")
    return pois, per_category


# ---------------------------------------------------------------------------
# Network snapping
# ---------------------------------------------------------------------------

def snap_pois_to_network(G, pois_df, utm_epsg=PH_UTM_EPSG):
    """
    Snap each POI to its nearest existing node in the road network graph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Road network graph returned by ``download_roads``.
    pois_df : GeoDataFrame
        Standardized POI table with ``lon`` and ``lat`` columns
        (as returned by ``download_pois``).
    utm_epsg : int
        EPSG code for a projected CRS, used to compute snap distances in metres.

    Returns
    -------
    pois_snapped : GeoDataFrame
        Copy of *pois_df* with two new columns:
        - ``snapped_node``: osmid of the nearest graph node.
        - ``snap_distance_m``: distance in metres from the POI to that node.
    """
    pois_snapped = pois_df.copy()

    # Project graph to UTM so nearest_nodes works without scikit-learn
    G_proj = ox.project_graph(G, to_crs=f"EPSG:{utm_epsg}")
    pois_proj = pois_snapped.to_crs(epsg=utm_epsg)

    snapped_nodes = ox.nearest_nodes(G_proj, pois_proj.geometry.x.values, pois_proj.geometry.y.values)
    pois_snapped["snapped_node"] = snapped_nodes

    # Compute snap distances in metres (already in projected CRS)
    nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
    node_geom = nodes_proj.geometry.reindex(pois_snapped["snapped_node"].values)
    node_geom_series = gpd.GeoSeries(node_geom.values, crs=pois_proj.crs).reset_index(drop=True)
    poi_geom_series = pois_proj.geometry.reset_index(drop=True)

    pois_snapped["snap_distance_m"] = poi_geom_series.distance(node_geom_series).values

    dists = pois_snapped["snap_distance_m"]
    print(f"  Snapped {len(pois_snapped)} POIs to nearest road nodes")
    print(f"  Median snap distance: {dists.median():.0f} m")
    print(f"  Max snap distance:    {dists.max():.0f} m")

    return pois_snapped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_polygon(geom):
    """Accept a Polygon, MultiPolygon, or GeoSeries and return a Polygon/MultiPolygon."""
    if isinstance(geom, (gpd.GeoSeries, gpd.GeoDataFrame)):
        geom = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")
    return geom


def _query_osm_tags(polygon, tags):
    """Query OSM for each key/value pair in *tags* and return a list of GeoDataFrames."""
    frames = []
    for key, values in tags.items():
        for val in values:
            try:
                gdf = ox.features_from_polygon(polygon, tags={key: val})
                if not gdf.empty:
                    frames.append(gdf)
            except Exception:
                pass
    return frames


def _geometries_to_points(gdf, utm_epsg, target_crs):
    """Convert any polygon/multipolygon geometries to their centroids."""
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    gdf = gdf.to_crs(epsg=utm_epsg)
    gdf["geometry"] = gdf.geometry.centroid
    gdf = gdf.to_crs(target_crs)
    return gdf


def _drop_bad_geometry(gdf):
    """Remove rows with null, empty, or invalid geometries."""
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def _deduplicate(gdf):
    """Remove duplicate features, preferring the first occurrence."""
    if "osmid" in gdf.columns:
        gdf = gdf.drop_duplicates(subset="osmid", keep="first")
    return gdf.loc[~gdf.index.duplicated(keep="first")]


def _clip_to_boundary(gdf, polygon, crs):
    """Keep only POIs that fall inside *polygon*."""
    boundary = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    clipped = gpd.sjoin(gdf, boundary, how="inner", predicate="within")
    clipped = clipped.drop(columns=["index_right"], errors="ignore")
    return clipped


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("download_osm.py — standalone test")
    print("Provide a place name to test (e.g. 'Tuguegarao City, Cagayan, Philippines'):")
    place = input("> ").strip() or "Tuguegarao City, Cagayan, Philippines"
    area = ox.geocode_to_gdf(place)
    poly = area.geometry.union_all()

    roads, _ = download_roads(poly)
    pois, by_cat = download_pois(poly)

    print(f"\nRoads: {len(roads)} segments")
    print(f"POIs:  {len(pois)} features")
    if not pois.empty:
        print("\nSample rows:")
        print(pois[["poi_id", "name", "category", "category_group", "lon", "lat"]].head(10).to_string())
        print("\nPer-category counts:")
        print(pois["category"].value_counts().to_string())
