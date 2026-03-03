import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import osmnx as ox
import pandas as pd

# Configuration
MUNICIPALITY = "LipaCity"
DISPLAY_NAME = "Lipa City"

# Load data
print("Loading flood hazard data...")
flood_5yr = gpd.read_file("data/noah/5yr/Batangas/Batangas_Flood_5yr.shp")

print("Loading municipality boundaries...")
ph_admin = gpd.read_file(
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json"
)
muni = ph_admin[
    (ph_admin["NAME_1"] == "Batangas") & (ph_admin["NAME_2"] == MUNICIPALITY)
]

if muni.empty:
    available = ph_admin[ph_admin["NAME_1"] == "Batangas"]["NAME_2"].sort_values().tolist()
    print(f"Municipality '{MUNICIPALITY}' not found. Available options:")
    for m in available:
        print(f"  - {m}")
    raise SystemExit(1)

print(f"Using municipality: {DISPLAY_NAME}")
muni_polygon = muni.geometry.union_all()

# Clip flood data to municipality
print("Clipping flood data to municipality bounds...")
flood_clipped = gpd.clip(flood_5yr, muni)

# Download road network from OSM
print("Downloading road network from OpenStreetMap...")
G = ox.graph_from_polygon(muni_polygon, network_type="drive")
roads = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Download Points of Interest from OSM
print("Downloading points of interest from OpenStreetMap...")

poi_queries = {
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

# Marker styles for each POI category
poi_styles = {
    "Hospitals & Health Centers": {"color": "#E41A1C", "marker": "+", "size": 60},
    "Schools": {"color": "#377EB8", "marker": "s", "size": 30},
    "Markets & Grocery": {"color": "#4DAF4A", "marker": "^", "size": 30},
    "Pharmacies": {"color": "#984EA3", "marker": "D", "size": 25},
    "Barangay Halls / Gov't": {"color": "#FF7F00", "marker": "P", "size": 40},
}

all_pois = {}

for category, tags in poi_queries.items():
    frames = []
    for key, values in tags.items():
        for val in values:
            try:
                gdf = ox.features_from_polygon(muni_polygon, tags={key: val})
                if not gdf.empty:
                    frames.append(gdf)
            except Exception:
                pass  # No features found for this tag

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Convert polygons (e.g. building footprints) to centroids
        combined = combined.to_crs(epsg=32651)
        combined["geometry"] = combined.geometry.centroid
        combined = combined.to_crs(epsg=4326)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        all_pois[category] = combined
        print(f"  {category}: {len(combined)} features")
    else:
        print(f"  {category}: 0 features")

# Identify flooded roads
print("Identifying flooded road segments...")
roads_in_flood = gpd.sjoin(roads, flood_clipped, how="inner", predicate="intersects")
flooded_idx = roads_in_flood.index.unique()
roads_safe = roads[~roads.index.isin(flooded_idx)]

roads_max_hazard = roads_in_flood.groupby(roads_in_flood.index)["Var"].max()
roads_in_flood = roads_in_flood[~roads_in_flood.index.duplicated(keep="first")].copy()
roads_in_flood["max_hazard"] = roads_max_hazard

# Identify flooded POIs
print("Identifying POIs in flood zones...")
poi_flood_summary = {}

for category, pois in all_pois.items():
    flooded = gpd.sjoin(pois, flood_clipped, how="inner", predicate="within")
    # Deduplicate (a POI might intersect multiple flood polygons)
    flooded = flooded[~flooded.index.duplicated(keep="first")]
    n_total = len(pois)
    n_flooded = len(flooded)
    poi_flood_summary[category] = {
        "total": n_total,
        "flooded": n_flooded,
        "pct": (n_flooded / n_total * 100) if n_total > 0 else 0,
    }

# Plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(14, 14))

# Municipality boundary
muni.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, zorder=1)

# Flood hazard zones
hazard_cmap = ListedColormap(["#FFD700", "#FF8C00", "#CC0000"])
flood_clipped.plot(
    column="Var",
    cmap=hazard_cmap,
    vmin=1,
    vmax=3,
    ax=ax,
    alpha=0.35,
    edgecolor="none",
    zorder=2,
)

# Safe roads (light gray)
roads_safe.plot(ax=ax, color="#CCCCCC", linewidth=0.4, zorder=3)

# Flooded roads colored by hazard level
hazard_colors = {1: "#FFD700", 2: "#FF8C00", 3: "#CC0000"}
for hazard_level, color in hazard_colors.items():
    subset = roads_in_flood[roads_in_flood["max_hazard"] == hazard_level]
    if not subset.empty:
        subset.plot(ax=ax, color=color, linewidth=1.0, zorder=4)

# POIs
for category, pois in all_pois.items():
    style = poi_styles[category]
    pois.plot(
        ax=ax,
        color=style["color"],
        marker=style["marker"],
        markersize=style["size"],
        zorder=5,
        edgecolor="white" if style["marker"] not in ["+", "x"] else style["color"],
        linewidth=0.3,
    )

# Legend
legend_elements = [
    # Flood hazard
    Patch(facecolor="#FFD700", alpha=0.4, label="Low hazard (0–0.5 m)"),
    Patch(facecolor="#FF8C00", alpha=0.4, label="Medium hazard (0.5–1.5 m)"),
    Patch(facecolor="#CC0000", alpha=0.4, label="High hazard (>1.5 m)"),
    # Roads
    Line2D([0], [0], color="#CCCCCC", linewidth=1, label="Roads (not flooded)"),
    Line2D([0], [0], color="#CC0000", linewidth=1.5, label="Roads (in flood zone)"),
    # Separator
    Line2D([0], [0], color="none", label=""),
]

# POI legend entries
for category in poi_queries:
    style = poi_styles[category]
    count = len(all_pois.get(category, []))
    summary = poi_flood_summary.get(category, {"flooded": 0})
    legend_elements.append(
        Line2D(
            [0], [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markersize=8,
            label=f"{category} ({count})",
        )
    )

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=9,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
)

ax.set_title(
    f"5-Year Flood Hazard, Roads & Critical Facilities\n{DISPLAY_NAME}, Batangas",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Stats box
total_roads_km = roads.geometry.to_crs(epsg=32651).length.sum() / 1000
flooded_roads_km = roads_in_flood.geometry.to_crs(epsg=32651).length.sum() / 1000
roads_pct = (flooded_roads_km / total_roads_km) * 100

stats_lines = [
    f"Roads: {flooded_roads_km:.1f} / {total_roads_km:.1f} km flooded ({roads_pct:.1f}%)",
    "",
]
for category in poi_queries:
    s = poi_flood_summary.get(category, {"total": 0, "flooded": 0, "pct": 0})
    stats_lines.append(f"{category}: {s['flooded']}/{s['total']} in flood zone ({s['pct']:.0f}%)")

ax.text(
    0.98, 0.98,
    "\n".join(stats_lines),
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    family="monospace",
)

plt.tight_layout()
plt.savefig("output/batangas_poi_flood_5yr.png", dpi=200, bbox_inches="tight")
plt.show()

# Print summary
print(f"\n{'='*55}")
print(f" FLOOD EXPOSURE SUMMARY — {DISPLAY_NAME}")
print(f"{'='*55}")
print(f"\n Roads")
print(f"   Total:   {total_roads_km:.1f} km")
print(f"   Flooded: {flooded_roads_km:.1f} km ({roads_pct:.1f}%)")
print()
for category in poi_queries:
    s = poi_flood_summary.get(category, {"total": 0, "flooded": 0, "pct": 0})
    print(f" {category}")
    print(f"   Total:   {s['total']}")
    print(f"   Flooded: {s['flooded']} ({s['pct']:.0f}%)")
print()
print("Map saved to output/batangas_poi_flood_5yr.png")
