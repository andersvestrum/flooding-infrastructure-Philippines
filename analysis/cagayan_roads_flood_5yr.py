import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import osmnx as ox
import os

# Paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
MUNICIPALITY = "TuguegaraoCity"  # Change this to any municipality in Cagayan (see GADM names)
DISPLAY_NAME = "Tuguegarao City"  # Pretty name for titles

# Load data
print("Loading flood hazard data...")
flood_5yr = gpd.read_file(os.path.join(ROOT, "data/noah/5yr/Cagayan/Cagayan_Flood_5year.shp"))

print("Loading municipality boundaries...")
ph_admin = gpd.read_file(
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json"
)
muni = ph_admin[
    (ph_admin["NAME_1"] == "Cagayan") & (ph_admin["NAME_2"] == MUNICIPALITY)
]

if muni.empty:
    # Show available municipalities to help pick one
    available = ph_admin[ph_admin["NAME_1"] == "Cagayan"]["NAME_2"].sort_values().tolist()
    print(f"Municipality '{MUNICIPALITY}' not found. Available options:")
    for m in available:
        print(f"  - {m}")
    raise SystemExit(1)

print(f"Using municipality: {MUNICIPALITY}")

# Clip flood data to municipality
print("Clipping flood data to municipality bounds...")
flood_clipped = gpd.clip(flood_5yr, muni)

# Download road network from OSM
print("Downloading road network from OpenStreetMap...")
muni_polygon = muni.geometry.union_all()
G = ox.graph_from_polygon(muni_polygon, network_type="drive")
roads = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Identify flooded roads
print("Identifying flooded road segments...")
roads_in_flood = gpd.sjoin(roads, flood_clipped, how="inner", predicate="intersects")

# Get roads NOT in flood zones
flooded_idx = roads_in_flood.index.unique()
roads_safe = roads[~roads.index.isin(flooded_idx)]

# Classify flooded roads by max hazard level
roads_max_hazard = roads_in_flood.groupby(roads_in_flood.index)["Var"].max()
roads_in_flood = roads_in_flood[~roads_in_flood.index.duplicated(keep="first")].copy()
roads_in_flood["max_hazard"] = roads_max_hazard

# Plot — side-by-side: roads only vs. roads + flood
print("Plotting...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))

# ── Left panel: Road network only ──
muni.boundary.plot(ax=ax1, edgecolor="black", linewidth=1.5, zorder=1)
roads.plot(ax=ax1, color="#444444", linewidth=0.5, zorder=2)

ax1.set_title("Road Network", fontsize=14, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")

road_legend = [
    Line2D([0], [0], color="#444444", linewidth=1, label="Roads"),
]
ax1.legend(handles=road_legend, loc="lower left", fontsize=10, frameon=True)

# ── Right panel: Roads + flood hazard overlay ──
muni.boundary.plot(ax=ax2, edgecolor="black", linewidth=1.5, zorder=1)

# Flood hazard zones
hazard_cmap = ListedColormap(["#FFD700", "#FF8C00", "#CC0000"])
flood_clipped.plot(
    column="Var",
    cmap=hazard_cmap,
    vmin=1,
    vmax=3,
    ax=ax2,
    alpha=0.4,
    edgecolor="none",
    zorder=2,
)

# Safe roads (gray)
roads_safe.plot(ax=ax2, color="#AAAAAA", linewidth=0.5, zorder=3)

# Flooded roads colored by hazard level
hazard_colors = {1: "#FFD700", 2: "#FF8C00", 3: "#CC0000"}
for hazard_level, color in hazard_colors.items():
    subset = roads_in_flood[roads_in_flood["max_hazard"] == hazard_level]
    if not subset.empty:
        subset.plot(ax=ax2, color=color, linewidth=1.2, zorder=4)

ax2.set_title("Road Network + 5-Year Flood Hazard", fontsize=14, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")

# Legend for right panel
legend_elements = [
    Patch(facecolor="#FFD700", alpha=0.4, label="Low flood hazard (0–0.5 m)"),
    Patch(facecolor="#FF8C00", alpha=0.4, label="Medium flood hazard (0.5–1.5 m)"),
    Patch(facecolor="#CC0000", alpha=0.4, label="High flood hazard (>1.5 m)"),
    Line2D([0], [0], color="#AAAAAA", linewidth=1, label="Roads (not flooded)"),
    Line2D([0], [0], color="#CC0000", linewidth=1.5, label="Roads (in flood zone)"),
]
ax2.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=True)

# Stats
total_roads_km = roads.geometry.to_crs(epsg=32651).length.sum() / 1000
flooded_roads_km = roads_in_flood.geometry.to_crs(epsg=32651).length.sum() / 1000
pct = (flooded_roads_km / total_roads_km) * 100

stats_text = (
    f"Total roads: {total_roads_km:.1f} km\n"
    f"Flooded roads: {flooded_roads_km:.1f} km ({pct:.1f}%)"
)
ax2.text(
    0.98, 0.98, stats_text,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Match axis limits so both panels show the same extent
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())

fig.suptitle(
    f"Road Network & Flood Exposure — {DISPLAY_NAME}, Cagayan",
    fontsize=16, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "output/cagayan_roads_flood_5yr.png"), dpi=200, bbox_inches="tight")
plt.show()

print(f"\n--- Summary for {DISPLAY_NAME} ---")
print(f"Total road network: {total_roads_km:.1f} km")
print(f"Roads in flood zones: {flooded_roads_km:.1f} km ({pct:.1f}%)")
print("Map saved to output/cagayan_roads_flood_5year.png")
