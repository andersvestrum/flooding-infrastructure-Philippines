import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os

# Paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load 5-year flood
flood_5yr = gpd.read_file(os.path.join(ROOT, "data/noah/5yr/Cagayan/Cagayan_Flood_5year.shp"))

# Load 100-year flood
flood_100yr = gpd.read_file(os.path.join(ROOT, "data/noah/100yr/Cagayan/Cagayan_Flood_100year.shp"))

# Load province/municipality boundaries from GADM (level 2 = municipalities)
ph_admin = gpd.read_file(
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json"
)
cagayan = ph_admin[ph_admin["NAME_1"] == "Cagayan"]

# Hazard color map (yellow=low, orange=medium, red=high)
hazard_cmap = ListedColormap(["#FFD700", "#FF8C00", "#CC0000"])
hazard_labels = {
    1: "Low (0–0.5 m)",
    2: "Medium (0.5–1.5 m)",
    3: "High (>1.5 m)",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Plot municipality borders as background on both axes ---
for ax in axes:
    cagayan.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

flood_5yr.plot(
    column="Var",
    cmap=hazard_cmap,
    vmin=1,
    vmax=3,
    ax=axes[0],
    edgecolor="none",
)
axes[0].set_title("5-Year Return Period Flood Hazard", fontsize=14)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

flood_100yr.plot(
    column="Var",
    cmap=hazard_cmap,
    vmin=1,
    vmax=3,
    ax=axes[1],
    edgecolor="none",
)
axes[1].set_title("100-Year Return Period Flood Hazard", fontsize=14)
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

legend_patches = [
    Patch(facecolor="#FFD700", label=hazard_labels[1]),
    Patch(facecolor="#FF8C00", label=hazard_labels[2]),
    Patch(facecolor="#CC0000", label=hazard_labels[3]),
]
fig.legend(
    handles=legend_patches,
    title="Flood Hazard Level",
    loc="lower center",
    ncol=3,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
)

fig.suptitle("NOAH Flood Hazard Map — Cagayan Province", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "output/cagayan_flood_hazard.png"), dpi=200, bbox_inches="tight")
plt.show()