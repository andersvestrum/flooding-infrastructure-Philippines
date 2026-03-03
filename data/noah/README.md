# NOAH Flood Hazard Maps

Flood hazard shapefiles from [Project NOAH](https://noah.up.edu.ph/) (Nationwide Operational Assessment of Hazards), developed by the University of the Philippines.

## What's in here

```
noah/
├── 5yr/
│   └── Batangas/
│       └── Batangas_Flood_5yr.*      ← 5-year return period flood
└── 100yr/
    └── Batangas/
        └── Batangas_Flood_100yr.*    ← 100-year return period flood
```

Each flood scenario is stored as an ESRI Shapefile (`.shp`, `.shx`, `.dbf`, `.prj`).

## Hazard levels

The `Var` column in each shapefile indicates the hazard classification:

| Var | Hazard Level | Flood Depth       | What it means                          |
|-----|-------------|-------------------|----------------------------------------|
| 1   | Low         | 0 – 0.5 m         | Knee-level or below                    |
| 2   | Medium      | 0.5 – 1.5 m       | Knee to neck height                    |
| 3   | High        | > 1.5 m            | Above neck height, dangerous           |

Hazard levels account for both flood **depth** and **velocity** — shallow but fast-flowing water can be classified as higher hazard than depth alone would suggest.

## Return periods

- **5-year flood** — a flood with a 20% chance of occurring in any given year. Represents more frequent, moderate flooding events.
- **100-year flood** — a flood with a 1% chance of occurring in any given year. Represents rare but severe flooding events.

## Coordinate system

- **Datum:** WGS 84
- **EPSG:** 4326 (geographic lat/lon)

## Source & license

- **Source:** [Project NOAH – UP DOST](https://noah.up.edu.ph/)
- **Coverage:** Entire Philippines; LiDAR data covers 18 major river basins
- **License:** Open Data Commons Open Database License (ODbL) — if you alter or build upon this data, you must distribute the result under the same license.
- **Update frequency:** Annual, or as needed based on field validation and new data inputs

## How to load

```python
import geopandas as gpd

flood_5yr = gpd.read_file("data/noah/5yr/Batangas/Batangas_Flood_5yr.shp")
flood_100yr = gpd.read_file("data/noah/100yr/Batangas/Batangas_Flood_100yr.shp")
```
