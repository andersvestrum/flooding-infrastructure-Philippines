# Flooding & Infrastructure — Philippines

Analyzing flood hazard exposure of road networks and critical infrastructure in the Philippines across 10 selected provinces.

## Data

The flood hazard shapefiles are too large for GitHub (>100 MB). They are hosted on Google Drive:

- **[5yr flood hazard data](https://drive.google.com/drive/folders/17ecJuf2vnkrpCzNVLes0fI08XFsR1x8N?usp=drive_link)**
- **[100yr flood hazard data](https://drive.google.com/drive/folders/10pCWTfU-gVuAbdx4gdUGaDcNrSzMz0Mm?usp=drive_link)**

After downloading, your folder structure should look like:

```
data/
└── noah/
    ├── README.md
    ├── 5yr/
    │   ├── Cagayan/
    │   ├── Pangasinan/
    │   ├── Pampanga/
    │   ├── Metropolitan Manila/
    │   ├── Camarines Norte/
    │   ├── Camarines Sur/
    │   ├── Isabela/
    │   ├── Maguindanao/
    │   ├── Misamis Oriental/
    │   └── ...
    └── 100yr/
        ├── Cagayan/
        ├── Pangasinan/
        ├── Agusan del Norte/
        ├── Metropolitan Manila/
        └── ...
```

See [`data/noah/README.md`](data/noah/README.md) for details on the NOAH flood hazard data format and hazard levels.

## Analysis Scripts

| Script | Scope | What it shows |
|--------|-------|---------------|
| `analysis/cagayan_flood_5yr_vs_100yr.py` | Province | Side-by-side 5yr vs 100yr flood hazard maps |
| `analysis/cagayan_roads_flood_5yr.py` | Municipality | Road network flood exposure |
| `analysis/cagayan_poi_flood_5yr.py` | Municipality | Roads + critical facilities (hospitals, schools, markets, pharmacies, barangay halls) flood exposure |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_data/download_noah.py
```

The last command downloads the flood hazard shapefiles for the 10 configured provinces from Google Drive automatically.

## Data Sources

- **Flood hazard maps:** [Project NOAH](https://noah.up.edu.ph/) (UP DOST) — ODbL license
- **Road networks & POIs:** [OpenStreetMap](https://www.openstreetmap.org/) via [OSMnx](https://osmnx.readthedocs.io/) — ODbL license
- **Administrative boundaries:** [GADM](https://gadm.org/) v4.1