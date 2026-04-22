# Flooding & Infrastructure — Philippines

Validation of the **NOAH flood hazard dataset** as a ground-truth source for
flood risk analysis in the Philippines, by cross-comparing it against three
independent observational datasets across five study cities.

## Research goal

NOAH provides static, model-based flood hazard maps (Low / Medium / High,
5-year return period) derived from LiDAR and hydrodynamic modelling. This
project asks: **do observed floods corroborate NOAH's hazard classifications?**

To answer that, we compare NOAH against:

| Source | Type | Period | Resolution |
|--------|------|--------|-----------|
| **AI4G** (ai-for-good-lab/ai4g-flood-dataset) | Sentinel-1 SAR flood detections | Oct 2014 – Sep 2024 | 20 m |
| **PhilSA** (HDX) | Multi-sensor SAR flood extents (Sentinel-1, SAOCOM, TerraSAR-X, …) | 2022 – 2024 | 250 m |
| **Google Ground-source** | Crowdsourced flood event reports | 2011 – 2025 | Point → 250 m grid |

Each comparison rasterises observed flood frequency onto the same **250 m UTM
grid** used by NOAH, then computes agreement metrics (Spearman ρ, exact class
match, consensus quadrants).

## Study cities

| City | Province | Region | Radius |
|------|----------|--------|--------|
| Tuguegarao | Cagayan | Cagayan Valley | 10 km |
| Ilagan | Isabela | Cagayan Valley | 10 km |
| Dagupan | Pangasinan | Ilocos | 12 km |
| San Fernando | Pampanga | Central Luzon | 12 km |
| Manila | Metropolitan Manila | NCR | 20 km |
| Naga | Camarines Sur | Bicol | 10 km |
| Daet | Camarines Norte | Bicol | 8 km |
| Cagayan de Oro | Misamis Oriental | Northern Mindanao | 12 km |
| Butuan | Agusan del Norte* | Caraga | 10 km |
| Cotabato | Maguindanao | BARMM | 10 km |

*NOAH shapefile not available for Agusan del Norte — Butuan is included in Aqueduct and AI4G comparisons only.

## Repository structure

```
.
├── analysis/                   ← all runnable scripts (see below)
├── data/
│   ├── noah/
│   │   ├── README.md           ← NOAH data format & hazard levels
│   │   ├── 5yr/                ← 5-year return-period shapefiles
│   │   └── 100yr/              ← 100-year return-period shapefiles
│   ├── ai4g/
│   │   └── philippines_floods.parquet   ← cached after first run
│   ├── philsa_satellite_flood/ ← PhilSA zipped shapefiles from HDX
│   └── google_gemini_flood/    ← ground-source parquet
│   ├── aqueduct/               ← WRI Aqueduct GeoTIFF (cached on first run)
├── output/
│   ├── noah_validation/
│   │   ├── ai4g/               ← NOAH vs AI4G outputs
│   │   ├── aqueduct/           ← NOAH vs WRI Aqueduct outputs + per-cell parquet
│   │   ├── philsa/             ← NOAH vs PhilSA outputs
│   │   ├── groundsource/       ← NOAH vs ground-source outputs
│   │   ├── noah_robustness_A_sources.png   ← ρ heatmap across data sources
│   │   └── noah_robustness_B_scale.png     ← ρ vs analysis radius
│   ├── urban_bias/             ← satellite under-detection in urban areas
│   └── paper_figures/          ← publication-ready figures
├── requirements.txt
└── README.md
```

## Analysis scripts

### NOAH vs AI4G (Sentinel-1 SAR) — primary validation

| Script | Description |
|--------|-------------|
| `noah_ai4g_comparison.py` | Full 10-year comparison: NOAH hazard class vs AI4G flood frequency across 5 cities. Outputs consensus risk maps and diagnostics. |
| `noah_ai4g_5yr_windows.py` | Same comparison windowed into 5-year periods — **non-overlapping** (W1: 2014–19, W2: 2019–24) and **overlapping** (6 windows, 1-year step). Averages frequency across windows before comparing to NOAH. |

### NOAH vs PhilSA (multi-sensor SAR) — secondary validation

| Script | Description |
|--------|-------------|
| `noah_philsa_consensus.py` | Core NOAH vs PhilSA comparison with consensus risk quadrants (confirmed / modelled-only / empirical-gap / low). |
| `noah_philsa_allfiles_comparison.py` | Same, using all available PhilSA HDX files. |
| `noah_philsa_gfd_consensus.py` | Extended version adding Global Flood Database (GFD) as a third source. |
| `manila_philsa_noah_diagnosis.py` | Manila-specific deep-dive: PhilSA vs NOAH disaggregated by urban density. |
| `philsa_groundsource_comparison.py` | Cross-validates PhilSA against ground-source observations (supports the NOAH chain). |

### NOAH vs ground-source observations

| Script | Description |
|--------|-------------|
| `noah_groundsource_grid_validation.py` | Grid-based validation: NOAH hazard class vs ground-source flood frequency on a common 250 m grid. |
| `noah_groundsource_all_windows.py` | Aggregates ground-source across all valid 5-year windows (2011–2025), masks permanent water. |
| `noah_groundsource_5yr_intervals.py` | Comparison using matched 5-year intervals. |
| `noah_groundsource_standardized_hazard.py` | Tests whether NOAH's hazard classes have consistent meaning across provinces. |
| `noah_groundsource_smoothed_empirical_hazard.py` | Smoothed empirical hazard from ground-source matched to NOAH's 5-year window. |

### NOAH vs WRI Aqueduct (model-based independent reference)

| Script | Description |
|--------|-------------|
| `noah_aqueduct_comparison.py` | NOAH vs WRI Aqueduct Floods v2 (WATCH historical, RP-5 riverine depth). Downloads ~83 MB GeoTIFF once, classifies depth into Low/Medium/High using NOAH's own breakpoints (0.5 m, 1.5 m), computes Spearman ρ, exact match, consensus quadrants across all 10 cities. Saves per-cell parquet for robustness analysis. |

### Robustness analysis

| Script | Description |
|--------|-------------|
| `noah_source_robustness.py` | Two separate robustness plots: **(A)** heatmap + grouped bars of Spearman ρ across all 10 cities × data sources (AI4G SAR, Aqueduct); **(B)** ρ vs analysis radius (40–120 % of nominal) per city, using the Aqueduct per-cell parquet. Run after both comparison scripts. |

### Supporting analyses

| Script | Description |
|--------|-------------|
| `philippines_urban_satellite_bias.py` | Quantifies satellite under-detection of urban flooding (PhilSA, ground-source, NOAH comparison across cities). |
| `philippines_groundsource_philsa_national_urban_overlap.py` | National-level urban overlap analysis supporting the bias study. |
| `paper_figure_validation.py` | Generates publication-quality NOAH validation figures. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

**NOAH shapefiles** are too large for Git. Download from Google Drive:

- [5-year return period](https://drive.google.com/drive/folders/17ecJuf2vnkrpCzNVLes0fI08XFsR1x8N?usp=drive_link)
- [100-year return period](https://drive.google.com/drive/folders/10pCWTfU-gVuAbdx4gdUGaDcNrSzMz0Mm?usp=drive_link)

Place under `data/noah/5yr/` and `data/noah/100yr/` respectively (one folder per province).

**AI4G data** is downloaded automatically from HuggingFace on first run and
cached at `data/ai4g/philippines_floods.parquet`:

```python
# handled inside noah_ai4g_comparison.py via huggingface_hub
```

**PhilSA** zipped shapefiles go in `data/philsa_satellite_flood/`.  
**Ground-source** parquet goes at `data/google_gemini_flood/groundsource_2026.parquet`.

### Running

```bash
# Primary validation (AI4G Sentinel-1, 10 cities)
python3 analysis/noah_ai4g_comparison.py
python3 analysis/noah_ai4g_5yr_windows.py

# WRI Aqueduct validation (10 cities, downloads ~83 MB TIF once)
python3 analysis/noah_aqueduct_comparison.py

# Robustness plots (requires both comparison CSVs above)
python3 analysis/noah_source_robustness.py

# PhilSA validation
python3 analysis/noah_philsa_consensus.py

# Ground-source validation
python3 analysis/noah_groundsource_grid_validation.py

# Urban bias study
python3 analysis/philippines_urban_satellite_bias.py
```

All outputs are written directly into the appropriate `output/` subfolder.

## Key findings (summary)

| City | AI4G ρ | Aqueduct ρ | AI4G conf. | AI4G mod. | AI4G emp. |
|------|--------|-----------|-----------|----------|----------|
| Ilagan | **0.60** | 0.26 | 13% | 30% | 1% |
| Naga | **0.48** | 0.24 | 8% | 16% | 11% |
| Tuguegarao | **0.32** | 0.03 | 5% | 13% | 12% |
| Dagupan | 0.26 | 0.17 | 2% | 6% | 11% |
| Daet | 0.20 | 0.22 | 2% | 15% | 9% |
| San Fernando | 0.18 | −0.00 | 6% | 16% | 10% |
| Cagayan de Oro | 0.07 | 0.19 | 0% | 4% | 1% |
| Cotabato | 0.03 | −0.01 | 0% | 3% | 6% |
| Manila | ~0 | 0.07 | 0% | 6% | 1% |
| Butuan | — (no NOAH) | — | — | — | — |

- **Ilagan** and **Naga** show the strongest AI4G–NOAH agreement (ρ = 0.60, 0.48), both in regions with high observed flood frequency.
- **Tuguegarao** and **Dagupan** also show good agreement; Dagupan has the highest empirical gap (11%) suggesting the 5-yr hazard map may underestimate risk.
- **Manila** and **Cotabato** show near-zero correlation despite substantial NOAH hazard coverage — consistent with SAR under-detection in dense urban areas and with Aqueduct (model-based) also showing low correlation, suggesting these cities have atypical flood regimes relative to the model assumptions.
- The robustness plots (`noah_source_robustness.py`) confirm cross-source consistency for Cagayan Valley and Bicol cities, while Manila and Cotabato are consistently low-agreement across both sources and all analysis radii.

## Data sources & licences

| Dataset | Licence |
|---------|---------|
| [Project NOAH](https://noah.up.edu.ph/) (UP DOST) | ODbL |
| [AI4G flood dataset](https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset) | See dataset card |
| [WRI Aqueduct Floods v2](https://www.wri.org/data/aqueduct-floods-hazard-maps) | CC BY 4.0 |
| [PhilSA / HDX](https://data.humdata.org/) | CC BY-IGO |
| [OpenStreetMap](https://www.openstreetmap.org/) (water masks) | ODbL |
