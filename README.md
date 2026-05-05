# Flooding & Infrastructure — Philippines

This repository has been trimmed to the materials that support the final paper
and supporting information for:

`Infrastructure form shapes equitable access to essential services under flooding in Philippine cities`

The cleanup keeps:

- `report/` manuscript and SI sources
- `data/` inputs
- `output/` derived outputs and figures
- `setup_data/` download / preprocessing loaders
- the active `analysis/` scripts that generate the cited flood-validation
  figures and table

Exploratory side analyses, legacy validation branches, and large local cache
directories were removed.

## Active analysis scripts

The remaining analysis scripts are the ones still tied to the final paper / SI
validation workflow:

- `analysis/noah_ai4g_comparison.py`
  Builds the NOAH vs AI4G comparison outputs and the per-cell AI4G parquet used
  downstream.
- `analysis/noah_philsa_allfiles_comparison.py`
  Builds the NOAH vs PhilSA comparison outputs and the per-cell PhilSA parquet
  used downstream.
- `analysis/noah_event_validation.py`
  Builds event-level validation summaries and event recall metrics.
- `analysis/noah_event_four_city_maps.py`
  Builds the four-city NOAH vs satellite support figure:
  `output/noah_validation/events/noah_event_05_fourcity_maps.png`
- `analysis/noah_groundsource_grid_validation.py`
  Builds the Groundsource diagnostics that support the SI table discussion.
- `analysis/urban_noah_correlation_matrix.py`
  Builds the city-level correlation matrix:
  `output/urban_bias/urban_noah_correlation_citylevel_clear.png`

## Repository structure

```text
.
├── analysis/        Active paper/SI validation scripts only
├── data/            Input datasets and local caches needed by loaders
├── output/          All generated outputs retained, including non-paper runs
├── report/          Main paper and SI LaTeX sources plus figure assets
├── setup_data/      Data download / setup scripts
├── download_data.py
├── requirements.txt
└── README.md
```

## Main report assets

The manuscript sources live in:

- `report/INFO_288__Philippines_flooding_and_roads(1)/Research_report.tex`
- `report/INFO_288__Philippines_flooding_and_roads(1)/PNAS-SI.tex`

Key validation figures used by the paper / SI are loaded from:

- `output/noah_validation/events/noah_event_05_fourcity_maps.png`
- `output/urban_bias/urban_noah_correlation_citylevel_clear.png`

Additional paper figure assets that are already rendered and kept in the repo
live under:

- `report/INFO_288__Philippines_flooding_and_roads(1)/figures/`

## Data notes

- NOAH 5-year hazard data are stored under `data/noah/5yr/`
- Butuan now uses the official NOAH 5-year layer placed under:
  `data/noah/5yr/Agusan del Norte/`
- PhilSA inputs live under `data/philsa_satellite_flood/`
- AI4G inputs / cache live under `data/ai4g/`
- GHSL inputs live under `data/ghsl/`
- Groundsource inputs live under `data/google_gemini_flood/`

## Rebuilding the SI validation figures

```bash
python3 analysis/noah_philsa_allfiles_comparison.py
python3 analysis/noah_ai4g_comparison.py
python3 analysis/noah_event_validation.py
python3 analysis/noah_event_four_city_maps.py
python3 analysis/noah_groundsource_grid_validation.py
python3 analysis/urban_noah_correlation_matrix.py
```

Those scripts regenerate the main retained validation outputs without bringing
back the older exploratory branches that were removed during cleanup.
