# Urban Satellite-Bias Analysis Methods

## Purpose

The goal of this analysis was to test whether satellite-derived flood extents under-detect flooding in dense urban areas, and whether that makes a modelled hazard layer such as NOAH useful for retaining urban flood risk that observational satellite products may miss.

## Datasets

- **PhilSA flood extents**: all vector flood polygons available locally in `data/philsa_satellite_flood` for the matched period **2022-08-05 to 2026-02-09**.
- **Groundsource flood polygons**: `data/google_gemini_flood/groundsource_2026.parquet`, filtered to the same matched period.
- **NOAH 5-year hazard**: local 5-year flood hazard shapefiles in `data/noah/5yr`.
- **Urban class**: GHSL SMOD 2020 classes from JRC/Copernicus, aggregated as **rural (`11,12,13`) / peri-urban (`21`) / urban (`22,23,30`)**, with **water (`10`) excluded**.

## NOAH-linked city analysis used for the paper figure

The stand-alone panel-D figure is based on the five city case studies already used elsewhere in the project:

- Tuguegarao
- Dagupan
- Manila
- Cagayan de Oro
- Cotabato

For each city:

1. A **250 m grid** was built inside the city buffer in **EPSG:32651**.
2. The **maximum NOAH 5-year hazard class** was sampled per grid cell.
3. All **PhilSA** flood polygons intersecting the city buffer were rasterized to the same grid, treating each PhilSA source file as one binary flood observation.
4. All **Groundsource** polygons in the same date window were rasterized to the grid and converted to a smoothed support surface using a **Gaussian filter with sigma = 750 m**.
5. PhilSA counts and Groundsource scores were each classified into **low / medium / high** using tertiles of cells with positive support.
6. Grid cells with **NOAH class >= medium** were treated as **NOAH-active**.
7. Each NOAH-active cell was assigned one of four diagnostic categories:
   - **Both support NOAH**
   - **Groundsource supports, PhilSA misses**
   - **Weak in both**
   - **PhilSA only**
8. GHSL urban classes were sampled at grid-cell centers and used to aggregate the diagnostic categories by **rural / peri-urban / urban**, with GHSL **water** cells excluded from the urbanisation grouping.

The resulting panel-D figure therefore answers:

> Among NOAH-active cells, how often does Groundsource support the NOAH signal while PhilSA misses it, and does that happen more often in urban cells than in rural cells?

## National extension

A full **nationwide NOAH-linked** analysis was **not possible** from the local repository alone, because the repository currently contains NOAH 5-year hazard data for only **nine provinces**, not the full Philippines.

To extend the urban-bias test nationally in a way that remained methodologically defensible, I ran a second, **Groundsource-vs-PhilSA** analysis across the Philippines:

1. Groundsource polygons were filtered to the Philippines bounding box (**116-127E, 4-22N**) and the same matched period.
2. PhilSA polygons were filtered to the same period.
3. Each Groundsource polygon was assigned an urban class using the **representative point** of the polygon and GHSL SMOD.
4. A Groundsource polygon was marked as **seen by PhilSA** if it intersected any PhilSA flood polygon in the matched period.
5. Overlap rates were summarized by urban class using:
   - **polygon overlap share**
   - **area-weighted overlap share**

This national extension is a **supporting analysis**, not a NOAH validation. Its purpose is to test whether PhilSA’s overlap with an independent observational flood source becomes weaker in urban settings at the national scale.

## Interpretation logic

The key interpretation rule is:

- If **PhilSA is weak** in urban areas **while Groundsource still supports flooding there**, that points to an **urban satellite-detection problem**.
- If **both** PhilSA and Groundsource are weak in the same places, that would point more toward **drainage, timing, or NOAH overprediction**.

## Why this is reasonable

This workflow follows prior research showing that flood mapping from satellite imagery can be difficult in dense urban terrain because of SAR-specific issues such as **layover, shadow, and double-bounce**, and that combining satellite observations with hazard information can improve urban flood interpretation.

## Main caveats

- Groundsource is **not perfect ground truth**; it is an auxiliary observational reference.
- The national extension is **not a nationwide NOAH validation**, because local NOAH coverage is incomplete.
- GHSL urban class is sampled at a point and therefore simplifies within-polygon land-use heterogeneity.
- PhilSA and Groundsource have different observation processes, so the comparison should be interpreted as **relative detection performance**, not exact event matching.
