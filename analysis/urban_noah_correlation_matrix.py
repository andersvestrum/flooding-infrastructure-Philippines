"""
urban_noah_correlation_matrix.py
================================
Build correlation matrices for the urban satellite-detection question.

The goal is to relate urbanisation, NOAH hazard coverage, and observed flood
support from satellite products / Groundsource. The output is intentionally
exploratory: it is useful for hypothesis-building, not for causal proof.

Outputs
-------
  output/urban_bias/urban_noah_city_metrics.csv
  output/urban_bias/urban_noah_correlation_citylevel.csv
  output/urban_bias/urban_noah_correlation_citylevel.png
  output/urban_bias/urban_noah_correlation_citylevel.pdf
  output/urban_bias/urban_noah_correlation_citylevel_clear.png
  output/urban_bias/urban_noah_correlation_citylevel_clear.pdf
  output/urban_bias/urban_noah_correlation_groundsource.csv
  output/urban_bias/urban_noah_correlation_groundsource.png
  output/urban_bias/urban_noah_correlation_groundsource.pdf
  output/urban_bias/urban_noah_correlation_groundsource_clear.png
  output/urban_bias/urban_noah_correlation_groundsource_clear.pdf
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "urban_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHILSA_CELLS = ROOT / "output" / "noah_validation" / "philsa" / "noah_philsa_allfiles_cells.parquet"
AI4G_CELLS = ROOT / "output" / "noah_validation" / "ai4g" / "noah_ai4g_cells.parquet"
EVENT_SUMMARY = ROOT / "output" / "noah_validation" / "events" / "noah_event_summary.csv"
URBAN_BIAS_SUMMARY = OUT_DIR / "philippines_urban_satellite_bias_summary.csv"
GHSL_DIR = ROOT / "data" / "ghsl"


GHSL_CLASS_MAP = {
    10: "water",
    11: "rural",
    12: "rural",
    13: "rural",
    21: "peri-urban",
    22: "urban",
    23: "urban",
    30: "urban",
}

CITY_LABELS = {
    "urban_share": "Urban share",
    "noah_any_share": "NOAH hazard share",
    "noah_any_urban_share": "Urban share of NOAH hazard",
    "philsa_any_in_noah_any_share": "PhilSA hit in NOAH hazard",
    "ai4g_any_in_noah_any_share": "AI4G hit in NOAH hazard",
    "satellite_any_in_noah_any_share": "Any satellite hit in NOAH hazard",
    "event_weighted_noah_any_recall": "Observed flood recall",
    "philsa_mean_freq_noah_any": "PhilSA freq in NOAH hazard",
}

GS_LABELS = {
    "urban_share": "Urban share",
    "noah_active_share": "NOAH active share",
    "noah_active_urban_share": "Urban share of NOAH active",
    "active_cells_sat_gap_share": "GS yes / PhilSA no",
    "active_cells_with_hot_philsa_share": "PhilSA hotspot",
    "active_cells_with_hot_groundsource_share": "GS hotspot",
    "philsa_spearman_noah_vs_count": "PhilSA ~ NOAH rho",
    "groundsource_spearman_noah_vs_score": "GS ~ NOAH rho",
}

CLEAR_CITY_DRIVERS = {
    "urban_share": "Urban share\nof city buffer",
    "noah_any_share": "NOAH share\nof city buffer",
    "noah_any_urban_share": "Urban share of\nNOAH cells",
}

CLEAR_CITY_OUTCOMES = {
    "philsa_any_in_noah_any_share": "PhilSA recall\nin NOAH",
    "ai4g_any_in_noah_any_share": "AI4G recall\nin NOAH",
    "satellite_any_in_noah_any_share": "Any satellite recall\nin NOAH",
    "event_weighted_noah_any_recall": "Event recall\nin NOAH",
}

CLEAR_GS_DRIVERS = {
    "urban_share": "Urban share\nof city buffer",
    "noah_active_share": "NOAH Med+High\nshare of city buffer",
    "noah_active_urban_share": "Urban share of\nNOAH Med+High cells",
}

CLEAR_GS_OUTCOMES = {
    "active_cells_sat_gap_share": "Groundsource yes\nPhilSA no",
    "active_cells_with_hot_philsa_share": "PhilSA hotspot\ninside NOAH",
    "active_cells_with_hot_groundsource_share": "Groundsource hotspot\ninside NOAH",
    "philsa_spearman_noah_vs_count": "PhilSA-NOAH\nrank match",
    "groundsource_spearman_noah_vs_score": "Groundsource-NOAH\nrank match",
}


def _load_cells():
    philsa = pd.read_parquet(PHILSA_CELLS)
    ai4g = pd.read_parquet(AI4G_CELLS)
    key_cols = ["city", "lon", "lat", "noah_cls", "is_water"]
    cells = philsa.merge(ai4g[key_cols + ["ai4g_freq"]], on=key_cols, how="left")
    cells["ai4g_freq"] = cells["ai4g_freq"].fillna(0.0)
    return cells


def _sample_ghsl(points_df):
    points = gpd.GeoDataFrame(
        points_df[["city", "lon", "lat"]].copy(),
        geometry=[Point(xy) for xy in zip(points_df["lon"], points_df["lat"])],
        crs="EPSG:4326",
    ).to_crs("ESRI:54009")

    values = np.full(len(points), -9999, dtype=int)
    coords = np.array([(geom.x, geom.y) for geom in points.geometry])

    # Use the explicit 1 km GHSL tiles. Avoid duplicate/derived national/manila
    # rasters where possible so each point is sampled once from the official tile.
    tile_paths = sorted(GHSL_DIR.glob("ghsl_smod_r*_c*.tif"))
    for path in tile_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            mask = (
                (values == -9999)
                & (coords[:, 0] >= b.left)
                & (coords[:, 0] < b.right)
                & (coords[:, 1] >= b.bottom)
                & (coords[:, 1] < b.top)
            )
            if not mask.any():
                continue
            sampled = list(src.sample(coords[mask]))
            values[np.where(mask)[0]] = [int(v[0]) for v in sampled]

    return np.array([GHSL_CLASS_MAP.get(int(v), "unknown") for v in values], dtype=object)


def _safe_share(num, den):
    return float(num) / float(den) if den else np.nan


def _city_metrics(cells):
    rows = []
    for city, sub in cells.groupby("city"):
        valid = (~sub["is_water"]) & (~sub["urban_cls"].isin(["water", "unknown"]))
        s = sub.loc[valid].copy()
        if s.empty:
            continue

        urban = s["urban_cls"] == "urban"
        peri = s["urban_cls"] == "peri-urban"
        noah_any = s["noah_cls"] > 0
        noah_active = s["noah_cls"] >= 2
        noah_high = s["noah_cls"] == 3
        philsa_any = s["philsa_freq"] > 0
        ai4g_any = s["ai4g_freq"] > 0
        satellite_any = philsa_any | ai4g_any

        rows.append(
            {
                "city": city,
                "n_valid_cells": int(len(s)),
                "urban_share": float(urban.mean()),
                "periurban_share": float(peri.mean()),
                "noah_any_share": float(noah_any.mean()),
                "noah_active_share": float(noah_active.mean()),
                "noah_high_share": float(noah_high.mean()),
                "urban_noah_any_share": float((urban & noah_any).mean()),
                "noah_any_urban_share": _safe_share((urban & noah_any).sum(), noah_any.sum()),
                "urban_noah_active_share": float((urban & noah_active).mean()),
                "noah_active_urban_share": _safe_share((urban & noah_active).sum(), noah_active.sum()),
                "philsa_any_share": float(philsa_any.mean()),
                "philsa_any_in_noah_any_share": _safe_share((philsa_any & noah_any).sum(), noah_any.sum()),
                "philsa_mean_freq_noah_any": float(s.loc[noah_any, "philsa_freq"].mean()) if noah_any.any() else np.nan,
                "philsa_any_in_noah_active_share": _safe_share((philsa_any & noah_active).sum(), noah_active.sum()),
                "philsa_mean_freq_noah_active": float(s.loc[noah_active, "philsa_freq"].mean()) if noah_active.any() else np.nan,
                "ai4g_any_share": float(ai4g_any.mean()),
                "ai4g_any_in_noah_any_share": _safe_share((ai4g_any & noah_any).sum(), noah_any.sum()),
                "ai4g_mean_freq_noah_any": float(s.loc[noah_any, "ai4g_freq"].mean()) if noah_any.any() else np.nan,
                "ai4g_any_in_noah_active_share": _safe_share((ai4g_any & noah_active).sum(), noah_active.sum()),
                "ai4g_mean_freq_noah_active": float(s.loc[noah_active, "ai4g_freq"].mean()) if noah_active.any() else np.nan,
                "satellite_any_share": float(satellite_any.mean()),
                "satellite_any_in_noah_any_share": _safe_share((satellite_any & noah_any).sum(), noah_any.sum()),
                "satellite_any_in_noah_active_share": _safe_share((satellite_any & noah_active).sum(), noah_active.sum()),
            }
        )

    return pd.DataFrame(rows)


def _event_metrics():
    if not EVENT_SUMMARY.exists():
        return pd.DataFrame(columns=["city", "event_weighted_noah_any_recall"])
    events = pd.read_csv(EVENT_SUMMARY)
    events = events[events["n_flooded_total"] >= 20].copy()
    rows = []
    for city, sub in events.groupby("city"):
        n_flood = sub["n_flooded_total"].sum()
        n_inside_any = (sub["n_in_noah_low"] + sub["n_in_noah_med"] + sub["n_in_noah_high"]).sum()
        n_inside = sub["n_flooded_in_noah_active"].sum()
        rows.append(
            {
                "city": city,
                "event_flooded_cells_total": int(n_flood),
                "event_flooded_cells_in_noah_any": int(n_inside_any),
                "event_flooded_cells_in_noah_active": int(n_inside),
                "event_weighted_noah_any_recall": _safe_share(n_inside_any, n_flood),
                "event_weighted_noah_recall": _safe_share(n_inside, n_flood),
            }
        )
    return pd.DataFrame(rows)


def _spearman(df, columns):
    return df[columns].corr(method="spearman", min_periods=3)


def _rect_spearman(df, row_cols, col_cols):
    corr = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
    for row_col in row_cols:
        for col_col in col_cols:
            pair = df[[row_col, col_col]].dropna()
            if len(pair) < 3:
                corr.loc[row_col, col_col] = np.nan
            else:
                corr.loc[row_col, col_col] = pair[row_col].corr(pair[col_col], method="spearman")
    return corr


def _plot_corr(corr, labels, title, subtitle, out_stem):
    labels_ordered = [labels.get(col, col) for col in corr.columns]
    data = corr.values

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    fig.patch.set_facecolor("#F7F7F7")
    ax.set_facecolor("white")
    im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(len(labels_ordered)))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels_ordered)))
    ax.set_yticklabels(labels_ordered, fontsize=8.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "NA"
                color = "#666666"
            else:
                text = f"{val:.2f}"
                color = "white" if abs(val) > 0.55 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=8.5, color="#555555")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Spearman correlation", fontsize=9)
    ax.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    png = OUT_DIR / f"{out_stem}.png"
    pdf = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")


def _plot_rect_corr(corr, row_labels, col_labels, title, subtitle, note, out_stem):
    row_names = [row_labels.get(col, col) for col in corr.index]
    col_names = [col_labels.get(col, col) for col in corr.columns]
    data = corr.values.astype(float)

    fig_w = max(8.0, 1.75 * len(col_names) + 3.0)
    fig_h = max(5.2, 0.85 * len(row_names) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#F7F7F7")
    ax.set_facecolor("white")

    im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=9)

    ax.set_xlabel("Validation / support outcome", fontsize=10, fontweight="bold", labelpad=14)
    ax.set_ylabel("City-level condition", fontsize=10, fontweight="bold", labelpad=14)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "NA"
                color = "#555555"
            else:
                text = f"{val:+.2f}"
                color = "white" if abs(val) > 0.55 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(col_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=8.5, color="#555555")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("Spearman correlation", fontsize=9)
    fig.tight_layout(rect=[0.03, 0.07, 0.98, 0.9])

    png = OUT_DIR / f"{out_stem}.png"
    pdf = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")


def main():
    print("Urban / NOAH / satellite correlation analysis")
    cells = _load_cells()
    print(f"Loaded {len(cells):,} city-grid cells")

    cells["urban_cls"] = _sample_ghsl(cells)
    metrics = _city_metrics(cells)
    metrics = metrics.merge(_event_metrics(), on="city", how="left")

    if URBAN_BIAS_SUMMARY.exists():
        urban_bias = pd.read_csv(URBAN_BIAS_SUMMARY)
        metrics = metrics.merge(
            urban_bias.drop(columns=["region"], errors="ignore"),
            left_on="city",
            right_on="city",
            how="left",
            suffixes=("", "_urban_bias"),
        )

    city_metrics_path = OUT_DIR / "urban_noah_city_metrics.csv"
    metrics.to_csv(city_metrics_path, index=False)
    print(f"Saved -> {city_metrics_path}")

    city_cols = [col for col in CITY_LABELS if col in metrics.columns]
    city_corr = _spearman(metrics, city_cols)
    city_corr_path = OUT_DIR / "urban_noah_correlation_citylevel.csv"
    city_corr.to_csv(city_corr_path)
    print(f"Saved -> {city_corr_path}")
    _plot_corr(
        city_corr,
        CITY_LABELS,
        "Urbanisation, NOAH Hazard Coverage, and Satellite Flood Support",
        "City-level Spearman correlations across 10 study-city buffers; interpret as exploratory, not causal.",
        "urban_noah_correlation_citylevel",
    )
    city_driver_cols = [col for col in CLEAR_CITY_DRIVERS if col in metrics.columns]
    city_outcome_cols = [col for col in CLEAR_CITY_OUTCOMES if col in metrics.columns]
    city_clear_corr = _rect_spearman(metrics, city_driver_cols, city_outcome_cols)
    city_clear_path = OUT_DIR / "urban_noah_correlation_citylevel_clear.csv"
    city_clear_corr.to_csv(city_clear_path)
    print(f"Saved -> {city_clear_path}")
    _plot_rect_corr(
        city_clear_corr,
        CLEAR_CITY_DRIVERS,
        CLEAR_CITY_OUTCOMES,
        "Urbanized NOAH Hazard Areas Tend to Receive Weaker Independent Satellite Support",
        "Main figure uses NOAH + GHSL urbanization + PhilSA + AI4G + event recall only. Red means higher row values tend to coincide with higher column values; blue means the opposite.",
        "",
        "urban_noah_correlation_citylevel_clear",
    )

    gs_cols = [col for col in GS_LABELS if col in metrics.columns]
    gs_df = metrics.dropna(subset=["active_cells_sat_gap_share"]).copy()
    gs_corr = _spearman(gs_df, gs_cols)
    gs_corr_path = OUT_DIR / "urban_noah_correlation_groundsource.csv"
    gs_corr.to_csv(gs_corr_path)
    print(f"Saved -> {gs_corr_path}")
    _plot_corr(
        gs_corr,
        GS_LABELS,
        "Urbanisation and the Groundsource-vs-PhilSA Satellite Gap",
        "Groundsource-enhanced Spearman correlations across 5 city buffers; useful as directional evidence only.",
        "urban_noah_correlation_groundsource",
    )
    gs_driver_cols = [col for col in CLEAR_GS_DRIVERS if col in gs_df.columns]
    gs_outcome_cols = [col for col in CLEAR_GS_OUTCOMES if col in gs_df.columns]
    gs_clear_corr = _rect_spearman(gs_df, gs_driver_cols, gs_outcome_cols)
    gs_clear_path = OUT_DIR / "urban_noah_correlation_groundsource_clear.csv"
    gs_clear_corr.to_csv(gs_clear_path)
    print(f"Saved -> {gs_clear_path}")
    _plot_rect_corr(
        gs_clear_corr,
        CLEAR_GS_DRIVERS,
        CLEAR_GS_OUTCOMES,
        "Groundsource Suggests a Larger PhilSA Gap in More Urban NOAH Areas",
        "Spearman correlations across the 5 city buffers with Groundsource data. Use as directional evidence only because n=5.",
        "A positive Groundsource yes / PhilSA no correlation means the gap is larger in cities with higher row values; this is consistent with literature on urban flood under-detection from space.",
        "urban_noah_correlation_groundsource_clear",
    )


if __name__ == "__main__":
    main()
