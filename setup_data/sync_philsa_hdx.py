"""
Sync PhilSA flood resources from HDX into data/philsa_satellite_flood.

This mirrors the exact resource filenames published by PhilSA on HDX for flood
datasets, including archived datasets. It keeps distinct resources for the same
date when PhilSA publishes multiple positions or satellite types.

It does not delete locally derived files such as generated shapefiles from map
tiles; those are reported separately in the audit but left untouched.

Usage:
  python setup_data/sync_philsa_hdx.py
  python setup_data/sync_philsa_hdx.py --audit-only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "philsa_satellite_flood"
OUT_DIR = ROOT / "output"
HDX_API = "https://data.humdata.org/api/3/action/package_search"
USER_AGENT = "flooding-infrastructure-philippines/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Do not download missing resources.")
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Max number of HDX datasets to request.",
    )
    return parser.parse_args()


def fetch_hdx_results(rows: int) -> list[dict]:
    params = {
        "q": "flood",
        "fq": "organization:philsa",
        "rows": rows,
        "sort": "metadata_modified desc",
    }
    resp = requests.get(HDX_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"HDX API failed: {payload.get('error')}")
    return payload["result"]["results"]


def is_target_resource(name: str, fmt: str) -> bool:
    if not name:
        return False
    fmt = (fmt or "").upper()
    return (
        fmt in {"SHP", "PNG"}
        or name.endswith("_shp.zip")
        or name.endswith("_maps.zip")
        or "_shp_" in name
        or "_maps_" in name
    )


def build_inventory(results: list[dict]) -> list[dict]:
    rows = []
    for ds in results:
        for res in ds.get("resources", []):
            name = (res.get("name") or "").strip()
            fmt = (res.get("format") or "").upper()
            if not is_target_resource(name, fmt):
                continue
            rows.append(
                {
                    "dataset_name": ds.get("name"),
                    "dataset_title": ds.get("title"),
                    "archived": bool(ds.get("archived")),
                    "resource_name": name,
                    "resource_format": fmt,
                    "resource_url": res.get("url") or res.get("download_url") or "",
                }
            )
    return rows


def assert_no_remote_collisions(rows: list[dict]) -> None:
    name_counts = Counter(r["resource_name"] for r in rows)
    dup_names = [name for name, count in name_counts.items() if count > 1]
    if dup_names:
        raise RuntimeError(f"Duplicate remote resource names detected: {dup_names[:10]}")

    url_counts = Counter(r["resource_url"] for r in rows if r["resource_url"])
    dup_urls = [url for url, count in url_counts.items() if count > 1]
    if dup_urls:
        raise RuntimeError(f"Duplicate remote resource URLs detected: {dup_urls[:10]}")


def classify_local_status(rows: list[dict]) -> list[dict]:
    local_files = {p.name for p in DATA_DIR.iterdir() if p.is_file()}
    local_dirs = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}
    remote_shp_names = {
        r["resource_name"]
        for r in rows
        if r["resource_format"] == "SHP" or r["resource_name"].endswith("_shp.zip") or "_shp_" in r["resource_name"]
    }

    audited = []
    for row in rows:
        name = row["resource_name"]
        status = "missing"
        represented_by = ""
        if name in local_files:
            status = "present_exact"
            represented_by = name
        elif name.endswith(".zip") and "_maps" in name and name[:-4] in local_dirs:
            status = "present_as_extracted_folder"
            represented_by = name[:-4]
        elif name.endswith(".zip") and "_maps" in name:
            shp_equiv = name.replace("_maps", "_shp")
            if shp_equiv in local_files:
                if shp_equiv in remote_shp_names:
                    status = "represented_by_shapefile_zip"
                else:
                    status = "represented_by_generated_shapefile"
                represented_by = shp_equiv

        audited.append({**row, "local_status": status, "represented_by": represented_by})
    return audited


def write_audit_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "dataset_title",
        "archived",
        "resource_name",
        "resource_format",
        "resource_url",
        "local_status",
        "represented_by",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset_name"], r["resource_name"])))


def download_resource(row: dict) -> str:
    url = row["resource_url"]
    name = row["resource_name"]
    dest = DATA_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return str(dest)


def main() -> int:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PhilSA HDX sync")
    print("=" * 72)
    print(f"Data dir: {DATA_DIR}")

    results = fetch_hdx_results(args.rows)
    print(f"Fetched {len(results)} HDX datasets")

    inventory = build_inventory(results)
    assert_no_remote_collisions(inventory)
    print(f"Target resources: {len(inventory)}")

    audited = classify_local_status(inventory)
    audit_path = OUT_DIR / "philsa_hdx_audit_2026-04-14.csv"
    write_audit_csv(audited, audit_path)

    status_counts = Counter(row["local_status"] for row in audited)
    for key in sorted(status_counts):
        print(f"{key}: {status_counts[key]}")

    missing_exact = [row for row in audited if row["local_status"] != "present_exact"]
    print(f"Missing exact resource files: {len(missing_exact)}")

    if args.audit_only:
        print(f"Audit only. Wrote {audit_path}")
        return 0

    downloaded = 0
    for idx, row in enumerate(missing_exact, start=1):
        name = row["resource_name"]
        print(f"[{idx}/{len(missing_exact)}] {name}")
        try:
            download_resource(row)
            downloaded += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print(f"Downloaded {downloaded} resource(s)")

    audited = classify_local_status(inventory)
    write_audit_csv(audited, audit_path)
    status_counts = Counter(row["local_status"] for row in audited)
    print("Post-sync status:")
    for key in sorted(status_counts):
        print(f"{key}: {status_counts[key]}")
    print(f"Audit written to {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
