"""
Download flood hazard data from Google Drive.

This script downloads the NOAH flood hazard shapefiles that are too large
for GitHub. Run this once after cloning the repository.

Source:
    5yr:   https://drive.google.com/drive/folders/17ecJuf2vnkrpCzNVLes0fI08XFsR1x8N?usp=drive_link
    100yr: https://drive.google.com/drive/folders/10pCWTfU-gVuAbdx4gdUGaDcNrSzMz0Mm?usp=drive_link

Each zip is extracted to:
    data/noah/5yr/<Province>/
    data/noah/100yr/<Province>/

Usage:
    python setup_data/download_noah.py
"""

import os
import shutil
import time
import zipfile
import gdown

# --- Configuration ---

# Mapping: local folder name → (zip name in Drive, 5yr file ID, 100yr file ID)
# 5yr file IDs from the Drive listing. 100yr IDs need to be added.
# To get a file ID: right-click file in Drive → Share → Copy link
# Link format: https://drive.google.com/file/d/<FILE_ID>/view
PROVINCES = {
    "Cagayan": {
        "zip_name": "Cagayan",
        "5yr":   "1YXKoMUOe646cI5kvwWssO-S9lne9jEFu",
        "100yr": "1CfgJUqN6wv8wBG3Oiu2r-VR_lGk87V24",
    },
    "Agusan del Norte": {
        "zip_name": "AgusanDelNorte",
        "5yr":   None,
        "100yr": "1pIbz8VhHp8LoQNpDsWFk70uZpDBm9hki",
    },
    "Pangasinan": {
        "zip_name": "Pangasinan",
        "5yr":   "1OkYs5yioCHoNWuOjkDcnuEaHKOXUl-oM",
        "100yr": "151h2fMuz6MhulAUd6NPLJuwBvQTxhlUw",
    },
    "Pampanga": {
        "zip_name": "Pampanga",
        "5yr":   "1GWtqxt6xnEIkzd7NV-r9m8JmU4VJtbdV",
        "100yr": None,
    },
    "Maguindanao": {
        "zip_name": "Maguindanao",
        "5yr":   "1uTjR1j3jiWCJg_QQS3C4IH-buMC3uAIN",
        "100yr": "1iPXWaqGTZz4Aig7iDWQvzBi_nDUKTYIl",
    },
    "Metropolitan Manila": {
        "zip_name": "MetroManila",
        "5yr":   "1IC5LnscUt7D_5c1BvlwcngR3cFKgDshV",
        "100yr": "1n-IrwfWqLRDyDg-wmt68luk_DU5Xzg9S",
    },
    "Camarines Sur": {
        "zip_name": "CamarinesSur",
        "5yr":   "1yPQy0iuPo-B0MpVEcvoMZWDB9Pdts_kY",
        "100yr": "11FfVRGtXKnQwiJmzkNPBWWlyFgKr--9M",
    },
    "Misamis Oriental": {
        "zip_name": "MisamisOriental",
        "5yr":   "15L9FcynpUmvr3A5-QIf0YL4TchKpB0Xi",
        "100yr": "1kB65HdBLV2rur5kVzl_t9dHpu3k3bi5h",
    },
    "Camarines Norte": {
        "zip_name": "CamarinesNorte",
        "5yr":   "1lWve3Uzs4UF3OyM3vthGGOFVBU4dnZfI",
        "100yr": None,
    },
    "Isabela": {
        "zip_name": "Isabela",
        "5yr":   "1ZOn-QR9dm6fQDFSMRjNZIL_bQWVYfYvs",
        "100yr": "1rJvZpIXior4TmuzjEatcTYca2mHwl_3n",
    },
}

RETURN_PERIODS = ["5yr", "100yr"]

# Paths — script lives in setup_data/, data lives at project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
NOAH_DIR = os.path.join(DATA_DIR, "noah")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")


def province_exists(province, return_period):
    """Check if a province's flood data already exists."""
    # Check both the local name and the zip name as folder names
    zip_name = PROVINCES[province]["zip_name"]
    for folder_name in [province, zip_name]:
        province_dir = os.path.join(NOAH_DIR, return_period, folder_name)
        if os.path.exists(province_dir):
            if any(f.endswith(".shp") for f in os.listdir(province_dir)):
                return True
    return False


def download_and_extract():
    """Download flood data from Google Drive and extract zip files."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    downloaded = 0
    skipped_exists = 0
    skipped_no_id = 0

    print("Downloading flood hazard data from Google Drive...\n")

    for rp in RETURN_PERIODS:
        for province, info in PROVINCES.items():
            file_id = info.get(rp)

            # Skip if already downloaded
            if province_exists(province, rp):
                print(f"  [OK] {rp}/{province} — already exists, skipping")
                skipped_exists += 1
                continue

            # Skip if no file ID
            if file_id is None:
                print(f"  [SKIP] {rp}/{province} — no file ID configured, skipping")
                skipped_no_id += 1
                continue

            # Download the zip by file ID
            zip_path = os.path.join(DOWNLOAD_DIR, f"{info['zip_name']}_{rp}.zip")
            url = f"https://drive.google.com/file/d/{file_id}/view"

            print(f"  Downloading {rp}/{province}...")
            success = False
            for attempt in range(3):
                try:
                    gdown.download(url, zip_path, quiet=False, fuzzy=True)
                    if os.path.exists(zip_path):
                        success = True
                        break
                except Exception as e:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        print(f"      [WARN] Attempt {attempt+1} failed, retrying in {wait}s...")
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        time.sleep(wait)
                    else:
                        print(f"  [FAIL] {rp}/{province} — download failed after 3 attempts: {e}\n")
                        if os.path.exists(zip_path):
                            os.remove(zip_path)

            if not success:
                continue

            # Brief pause between downloads to avoid rate limiting
            time.sleep(2)

            # Extract into data/noah/<rp>/
            target_dir = os.path.join(NOAH_DIR, rp)
            os.makedirs(target_dir, exist_ok=True)

            print(f"  Extracting {rp}/{province}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                # Check if zip contains a top-level folder or loose files
                names = z.namelist()
                has_folder = any("/" in n for n in names)
                first_parts = set(n.split("/")[0] for n in names if "/" in n)
                single_folder = len(first_parts) == 1

                if has_folder and single_folder:
                    # Zip has a single top-level folder — extract as-is
                    z.extractall(target_dir)
                    # Rename extracted folder to local province name if different
                    extracted_folder = first_parts.pop()
                    extracted_path = os.path.join(target_dir, extracted_folder)
                    desired_path = os.path.join(target_dir, province)
                    if extracted_folder != province and os.path.exists(extracted_path):
                        os.rename(extracted_path, desired_path)
                else:
                    # Flat zip — extract into a province subfolder
                    province_dir = os.path.join(target_dir, province)
                    os.makedirs(province_dir, exist_ok=True)
                    z.extractall(province_dir)

            os.remove(zip_path)
            downloaded += 1

            if province_exists(province, rp):
                print(f"  [OK] {rp}/{province} — done\n")
            else:
                # List what was extracted to help debug
                print(f"  [WARN] {rp}/{province} — extracted but .shp not found")
                print(f"      Zip folder name may differ from '{province}'")
                extracted = os.listdir(target_dir)
                print(f"      Folders in {rp}/: {extracted}\n")

    # Clean up
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)

    # Summary
    print(f"\n{'='*55}")
    print(f" DOWNLOAD SUMMARY")
    print(f"{'='*55}")
    print(f"  Downloaded:  {downloaded}")
    print(f"  Already had: {skipped_exists}")
    print(f"  Skipped:     {skipped_no_id}\n")

    for rp in RETURN_PERIODS:
        for province in PROVINCES:
            status = "OK" if province_exists(province, rp) else "MISSING"
            print(f"  [{status}] {rp}/{province}")

    print()


if __name__ == "__main__":
    download_and_extract()
