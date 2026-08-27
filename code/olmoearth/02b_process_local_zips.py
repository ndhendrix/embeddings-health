"""
Process ZIP files downloaded manually from the OlmoEarth Studio web UI.

Place the downloaded ZIPs in data/olmoearth/downloads/ before running.
The ZIP filenames must include the job name, e.g. state_01_2022.zip.
The script extracts tiles, mosaics if needed, writes a COG, and marks
the corresponding manifest row as done.

Usage:
    uv run python 02b_process_local_zips.py
    uv run python 02b_process_local_zips.py --zip-dir /path/to/zips
"""
import argparse
import csv
import re
import tempfile
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from mosaic_utils import is_readable_tif, write_cog_from_tifs

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

PROJECT_ROOT = HERE.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "olmoearth"
MANIFEST = HERE / "manifest.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--zip-dir", type=Path,
                    default=OUT_DIR / "downloads",
                    help="Directory containing downloaded ZIP files")
args = parser.parse_args()

if not args.zip_dir.exists():
    raise SystemExit(f"ZIP directory not found: {args.zip_dir}\n"
                     "Create it and place downloaded ZIPs inside.")

zip_files = sorted(args.zip_dir.glob("*.zip"))
if not zip_files:
    raise SystemExit(f"No .zip files found in {args.zip_dir}")

print(f"Found {len(zip_files)} ZIP(s) in {args.zip_dir}")

# Load manifest (create minimal one if absent)
if MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
else:
    rows = []

manifest_by_name = {r["name"]: r for r in rows}


def save_manifest():
    with open(MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "prediction_id", "status"])
        writer.writeheader()
        writer.writerows(rows)


OUT_DIR.mkdir(parents=True, exist_ok=True)

processed = 0
for zip_path in zip_files:
    # Extract job name from filename — expects something like state_01_2022.zip
    m = re.search(r"(state_\d{2}_\d{4})", zip_path.stem)
    if not m:
        print(f"  SKIP {zip_path.name} — cannot parse job name (expected state_XX_YYYY)")
        continue
    name = m.group(1)
    fips = name.split("_")[1]
    year = name.split("_")[2]

    out_path = OUT_DIR / f"{name}.tif"
    if out_path.exists():
        if is_readable_tif(out_path):
            print(f"  {name}: COG already exists — skipping")
            if name in manifest_by_name:
                manifest_by_name[name]["status"] = "done"
            continue
        print(f"  {name}: existing COG is unreadable — rebuilding")

    print(f"  {name}: processing {zip_path.name}...")

    with tempfile.TemporaryDirectory(dir=OUT_DIR) as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            tif_names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
            zf.extractall(tmp)
        print(f"    {len(tif_names)} tile(s) in ZIP")

        tif_paths = [tmp / n for n in tif_names]
        if not tif_paths:
            print(f"    WARNING: no TIF files found in {zip_path.name}")
            continue

        data_shape = write_cog_from_tifs(tif_paths, out_path)
    print(f"    → {out_path.name}  shape={data_shape}")

    # Update or insert manifest row
    if name in manifest_by_name:
        manifest_by_name[name]["status"] = "done"
    else:
        row = {"name": name, "prediction_id": "", "status": "done"}
        rows.append(row)
        manifest_by_name[name] = row

    save_manifest()
    processed += 1

print(f"\nProcessed {processed} ZIP(s). Run 03_aggregate.py for each state next.")
