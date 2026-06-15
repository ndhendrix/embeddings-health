"""
Submit split jobs for all CONUS states that exceed the server download size limit.

The OlmoEarth download endpoint closes connections after ~5 minutes, limiting
each download to ~5 GB at the server's ~17 MB/s outbound rate.  States with
land area > 100 000 km² typically exceed this and must be submitted as multiple
smaller county-group pieces via 01b_submit_split_state.py.

Skips any state where:
  - state_{fips}_{year}.tif already exists on disk (downloaded successfully)
  - split piece jobs are already in the manifest

Usage:
    uv run python 04b_submit_large_states.py
    uv run python 04b_submit_large_states.py --year 2022
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "olmoearth"
MANIFEST = HERE / "manifest.csv"
MANIFEST_FIELDS = ["name", "prediction_id", "status"]

# Census 2020 land area (km²) for CONUS states + DC
STATE_AREA_KM2 = {
    "01": 131426, "04": 294207, "05": 134771, "06": 403466, "08": 268431,
    "09":  12542, "10":   5047, "11":    159, "12": 138887, "13": 148959,
    "16": 214045, "17": 143793, "18":  92789, "19": 144701, "20": 211754,
    "21": 102269, "22": 111898, "23":  79884, "24":  25142, "25":  20202,
    "26": 146435, "27": 206232, "28": 121531, "29": 178040, "30": 376962,
    "31": 198974, "32": 284332, "33":  23187, "34":  19047, "35": 314161,
    "36": 122057, "37": 125920, "38": 178711, "39": 105829, "40": 177847,
    "41": 248608, "42": 115883, "44":   2678, "45":  77857, "46": 196350,
    "47": 106798, "48": 676587, "49": 212818, "50":  23871, "51": 102279,
    "53": 172119, "54":  62259, "55": 140268, "56": 251470,
}

CONUS_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# Must match 01b_submit_split_state.py
THRESHOLD_KM2 = 100_000

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--split-mode", choices=["county", "grid"], default="county",
                    help="Split large states by county groups (default) or regular grid cells")
args = parser.parse_args()

# Load the full manifest so we can mark superseded whole-state jobs as "split"
piece_re = re.compile(r"^(state_\d{2}_\d{4})_p\d+$")
rows: list[dict] = []
manifest_by_name: dict[str, dict] = {}
pieces_submitted: set[str] = set()


def _clean_manifest_rows(raw_rows) -> list[dict]:
    """Drop stray CSV overflow fields before re-writing manifest rows."""
    return [{field: row.get(field, "") for field in MANIFEST_FIELDS} for row in raw_rows]


if MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = _clean_manifest_rows(csv.DictReader(fh))
    manifest_by_name = {r["name"]: r for r in rows}
    for r in rows:
        m = piece_re.match(r["name"])
        if m:
            pieces_submitted.add(m.group(1))


def save_manifest():
    with open(MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(_clean_manifest_rows(rows))


submit_script = HERE / "01b_submit_split_state.py"
skipped = needs_split = submitted = 0

for fips in CONUS_FIPS:
    area = STATE_AREA_KM2.get(fips, 0)
    state_name = f"state_{fips}_{args.year}"
    cog_path = OUT_DIR / f"{state_name}.tif"

    if cog_path.exists():
        print(f"  {state_name}: COG exists — skip")
        skipped += 1
        continue

    if state_name in pieces_submitted:
        print(f"  {state_name}: piece jobs already in manifest — skip")
        if state_name in manifest_by_name and manifest_by_name[state_name]["status"] == "submitted":
            manifest_by_name[state_name]["status"] = "split"
            save_manifest()
        skipped += 1
        continue

    if area <= THRESHOLD_KM2:
        print(f"  {state_name}: {area:,} km² ≤ threshold — no split needed")
        skipped += 1
        continue

    needs_split += 1
    result = subprocess.run(
        [sys.executable, str(submit_script),
         "--state", fips, "--year", str(args.year),
         "--split-mode", args.split_mode],
        capture_output=True, text=True,
    )
    out = (result.stdout.strip() + result.stderr.strip()).replace("\n", " | ")
    status = "OK" if result.returncode == 0 else f"ERROR {result.returncode}"
    print(f"  {state_name} ({area:,} km²): [{status}] {out}")
    if result.returncode == 0:
        submitted += 1
        # Reload manifest from disk to pick up piece entries just written by
        # 01b_submit_split_state.py, then mark the original job as "split".
        with open(MANIFEST) as fh:
            rows = _clean_manifest_rows(csv.DictReader(fh))
        manifest_by_name = {r["name"]: r for r in rows}
        if state_name in manifest_by_name:
            manifest_by_name[state_name]["status"] = "split"
            save_manifest()

print(f"\n{needs_split} large state(s) found; {submitted} submitted; {skipped} skipped.")
print("Run 02_collect_results.py (repeatedly) to download and mosaic.")
