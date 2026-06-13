"""Aggregate all collected state COGs, then concatenate to a single CONUS CSV.

For each FIPS code that has a state COG but no CSV yet, runs 03_aggregate.py.
Then concatenates all state CSVs into:
    data/olmoearth_studio_conus_{year}_embeddings.csv

Usage:
    uv run python 05_batch_aggregate.py
    uv run python 05_batch_aggregate.py --year 2022
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONUS_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2022)
args = parser.parse_args()

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
aggregate_script = HERE / "03_aggregate.py"

# ── Step 1: aggregate any state with a COG but no CSV ────────────────────────

for fips in CONUS_FIPS:
    cog = PROJECT_ROOT / "data" / "olmoearth" / f"state_{fips}_{args.year}.tif"
    out_csv = PROJECT_ROOT / "data" / f"olmoearth_studio_{fips}_{args.year}_embeddings.csv"
    if not cog.exists():
        print(f"[{fips}] no COG — skipping (run 02_collect_results.py first)")
        continue
    if out_csv.exists():
        print(f"[{fips}] CSV already exists — skipping")
        continue
    print(f"[{fips}] Aggregating...")
    result = subprocess.run(
        [sys.executable, str(aggregate_script), "--state", fips, "--year", str(args.year)],
    )
    if result.returncode != 0:
        print(f"[{fips}] ERROR (returncode {result.returncode}) — continuing")

# ── Step 2: concatenate all available state CSVs ─────────────────────────────

print("\nConcatenating state CSVs...")
dfs = []
missing = []
for fips in CONUS_FIPS:
    p = PROJECT_ROOT / "data" / f"olmoearth_studio_{fips}_{args.year}_embeddings.csv"
    if p.exists():
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  {fips}: {len(df):,} tracts")
    else:
        missing.append(fips)

if missing:
    print(f"\nWARNING: {len(missing)} states missing CSVs: {missing}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    out_path = PROJECT_ROOT / "data" / f"olmoearth_studio_conus_{args.year}_embeddings.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n{len(combined):,} total tracts → {out_path.name}")
else:
    print("No state CSVs found — nothing to concatenate.")
