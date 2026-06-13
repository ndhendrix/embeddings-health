"""Submit one OlmoEarth Studio job for every CONUS state (48 contiguous + DC).

Calls 01_submit_jobs.py for each FIPS code; that script is idempotent so
already-submitted states are skipped automatically.

Usage:
    uv run python 04_submit_conus.py
    uv run python 04_submit_conus.py --year 2022
"""
import argparse
import subprocess
import sys
from pathlib import Path

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

submit_script = Path(__file__).parent / "01_submit_jobs.py"
total = len(CONUS_FIPS)

for i, fips in enumerate(CONUS_FIPS, 1):
    result = subprocess.run(
        [sys.executable, str(submit_script), "--state", fips, "--year", str(args.year)],
        capture_output=True, text=True,
    )
    msg = (result.stdout.strip() or result.stderr.strip()).splitlines()[0]
    print(f"[{i}/{total}] {fips}: {msg}")
