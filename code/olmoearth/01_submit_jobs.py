"""
Submit one OlmoEarth Studio prediction job per state (or other large AOI).

The API handles internal tiling; we supply the full state boundary and get back
one COG per job. For CONUS, run this once per state (50 jobs total).

Usage:
    uv run python 01_submit_jobs.py                   # Rhode Island
    uv run python 01_submit_jobs.py --state 06 2022   # California, 2022
"""
import argparse
import csv
import json
import os
from pathlib import Path

import geopandas as gpd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

PROJECT_ROOT = HERE.parent.parent
TRACTS_DIR = PROJECT_ROOT / "data"
MANIFEST = HERE / "manifest.csv"
CONFIG = json.loads((HERE / "config.json").read_text())

BASE_URL = "https://olmoearth.allenai.org"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OLMOEARTH_API_KEY']}",
    "Content-Type": "application/json",
}

parser = argparse.ArgumentParser()
parser.add_argument("--state", default="44", help="State FIPS code (default: 44 = Rhode Island)")
parser.add_argument("--year", type=int, default=2022)
args = parser.parse_args()

state_fips = args.state.zfill(2)
year = args.year
job_name = f"state_{state_fips}_{year}"

# Resume check
if MANIFEST.exists():
    with open(MANIFEST) as fh:
        if any(row["name"] == job_name for row in csv.DictReader(fh)):
            print(f"{job_name} already in manifest — skipping submission.")
            raise SystemExit(0)

# Download tract file for this state if needed (used only for dissolve)
tracts_zip = TRACTS_DIR / f"tl_{year}_{state_fips}_tract.zip"
if not tracts_zip.exists():
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(tracts_zip, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)

tracts = gpd.read_file(f"zip://{tracts_zip}").to_crs("EPSG:4326")
state_geom = tracts.dissolve().geometry.iloc[0]
print(f"State boundary: {state_geom.geom_type}, {len(tracts)} tracts")

resp = requests.post(
    f"{BASE_URL}/api/v1/predictions",
    json={
        "project_id": CONFIG["project_id"],
        "model_id": CONFIG["model_id"],
        "name": job_name,
        "geojson": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "start_time": f"{year}-01-01T00:00:00Z",
                    "end_time": f"{year}-12-31T23:59:59Z",
                },
                "geometry": state_geom.__geo_interface__,
            }],
        },
    },
    headers=HEADERS,
)
resp.raise_for_status()
prediction_id = resp.json()["records"][0]["id"]

write_header = not MANIFEST.exists()
with open(MANIFEST, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["name", "prediction_id", "status"])
    if write_header:
        writer.writeheader()
    writer.writerow({"name": job_name, "prediction_id": prediction_id, "status": "submitted"})

print(f"Submitted {job_name} → {prediction_id}")
print("Run 02_collect_results.py to poll and download.")
