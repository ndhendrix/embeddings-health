"""
Submit OlmoEarth Studio jobs for one state split into N downloadable pieces.

Large states (> ~100 000 km²) produce ZIPs that exceed the server's ~5-minute
streaming timeout; this script splits the state boundary into a regular grid and
submits one job per cell.  02_collect_results.py will mosaic the pieces into a
single state COG automatically once all pieces are downloaded.

Usage:
    uv run python 01b_submit_split_state.py --state 01           # Alabama, auto-split
    uv run python 01b_submit_split_state.py --state 48 --pieces 8  # Texas, 8 pieces
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box
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

# ~100 000 km² per piece → ~4 GB compressed, well under the server's ~5-min timeout
KM2_PER_PIECE = 100_000


def split_geometry(geom, n_pieces):
    """Split geom into n_pieces using a regular grid that matches its aspect ratio."""
    minx, miny, maxx, maxy = geom.bounds
    aspect = (maxx - minx) / max(maxy - miny, 1e-10)

    best = None
    for n_rows in range(1, n_pieces + 1):
        n_cols = math.ceil(n_pieces / n_rows)
        score = abs(math.log((n_cols / n_rows) / aspect))
        if best is None or score < best[0]:
            best = (score, n_rows, n_cols)
    _, n_rows, n_cols = best

    cell_w = (maxx - minx) / n_cols
    cell_h = (maxy - miny) / n_rows

    pieces = []
    for r in range(n_rows):
        for c in range(n_cols):
            cell = box(
                minx + c * cell_w, miny + r * cell_h,
                minx + (c + 1) * cell_w, miny + (r + 1) * cell_h,
            )
            piece = geom.intersection(cell)
            if not piece.is_empty and piece.area > 1e-8:
                pieces.append(piece)
    return pieces


parser = argparse.ArgumentParser()
parser.add_argument("--state", required=True, help="State FIPS code, e.g. 01")
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--pieces", type=int, default=None,
                    help="Number of pieces (default: auto from land area)")
args = parser.parse_args()

state_fips = args.state.zfill(2)
year = args.year
base_name = f"state_{state_fips}_{year}"

# Load census tract file (download if needed) to get state boundary + area
tracts_zip = TRACTS_DIR / f"tl_{year}_{state_fips}_tract.zip"
if not tracts_zip.exists():
    url = (f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
           f"tl_{year}_{state_fips}_tract.zip")
    print(f"Downloading census tracts for {state_fips}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(tracts_zip, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)

tracts = gpd.read_file(f"zip://{tracts_zip}").to_crs("EPSG:4326")
state_geom = tracts.dissolve().geometry.iloc[0]

# Compute land area in km² (Albers Equal Area) for auto piece count
area_km2 = (
    tracts.to_crs("EPSG:5070").dissolve().geometry.iloc[0].area / 1e6
)
n_pieces = args.pieces or max(2, math.ceil(area_km2 / KM2_PER_PIECE))
print(f"{base_name}: land area ≈ {area_km2:,.0f} km² → {n_pieces} piece(s)")

# Load manifest to check for already-submitted pieces
if MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
else:
    rows = []
manifest_names = {r["name"] for r in rows}

# Split state boundary and submit one job per piece
pieces = split_geometry(state_geom, n_pieces)
print(f"Grid produced {len(pieces)} non-empty cell(s)")

new_rows = []
for i, piece in enumerate(pieces, 1):
    piece_name = f"{base_name}_p{i:02d}"
    if piece_name in manifest_names:
        print(f"  {piece_name} already in manifest — skipping")
        continue

    resp = requests.post(
        f"{BASE_URL}/api/v1/predictions",
        json={
            "project_id": CONFIG["project_id"],
            "model_id": CONFIG["model_id"],
            "name": piece_name,
            "geojson": {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "start_time": f"{year}-01-01T00:00:00Z",
                        "end_time": f"{year}-12-31T23:59:59Z",
                    },
                    "geometry": piece.__geo_interface__,
                }],
            },
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    prediction_id = resp.json()["records"][0]["id"]
    new_rows.append({"name": piece_name, "prediction_id": prediction_id, "status": "submitted"})
    print(f"  submitted {piece_name} → {prediction_id}")

if new_rows:
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "prediction_id", "status"])
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"Submitted {len(new_rows)} piece job(s). Run 02_collect_results.py to collect.")
else:
    print("All pieces already in manifest — nothing submitted.")
