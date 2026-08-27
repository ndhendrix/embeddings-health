"""
Submit OlmoEarth Studio jobs for one state split into N downloadable pieces.

Large states (> ~100 000 km²) produce ZIPs that exceed the server's ~5-minute
streaming timeout; this script groups whole counties into downloadable pieces
and submits one job per county group.  02_collect_results.py will mosaic the
pieces into a single state COG automatically once all pieces are downloaded.

Usage:
    uv run python 01b_submit_split_state.py --state 01             # Alabama, auto-split
    uv run python 01b_submit_split_state.py --state 48 --pieces 8  # Texas, 8 county groups
    uv run python 01b_submit_split_state.py --state 48 --dry-run   # preview groups only
    uv run python 01b_submit_split_state.py --state 48 --split-mode grid  # old grid mode
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
from shapely.ops import unary_union
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


def split_geometry_grid(geom, n_pieces):
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


def split_tracts_into_county_groups(
    tracts: gpd.GeoDataFrame,
    n_pieces: int | None,
    km2_per_piece: float,
) -> list[dict]:
    """Group whole counties into pieces sized for reliable downloads."""
    county_cols = ["STATEFP", "COUNTYFP"]
    counties = tracts.dissolve(by=county_cols, as_index=False)
    county_areas = (
        tracts.to_crs("EPSG:5070")
        .dissolve(by=county_cols, as_index=False)
        .assign(area_km2=lambda df: df.geometry.area / 1e6)
    )[county_cols + ["area_km2"]]
    counties = counties.merge(county_areas, on=county_cols, how="left")
    counties["county_geoid"] = counties["STATEFP"] + counties["COUNTYFP"]

    county_records = sorted(
        counties[["county_geoid", "area_km2", "geometry"]].to_dict("records"),
        key=lambda r: r["area_km2"],
        reverse=True,
    )

    if n_pieces is not None:
        bins = [{"area_km2": 0.0, "counties": [], "geometries": []} for _ in range(n_pieces)]
        for county in county_records:
            target = min(bins, key=lambda b: b["area_km2"])
            _add_county_to_group(target, county)
    else:
        bins = []
        for county in county_records:
            fitting = [
                b for b in bins
                if b["area_km2"] + county["area_km2"] <= km2_per_piece
            ]
            if fitting:
                target = min(fitting, key=lambda b: b["area_km2"])
            else:
                target = {"area_km2": 0.0, "counties": [], "geometries": []}
                bins.append(target)
            _add_county_to_group(target, county)

    groups = []
    for group in bins:
        if not group["counties"]:
            continue
        groups.append({
            "geometry": unary_union(group["geometries"]),
            "area_km2": group["area_km2"],
            "counties": sorted(group["counties"]),
        })
    return groups


def _add_county_to_group(group: dict, county: dict) -> None:
    group["area_km2"] += float(county["area_km2"])
    group["counties"].append(county["county_geoid"])
    group["geometries"].append(county["geometry"])


parser = argparse.ArgumentParser()
parser.add_argument("--state", required=True, help="State FIPS code, e.g. 01")
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--pieces", type=int, default=None,
                    help="Number of pieces (default: auto from land area)")
parser.add_argument("--split-mode", choices=["county", "grid"], default="county",
                    help="Split large states by county groups (default) or regular grid cells")
parser.add_argument("--dry-run", action="store_true",
                    help="Print planned pieces without submitting jobs or updating the manifest")
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
print(f"{base_name}: land area ≈ {area_km2:,.0f} km² → {n_pieces} piece(s)  mode={args.split_mode}")

# Load manifest to check for already-submitted pieces
if MANIFEST.exists():
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
else:
    rows = []
manifest_names = {r["name"] for r in rows}

# Split state boundary and submit one job per piece
if args.split_mode == "county":
    pieces = split_tracts_into_county_groups(
        tracts,
        n_pieces=args.pieces,
        km2_per_piece=KM2_PER_PIECE,
    )
    print(f"County grouping produced {len(pieces)} group(s)")
else:
    pieces = [
        {"geometry": geom, "area_km2": None, "counties": []}
        for geom in split_geometry_grid(state_geom, n_pieces)
    ]
    print(f"Grid produced {len(pieces)} non-empty cell(s)")

new_rows = []
for i, piece in enumerate(pieces, 1):
    piece_name = f"{base_name}_p{i:02d}"
    if piece_name in manifest_names and not args.dry_run:
        print(f"  {piece_name} already in manifest — skipping")
        continue
    if piece["counties"]:
        area = piece["area_km2"]
        print(f"  {piece_name}: {len(piece['counties'])} counties, {area:,.0f} km²")
    if args.dry_run:
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
                        "split_mode": args.split_mode,
                        "counties": ",".join(piece["counties"]),
                    },
                    "geometry": piece["geometry"].__geo_interface__,
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
elif args.dry_run:
    print("Dry run complete — no jobs submitted and manifest unchanged.")
else:
    print("All pieces already in manifest — nothing submitted.")
