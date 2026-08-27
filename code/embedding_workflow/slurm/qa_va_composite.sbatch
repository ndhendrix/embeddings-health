#!/bin/bash
#SBATCH --job-name=qa-va-composite
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${COMPOSITE:?COMPOSITE is required}"

CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
TRACTS="${TRACTS:-$SCRATCH/embeddings-health/data/census_tracts/tl_2020_51_tract.zip}"
PY="${PY:-$CACHE_ROOT/venv-3.11-cpu/bin/python}"
export COMPOSITE TRACTS

cd "$REPO_DIR"
"$PY" - <<'PY'
from pathlib import Path
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds
from pyproj import Transformer
from shapely.geometry import box

composite = Path(os.environ["COMPOSITE"])
tracts_path = Path(os.environ["TRACTS"])

if not composite.is_file() or composite.stat().st_size == 0:
    raise SystemExit(f"missing composite: {composite}")

points = [
    ("Richmond", -77.4360, 37.5407),
    ("Norfolk", -76.2859, 36.8508),
    ("Arlington", -77.0910, 38.8816),
    ("Roanoke", -79.9414, 37.2710),
]

with rasterio.open(composite) as src:
    print(f"composite={composite}")
    print(f"shape={(src.count, src.height, src.width)} crs={src.crs} bounds={src.bounds}")
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    point_hits = 0
    for name, lon, lat in points:
        x, y = transformer.transform(lon, lat)
        r, c = rowcol(src.transform, x, y)
        window = Window(max(0, c - 128), max(0, r - 128), 256, 256)
        data = src.read(window=window, boundless=True, fill_value=np.nan)
        any_valid = np.isfinite(data).any(axis=0)
        valid_pixels = int(any_valid.sum())
        print(f"{name}: rowcol={(r, c)} any_valid_pixels={valid_pixels}/{any_valid.size}")
        point_hits += valid_pixels > 0

    tracts = gpd.read_file(tracts_path).to_crs(src.crs)
    tracts = tracts[tracts.geometry.intersects(box(*src.bounds))]
    valid_tracts = 0
    checked = 0
    for _, tract in tracts.iterrows():
        raw = from_bounds(*tract.geometry.bounds, transform=src.transform)
        c0 = max(0, int(np.floor(raw.col_off)))
        r0 = max(0, int(np.floor(raw.row_off)))
        c1 = min(src.width, int(np.ceil(raw.col_off + raw.width)))
        r1 = min(src.height, int(np.ceil(raw.row_off + raw.height)))
        if c0 >= c1 or r0 >= r1:
            continue
        window = Window(c0, r0, c1 - c0, r1 - r0)
        mask = geometry_mask(
            [tract.geometry],
            out_shape=(r1 - r0, c1 - c0),
            transform=rasterio.windows.transform(window, src.transform),
            invert=True,
            all_touched=False,
        )
        if not mask.any():
            continue
        data = src.read(1, window=window)
        checked += 1
        if np.isfinite(data[mask]).any():
            valid_tracts += 1

    print(f"tracts_checked={checked} tracts_with_finite_band1={valid_tracts}")

if point_hits < 3:
    raise SystemExit(f"QA failed: only {point_hits}/4 named VA point windows had finite pixels")
if valid_tracts < 100:
    raise SystemExit(f"QA failed: only {valid_tracts} VA tracts had finite band-1 pixels")
print("QA passed")
PY
