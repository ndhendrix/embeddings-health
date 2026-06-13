"""
PCA-reduce OlmoEarth Studio embeddings and aggregate to census tract statistics.

Steps:
  1. Sample pixels from the state COG to fit PCA(64).
  2. For each tract: mask to tract boundary, apply PCA, compute
     mean / median / max / min / std across pixels for each PC dimension.
  3. Write CSV matching the schema of data/alphaearth_embeddings.csv.

Usage:
    uv run python 03_aggregate.py
    uv run python 03_aggregate.py --state 06 --year 2022   # California
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from sklearn.decomposition import PCA
from tqdm import tqdm
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

PROJECT_ROOT = HERE.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--state", default="44", help="State FIPS (default: 44 = Rhode Island)")
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--n-components", type=int, default=64)
parser.add_argument("--sample-frac", type=float, default=0.02,
                    help="Fraction of valid pixels to sample for PCA fitting (default 0.02)")
args = parser.parse_args()

state_fips = args.state.zfill(2)
cog_path = PROJECT_ROOT / "data" / "olmoearth" / f"state_{state_fips}_{args.year}.tif"
tracts_zip = PROJECT_ROOT / "data" / f"tl_{args.year}_{state_fips}_tract.zip"
out_path = PROJECT_ROOT / "data" / f"olmoearth_studio_{state_fips}_{args.year}_embeddings.csv"

if not cog_path.exists():
    raise FileNotFoundError(f"State COG not found: {cog_path}. Run 02_collect_results.py first.")

# ── Step 1: sample pixels and fit PCA ────────────────────────────────────────

print(f"Fitting PCA({args.n_components}) on sample from {cog_path.name}...")
samples = []
with rasterio.open(cog_path) as src:
    nodata = src.nodata
    n_bands = src.count
    crs = src.crs
    for _, window in src.block_windows(1):
        block = src.read(window=window).astype("float32")  # (C, h, w)
        valid_mask = block[0] != nodata
        valid = block[:, valid_mask].T  # (n_valid, C)
        if len(valid) == 0:
            continue
        n = max(1, int(len(valid) * args.sample_frac))
        idx = np.random.default_rng(seed=42).choice(len(valid), n, replace=False)
        samples.append(valid[idx])

X_sample = np.vstack(samples)
print(f"  Sample: {X_sample.shape[0]:,} pixels × {X_sample.shape[1]} bands")

pca = PCA(n_components=args.n_components, random_state=0)
pca.fit(X_sample)
explained = pca.explained_variance_ratio_.cumsum()
print(f"  Variance explained — PC64: {explained[-1]:.3f}, PC1: {pca.explained_variance_ratio_[0]:.3f}")

# ── Step 2: per-tract masking, PCA transform, and statistics ─────────────────

print(f"Aggregating {tracts_zip.name}...")
tracts = gpd.read_file(f"zip://{tracts_zip}")

STATS = [
    ("MEAN",   lambda x: x.mean(axis=0)),
    ("MEDIAN", lambda x: np.median(x, axis=0)),
    ("MAX",    lambda x: x.max(axis=0)),
    ("MIN",    lambda x: x.min(axis=0)),
    ("STD",    lambda x: x.std(axis=0)),
]

pc_names = [f"PC{i:02d}" for i in range(args.n_components)]
col_names = [f"{pc}_{stat}" for pc in pc_names for stat, _ in STATS]

rows = []
with rasterio.open(cog_path) as src:
    nodata = src.nodata
    tracts_proj = tracts.to_crs(src.crs)

    for _, tract in tqdm(tracts_proj.iterrows(), total=len(tracts_proj), desc="Tracts"):
        try:
            masked, _ = rasterio.mask.mask(
                src, [tract.geometry], crop=True, nodata=nodata, all_touched=False
            )
        except Exception:
            continue

        valid_mask = masked[0] != nodata
        valid = masked[:, valid_mask].T.astype("float32")  # (n_pixels, C)
        if len(valid) == 0:
            continue

        transformed = pca.transform(valid)  # (n_pixels, 64)

        row = {"GEOID": tract["GEOID"], "year": args.year}
        for pc_idx, pc_name in enumerate(pc_names):
            col = transformed[:, pc_idx]
            for stat_name, fn in STATS:
                row[f"{pc_name}_{stat_name}"] = float(fn(col.reshape(-1, 1)).squeeze())
        rows.append(row)

# ── Step 3: write CSV ─────────────────────────────────────────────────────────

df = pd.DataFrame(rows, columns=["GEOID", "year"] + col_names)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} tracts → {out_path}")
