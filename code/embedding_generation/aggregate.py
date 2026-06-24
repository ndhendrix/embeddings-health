"""
Aggregate embedding COG to census tract-level statistics.

Reads a multi-band embedding GeoTIFF and a census tract shapefile, then for each
tract computes mean / median / max / min / std across all embedding pixels within
the tract boundary.  Output CSV matches the schema of alphaearth_embeddings.csv.

When --pca-model is supplied the raw embedding pixels are transformed into the
national PCA space before statistics are computed, producing nationally comparable
PC00–PC63 dimensions rather than raw model-specific dimensions.

Usage:
  # With national PCA (recommended for cross-state comparability):
  python aggregate.py --embedding prithvi_tiny_RI_2022.tif \\
                      --tracts data/census_tracts_2020.gpkg \\
                      --output data/prithvi_tiny_RI_2022_tracts.csv \\
                      --pca-model national_pca/prithvi_tiny_national_pca.pkl \\
                      --year 2022 --model prithvi

  # Without PCA (raw embedding dimensions):
  python aggregate.py --embedding prithvi_tiny_RI_2022.tif \\
                      --tracts data/census_tracts_2020.gpkg \\
                      --output data/prithvi_tiny_RI_2022_tracts.csv \\
                      --year 2022 --model prithvi
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.windows
from rasterio.features import geometry_mask
from tqdm import tqdm

_STATS = [
    ("MEAN",    lambda x: np.nanmean(x, axis=1)),
    ("MEDIAN",  lambda x: np.nanmedian(x, axis=1)),
    ("MAXIMUM", lambda x: np.nanmax(x, axis=1)),
    ("MINIMUM", lambda x: np.nanmin(x, axis=1)),
    ("STD",     lambda x: np.nanstd(x, axis=1)),
]


def _precompute_pca_raster(src: rasterio.DatasetReader, pca) -> np.ndarray:
    """Load all bands in small batches, apply the PCA projection, return (n_comp, H, W).

    Streams 64 bands at a time so peak extra memory is ~2× the 64-band output,
    regardless of how many raw bands the file has. NaN pixels (NoData border) are
    preserved: any pixel where all raw bands are NaN stays NaN in the output.
    """
    H, W = src.height, src.width
    n_comp = pca.n_components_
    mean = pca.mean_.astype(np.float32)            # (n_features,)
    components = pca.components_.astype(np.float32)  # (n_comp, n_features)

    out = np.zeros((n_comp, H, W), dtype=np.float32)
    nodata_mask = None  # (H, W) bool, True = nodata

    BATCH = 64
    n_bands = src.count
    for b0 in range(0, n_bands, BATCH):
        b1 = min(b0 + BATCH, n_bands)
        data = src.read(list(range(b0 + 1, b1 + 1))).astype(np.float32)  # (batch, H, W)

        if nodata_mask is None:
            # Identify NoData pixels from the first batch (all-NaN across bands)
            nodata_mask = np.isnan(data).all(axis=0)  # (H, W)

        data -= mean[b0:b1, None, None]
        # out += components[:, b0:b1] @ data.reshape(batch, H*W)
        out += (components[:, b0:b1] @ data.reshape(b1 - b0, H * W)).reshape(n_comp, H, W)

    if nodata_mask is not None and nodata_mask.any():
        out[:, nodata_mask] = np.nan

    return out


def _fast_aggregate(pca_raster: np.ndarray, geom, transform) -> dict | None:
    """Per-tract zonal stats directly from an in-memory (n_comp, H, W) array.

    No disk I/O — all operations are numpy.  Uses the same all_touched=False
    rasterisation rule as the original rasterio.mask path.
    """
    H, W = pca_raster.shape[1:]

    window = rasterio.windows.from_bounds(*geom.bounds, transform=transform)
    row_start = max(0, int(np.floor(window.row_off)))
    col_start = max(0, int(np.floor(window.col_off)))
    row_end   = min(H, int(np.ceil(window.row_off + window.height)))
    col_end   = min(W, int(np.ceil(window.col_off + window.width)))

    if row_start >= row_end or col_start >= col_end:
        return None

    crop = pca_raster[:, row_start:row_end, col_start:col_end]  # (C, h, w)

    win = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
    win_transform = rasterio.windows.transform(win, transform)
    inside = geometry_mask([geom], transform=win_transform, invert=True,
                           out_shape=(row_end - row_start, col_end - col_start))

    flat = crop[:, inside]               # (C, N)
    valid = ~np.isnan(flat).any(axis=0)  # exclude NoData border pixels
    flat = flat[:, valid]

    if flat.shape[1] == 0:
        return None

    return {stat: fn(flat) for stat, fn in _STATS}


def aggregate_tract(src: rasterio.DatasetReader, geom, pca=None) -> dict | None:
    """Per-tract aggregation reading directly from the open raster (no-PCA fallback path).

    When pca is supplied, prefer _precompute_pca_raster + _fast_aggregate instead;
    this path reads all raw bands from disk for every tract and is slow on large files.
    """
    try:
        masked, _ = rasterio.mask.mask(src, [geom], crop=True, nodata=np.nan, all_touched=False)
    except Exception:
        return None

    # masked: (D, H, W) → flatten to (D, N_valid)
    flat = masked.reshape(masked.shape[0], -1)
    valid_mask = ~np.isnan(flat).all(axis=0)
    flat = flat[:, valid_mask]
    if flat.shape[1] == 0:
        return None

    if pca is not None:
        flat = pca.transform(flat.T).astype("float32").T  # → (n_components, N)

    return {stat: fn(flat) for stat, fn in _STATS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", type=Path, required=True,
                        help="Embedding COG GeoTIFF produced by embed.py.")
    parser.add_argument("--tracts", type=Path, required=True,
                        help="Census tract shapefile/GeoPackage. Must have a 'GEOID' column.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path.")
    parser.add_argument("--pca-model", type=Path, default=None,
                        help="National PCA .pkl fitted by fit_national_pca.py. "
                             "When provided, raw pixels are projected into the national "
                             "PCA space before computing tract statistics, yielding "
                             "nationally comparable PC00–PC63 columns.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], default="olmoearth",
                        help="Used to set fallback column prefix when no PCA model is provided.")
    args = parser.parse_args()

    prefix = "OE" if args.model == "olmoearth" else "PR"

    pca = None
    if args.pca_model is not None:
        if not args.pca_model.exists():
            raise FileNotFoundError(
                f"PCA model not found: {args.pca_model}\n"
                "Run fit_national_pca.sbatch first, or omit --pca-model to aggregate raw dims."
            )
        with open(args.pca_model, "rb") as f:
            pca = pickle.load(f)
        print(f"Loaded national PCA: {pca.n_components_} components from {args.pca_model}")

    print(f"Loading tracts from {args.tracts}…")
    tracts = gpd.read_file(args.tracts)

    with rasterio.open(args.embedding) as src:
        tracts = tracts.to_crs(src.crs)
        n_bands = src.count

        if pca is not None:
            band_names = [f"PC{i:02d}" for i in range(pca.n_components_)]
        else:
            band_names = [src.tags(i + 1).get("name", f"{prefix}{i:02d}") for i in range(n_bands)]

        print(f"Embedding bands: {n_bands}  CRS: {src.crs}  shape: {src.height}×{src.width}")
        if pca is not None:
            print(f"Output dims: {pca.n_components_} (national PCA space)")

        if pca is not None:
            print(f"Pre-computing PCA projection ({src.count} bands → {pca.n_components_} "
                  f"components, streaming {min(64, src.count)} bands at a time)…")
            pca_raster = _precompute_pca_raster(src, pca)
            src_transform = src.transform

        rows = []
        for _, tract in tqdm(tracts.iterrows(), total=len(tracts), desc="Aggregating tracts"):
            if pca is not None:
                stats = _fast_aggregate(pca_raster, tract.geometry, src_transform)
            else:
                stats = aggregate_tract(src, tract.geometry, pca=None)
            if stats is None:
                continue

            row = {"GEOID": tract["GEOID"], "year": args.year}
            for stat_name, values in stats.items():
                for band_idx, val in enumerate(values):
                    col = f"{band_names[band_idx]}_{stat_name}"
                    row[col] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} tracts → {args.output}")


if __name__ == "__main__":
    main()
