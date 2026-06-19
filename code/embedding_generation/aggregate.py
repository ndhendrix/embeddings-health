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
from tqdm import tqdm


def aggregate_tract(src: rasterio.DatasetReader, geom, pca=None) -> dict | None:
    """Extract pixels within geom, optionally apply national PCA, return per-dim stats.

    Returns None if no valid pixels intersect the tract.
    pca: a fitted sklearn PCA object, or None to skip transformation.
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
        # flat is (D, N); pca.transform expects (N, D)
        flat = pca.transform(flat.T).astype("float32").T  # → (n_components, N)

    stats = {}
    for stat, fn in [
        ("MEAN",    lambda x: np.nanmean(x, axis=1)),
        ("MEDIAN",  lambda x: np.nanmedian(x, axis=1)),
        ("MAXIMUM", lambda x: np.nanmax(x, axis=1)),
        ("MINIMUM", lambda x: np.nanmin(x, axis=1)),
        ("STD",     lambda x: np.nanstd(x, axis=1)),
    ]:
        stats[stat] = fn(flat)
    return stats


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

        rows = []
        for _, tract in tqdm(tracts.iterrows(), total=len(tracts), desc="Aggregating tracts"):
            stats = aggregate_tract(src, tract.geometry, pca=pca)
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
