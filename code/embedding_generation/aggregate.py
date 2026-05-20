"""
Aggregate embedding COG to census tract-level statistics.

Reads a multi-band embedding GeoTIFF and a census tract shapefile, then for each
tract computes mean / median / max / min / std across all embedding pixels within
the tract boundary.  Output CSV matches the schema of alphaearth_embeddings.csv.

Usage:
  python aggregate.py --embedding outputs/embeddings/olmoearth_RI_2022.tif \\
                      --tracts data/census_tracts_2020.gpkg \\
                      --output data/olmoearth_embeddings.csv \\
                      --year 2022 --model olmoearth
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from tqdm import tqdm


def aggregate_tract(src: rasterio.DatasetReader, geom) -> dict | None:
    """Extract and summarise embedding pixels within geom. Returns None if no valid pixels."""
    try:
        masked, _ = rasterio.mask.mask(src, [geom], crop=True, nodata=np.nan, all_touched=False)
    except Exception:
        return None

    # masked: (C, H, W); flatten to (C, N_valid)
    flat = masked.reshape(masked.shape[0], -1)
    valid_mask = ~np.isnan(flat).all(axis=0)
    flat = flat[:, valid_mask]
    if flat.shape[1] == 0:
        return None

    stats = {}
    for stat, fn in [
        ("MEAN",    lambda x: np.nanmean(x, axis=1)),
        ("MEDIAN",  lambda x: np.nanmedian(x, axis=1)),
        ("MAXIMUM", lambda x: np.nanmax(x, axis=1)),
        ("MINIMUM", lambda x: np.nanmin(x, axis=1)),
        ("STD",     lambda x: np.nanstd(x, axis=1)),
    ]:
        vals = fn(flat)
        stats[stat] = vals
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", type=Path, required=True,
                        help="Embedding COG GeoTIFF produced by embed.py.")
    parser.add_argument("--tracts", type=Path, required=True,
                        help="Census tract shapefile/GeoPackage. Must have a 'GEOID' column.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], default="olmoearth",
                        help="Used to set column prefix: OE or PR.")
    args = parser.parse_args()

    prefix = "OE" if args.model == "olmoearth" else "PR"

    print(f"Loading tracts from {args.tracts}…")
    tracts = gpd.read_file(args.tracts)

    with rasterio.open(args.embedding) as src:
        # Re-project tracts to match embedding CRS
        tracts = tracts.to_crs(src.crs)
        n_bands = src.count

        band_names = [src.tags(i + 1).get("name", f"{prefix}{i:02d}") for i in range(n_bands)]
        print(f"Embedding bands: {n_bands}  CRS: {src.crs}  shape: {src.height}×{src.width}")

        rows = []
        for _, tract in tqdm(tracts.iterrows(), total=len(tracts), desc="Aggregating tracts"):
            stats = aggregate_tract(src, tract.geometry)
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
