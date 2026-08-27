#!/usr/bin/env python3
"""Build a local Prithvi-compatible input from an OlmoEarth S2 composite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import from_bounds


DEFAULT_SOURCE = Path(
    "/scratch/users/nhendrix/embeddings-health/"
    "olmoearth_composites_tiled512/s2_annual_CA_2022_olmoearth.tif"
)
DEFAULT_OUTPUT = Path(
    "/scratch/users/nhendrix/embeddings-health/figures/"
    "figure1_prithvi_reconstructed/palo_alto_annual_30m.tif"
)
# OlmoEarth order: B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09.
# Prithvi order:   B02 B03 B04 B8A B11 B12.
PRITHVI_FROM_OLMOEARTH = (1, 2, 3, 8, 9, 10)
BAND_NAMES = ("B02", "B03", "B04", "B8A", "B11", "B12")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument(
        "--extent-km",
        type=float,
        default=36.0,
        help="Side length including inference context (default: 36 km)",
    )
    args = parser.parse_args()

    with rasterio.open(args.source) as src:
        x, y = Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        ).transform(args.lon, args.lat)
        target_grid = Affine(
            30.0,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            -30.0,
            src.transform.f,
        )
        half_m = args.extent_km * 500.0
        window = from_bounds(
            x - half_m,
            y - half_m,
            x + half_m,
            y + half_m,
            target_grid,
        ).round_offsets().round_lengths()
        transform = rasterio.windows.transform(window, target_grid)
        height, width = int(window.height), int(window.width)
        data = np.full((6, height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, list(PRITHVI_FROM_OLMOEARTH)),
            destination=data,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=4,
            init_dest_nodata=True,
        )
        crs = src.crs

    valid = np.isfinite(data).all(axis=0)
    if not valid.all():
        raise RuntimeError(
            f"Reconstructed input contains {int(np.count_nonzero(~valid))} "
            "invalid pixels"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 6,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "pixel",
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(
            source_composite=str(args.source),
            source_band_indexes="1,2,3,8,9,10",
            temporal_input="2022 annual composite repeated for all model frames",
            resampling="average 10m to 30m",
            purpose="Figure 1 Palo Alto Prithvi reconstruction",
        )

    metadata = {
        "source": str(args.source),
        "output": str(args.output),
        "center_lat": args.lat,
        "center_lon": args.lon,
        "extent_km": args.extent_km,
        "source_band_indexes": list(PRITHVI_FROM_OLMOEARTH),
        "band_names": list(BAND_NAMES),
        "resampling": "average",
        "source_resolution_m": 10,
        "output_resolution_m": 30,
        "temporal_input": "2022 annual composite repeated for all model frames",
        "shape": list(data.shape),
        "valid_pixel_fraction": float(valid.mean()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved reconstructed Prithvi input -> {args.output}")
    print(f"Shape: {data.shape}; valid pixels: {valid.mean():.1%}")


if __name__ == "__main__":
    main()
