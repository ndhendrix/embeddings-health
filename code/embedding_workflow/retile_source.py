"""Create a lossless tiled copy of a source composite for windowed inference."""
import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def sample_windows(width: int, height: int, size: int = 128) -> list[Window]:
    columns = sorted({0, max(0, (width - size) // 2), max(0, width - size)})
    rows = sorted({0, max(0, (height - size) // 2), max(0, height - size)})
    return [Window(column, row, min(size, width - column), min(size, height - row))
            for row in rows for column in columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    if args.block_size < 16 or args.block_size % 16:
        raise ValueError("block size must be a multiple of 16")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    with rasterio.open(args.input) as source:
        profile = source.profile.copy()
        profile.update(
            tiled=True,
            blockxsize=args.block_size,
            blockysize=args.block_size,
            compress="zstd",
            predictor=3,
            zstd_level=1,
            BIGTIFF="IF_SAFER",
            num_threads="ALL_CPUS",
        )
        with rasterio.open(temporary, "w", **profile) as target:
            target.update_tags(**source.tags())
            for band, description in enumerate(source.descriptions, 1):
                if description:
                    target.set_band_description(band, description)
            for row in range(0, source.height, args.block_size):
                for column in range(0, source.width, args.block_size):
                    window = Window(
                        column,
                        row,
                        min(args.block_size, source.width - column),
                        min(args.block_size, source.height - row),
                    )
                    target.write(source.read(window=window), window=window)
    os.replace(temporary, args.output)

    with rasterio.open(args.input) as source, rasterio.open(args.output) as target:
        if (source.crs, source.transform, source.shape, source.count, source.dtypes) != (
            target.crs, target.transform, target.shape, target.count, target.dtypes
        ):
            raise ValueError("retiled raster schema or geometry differs from source")
        if target.block_shapes[0] != (args.block_size, args.block_size):
            raise ValueError(f"unexpected output block shape: {target.block_shapes[0]}")
        for window in sample_windows(source.width, source.height):
            expected = source.read(window=window)
            actual = target.read(window=window)
            if not np.array_equal(np.isfinite(expected), np.isfinite(actual)):
                raise ValueError(f"finite mask differs in verification window {window}")
            valid = np.isfinite(expected)
            if valid.any() and not np.array_equal(expected[valid], actual[valid]):
                raise ValueError(f"values differ in verification window {window}")
    print(f"wrote lossless tiled source {args.output}")


if __name__ == "__main__":
    main()
