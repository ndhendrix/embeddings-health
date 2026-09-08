"""Aggregate a large embedding raster once per physical spatial block.

The legacy tract-by-tract aggregator repeatedly decompresses the same
pixel-interleaved GeoTIFF blocks.  This module reads each spatial block once,
computes exact sufficient statistics for every tract represented in that
block, and writes a partial compatible with ``aggregate_tiles.py reduce``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box


def aggregate_block(
    embedding: Path,
    tracts_path: Path,
    output: Path,
    block_row: int,
    block_col: int,
    expected_dim: int | None = None,
    band_chunk: int = 64,
) -> None:
    """Write sufficient statistics for one native raster block."""
    tracts = gpd.read_file(tracts_path)

    with rasterio.open(embedding) as src:
        if expected_dim is not None and src.count != expected_dim:
            raise ValueError(
                f"expected {expected_dim} bands, found {src.count}: {embedding}"
            )
        block_height, block_width = src.block_shapes[0]
        if any(shape != (block_height, block_width) for shape in src.block_shapes):
            raise ValueError("band-dependent block shapes are not supported")

        row_off = block_row * block_height
        col_off = block_col * block_width
        if row_off >= src.height or col_off >= src.width:
            raise ValueError(
                f"block ({block_row}, {block_col}) is outside "
                f"the {src.height}x{src.width} raster"
            )
        height = min(block_height, src.height - row_off)
        width = min(block_width, src.width - col_off)
        window = Window(col_off, row_off, width, height)
        transform = rasterio.windows.transform(window, src.transform)
        bounds = rasterio.windows.bounds(window, src.transform)

        tracts = tracts.to_crs(src.crs)
        tracts = tracts[tracts.geometry.intersects(box(*bounds))].copy()
        tracts = tracts.sort_values("GEOID").reset_index(drop=True)

        if tracts.empty:
            _write_partial(output, [], src.count)
            return

        shapes = [
            (geometry, local_id)
            for local_id, geometry in enumerate(tracts.geometry, start=1)
        ]
        labels = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="int32",
        ).reshape(-1)

        # The source is pixel-interleaved, so one aligned read decompresses the
        # physical block once. Statistics are then calculated in small band
        # chunks to keep temporary allocations bounded.
        data = src.read(window=window, out_dtype="float32")

    first_valid = np.isfinite(data[0].reshape(-1))
    last_valid = np.isfinite(data[-1].reshape(-1))
    if not np.array_equal(first_valid, last_valid):
        raise ValueError("band-dependent nodata is not supported")

    selected = (labels > 0) & first_valid
    if not selected.any():
        _write_partial(output, [], data.shape[0])
        return

    selected_labels = labels[selected]
    order = np.argsort(selected_labels, kind="stable")
    sorted_labels = selected_labels[order]
    unique_labels, starts, counts = np.unique(
        sorted_labels, return_index=True, return_counts=True
    )

    n_groups = len(unique_labels)
    n_bands = data.shape[0]
    sums = np.empty((n_groups, n_bands), dtype=np.float64)
    sumsqs = np.empty((n_groups, n_bands), dtype=np.float64)
    minima = np.empty((n_groups, n_bands), dtype=np.float32)
    maxima = np.empty((n_groups, n_bands), dtype=np.float32)

    flat = data.reshape(n_bands, -1)
    for band_start in range(0, n_bands, band_chunk):
        band_end = min(n_bands, band_start + band_chunk)
        values = flat[band_start:band_end, selected][:, order]
        if not np.isfinite(values).all():
            raise ValueError("band-dependent nodata is not supported")
        sl = slice(band_start, band_end)
        sums[:, sl] = np.add.reduceat(
            values, starts, axis=1, dtype=np.float64
        ).T
        values64 = values.astype(np.float64)
        sumsqs[:, sl] = np.add.reduceat(
            np.square(values64), starts, axis=1
        ).T
        minima[:, sl] = np.minimum.reduceat(values, starts, axis=1).T
        maxima[:, sl] = np.maximum.reduceat(values, starts, axis=1).T

    geoids = [str(tracts.iloc[label - 1]["GEOID"]) for label in unique_labels]
    _write_partial(
        output,
        geoids,
        n_bands,
        counts=counts,
        sums=sums,
        sumsqs=sumsqs,
        minima=minima,
        maxima=maxima,
    )


def _write_partial(
    output: Path,
    geoids: list[str],
    dimensions: int,
    *,
    counts: np.ndarray | None = None,
    sums: np.ndarray | None = None,
    sumsqs: np.ndarray | None = None,
    minima: np.ndarray | None = None,
    maxima: np.ndarray | None = None,
) -> None:
    n_rows = len(geoids)
    counts = np.empty(n_rows, dtype=np.int64) if counts is None else counts
    sums = np.empty((n_rows, dimensions), dtype=np.float64) if sums is None else sums
    sumsqs = (
        np.empty((n_rows, dimensions), dtype=np.float64)
        if sumsqs is None
        else sumsqs
    )
    minima = (
        np.empty((n_rows, dimensions), dtype=np.float32)
        if minima is None
        else minima
    )
    maxima = (
        np.empty((n_rows, dimensions), dtype=np.float32)
        if maxima is None
        else maxima
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            GEOID=np.asarray(geoids),
            count=np.asarray(counts, dtype=np.int64),
            sum=sums,
            sumsq=sumsqs,
            min=minima,
            max=maxima,
        )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--tracts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-row", type=int, required=True)
    parser.add_argument("--block-col", type=int, required=True)
    parser.add_argument("--expected-dim", type=int)
    parser.add_argument("--band-chunk", type=int, default=64)
    args = parser.parse_args()
    aggregate_block(
        args.embedding,
        args.tracts,
        args.output,
        args.block_row,
        args.block_col,
        args.expected_dim,
        args.band_chunk,
    )


if __name__ == "__main__":
    main()
