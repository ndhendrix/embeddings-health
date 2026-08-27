#!/usr/bin/env python3
"""Regenerate a local Prithvi crop using a fixed central token position.

Prithvi's statewide rasters were assembled from non-overlapping 224 px chips.
Their final-layer tokens retain a strong within-chip positional signal, which
becomes visible as 14 x 14 seams in a local PCA rendering. For every output
cell, this script centers the corresponding 16 x 16 source patch inside its
own 224 x 224 context window and retains the same central model token. The
result has consistent context and positional phase across the entire crop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "code" / "embedding_generation"
sys.path.insert(0, str(EMBED_DIR))

from embed import (  # noqa: E402
    PRITHVI_CHIP_PX,
    PRITHVI_MEANS,
    PRITHVI_PATCH_SIZE,
    PRITHVI_STDS,
    load_prithvi,
    run_prithvi_batch,
)
from utils.cog_writer import write_cog  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "/scratch/users/nhendrix/embeddings-health/data/prithvi_data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/nhendrix/embeddings-health/figures/figure1_prithvi_centered"
)


def infer_centered_crop(
    input_paths: list[Path],
    variant: str,
    lat: float,
    lon: float,
    box_km: float,
    batch_size: int,
) -> tuple[np.ndarray, Affine, object, dict]:
    """Return raw embeddings evaluated at one fixed token position."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Centered Prithvi regeneration requires a CUDA GPU")

    model, embed_dim, num_frames, model_family = load_prithvi(variant)
    model = model.to(device)

    paths = list(input_paths)
    if len(paths) < num_frames:
        paths.extend([paths[-1]] * (num_frames - len(paths)))
    elif len(paths) > num_frames:
        paths = paths[:num_frames]

    srcs = [rasterio.open(path) for path in paths]
    try:
        ref = srcs[0]
        for src in srcs[1:]:
            if src.crs != ref.crs or src.transform != ref.transform:
                raise ValueError("All seasonal inputs must share a CRS and transform")

        x, y = Transformer.from_crs(
            "EPSG:4326", ref.crs, always_xy=True
        ).transform(lon, lat)
        half_m = box_km * 500.0
        output_transform = Affine(
            ref.transform.a * PRITHVI_PATCH_SIZE,
            ref.transform.b,
            ref.transform.c,
            ref.transform.d,
            ref.transform.e * PRITHVI_PATCH_SIZE,
            ref.transform.f,
        )
        target = from_bounds(
            x - half_m,
            y - half_m,
            x + half_m,
            y + half_m,
            output_transform,
        ).round_offsets().round_lengths()
        row0, col0 = int(target.row_off), int(target.col_off)
        height, width = int(target.height), int(target.width)
        row1, col1 = row0 + height, col0 + width

        chip_tokens = PRITHVI_CHIP_PX // PRITHVI_PATCH_SIZE
        center_token = chip_tokens // 2
        min_chip_row = row0 - center_token
        min_chip_col = col0 - center_token
        max_chip_row = row1 - 1 - center_token
        max_chip_col = col1 - 1 - center_token
        if min_chip_row < 0 or min_chip_col < 0:
            raise ValueError("Requested crop is too close to the raster edge")
        if (
            (max_chip_row + chip_tokens) * PRITHVI_PATCH_SIZE > ref.height
            or (max_chip_col + chip_tokens) * PRITHVI_PATCH_SIZE > ref.width
        ):
            raise ValueError("Requested crop plus context exceeds the raster extent")

        local_window = Window(
            min_chip_col * PRITHVI_PATCH_SIZE,
            min_chip_row * PRITHVI_PATCH_SIZE,
            (max_chip_col - min_chip_col + chip_tokens) * PRITHVI_PATCH_SIZE,
            (max_chip_row - min_chip_row + chip_tokens) * PRITHVI_PATCH_SIZE,
        )
        frame_arrays = [src.read(window=local_window) for src in srcs]
        tasks = [(row, col) for row in range(row0, row1) for col in range(col0, col1)]
        output = np.empty((embed_dim, height, width), dtype="float32")

        print(
            f"{variant}: target={height}x{width} tokens; "
            f"centered chips={len(tasks)}; fixed token="
            f"({center_token},{center_token}); device={device}",
            flush=True,
        )

        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            frames = []
            for frame in frame_arrays:
                raw = np.stack(
                    [
                        frame[
                            :,
                            (row - center_token - min_chip_row)
                            * PRITHVI_PATCH_SIZE :
                            (row - center_token - min_chip_row)
                            * PRITHVI_PATCH_SIZE
                            + PRITHVI_CHIP_PX,
                            (col - center_token - min_chip_col)
                            * PRITHVI_PATCH_SIZE :
                            (col - center_token - min_chip_col)
                            * PRITHVI_PATCH_SIZE
                            + PRITHVI_CHIP_PX,
                        ]
                        for row, col in batch
                    ],
                    axis=0,
                )
                frames.append(
                    (raw - PRITHVI_MEANS[:, None, None])
                    / PRITHVI_STDS[:, None, None]
                )
            chips = np.stack(frames, axis=1)
            spatial = run_prithvi_batch(
                model, chips, device, model_family=model_family
            )

            for embedding, (row, col) in zip(spatial, batch):
                output[:, row - row0, col - col0] = embedding[
                    center_token, center_token, :
                ]

            completed = min(batch_start + batch_size, len(tasks))
            if completed == len(tasks) or completed % max(100, batch_size) == 0:
                print(f"  processed {completed}/{len(tasks)} chips", flush=True)

        local_transform = rasterio.windows.transform(target, output_transform)
        metadata = {
            "variant": variant,
            "embedding_dimensions": embed_dim,
            "model_family": model_family,
            "num_frames": num_frames,
            "input_paths": [str(path) for path in paths],
            "method": "fixed-central-token context",
            "chip_pixels": PRITHVI_CHIP_PX,
            "patch_pixels": PRITHVI_PATCH_SIZE,
            "fixed_token_index": [center_token, center_token],
            "context_chips": len(tasks),
            "target_shape": [height, width],
        }
        return output, local_transform, ref.crs, metadata
    finally:
        for src in srcs:
            src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("tiny", "300M-TL"), required=True)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--state", default="CA")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument("--box-km", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    inputs = list(args.input) if args.input else [
        args.data_root / f"s2_{season}_{args.state}_{args.year}_prithvi.tif"
        for season in ("spring", "summer", "fall")
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))

    batch_size = args.batch_size or (8 if args.variant == "tiny" else 1)
    embeddings, transform, crs, metadata = infer_centered_crop(
        inputs, args.variant, args.lat, args.lon, args.box_km, batch_size
    )

    safe_variant = args.variant.replace("-", "_").lower()
    output = args.output_root / f"prithvi_{safe_variant}_palo_alto_centered.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    band_names = [f"PR{i:04d}" for i in range(embeddings.shape[0])]
    write_cog(
        embeddings,
        transform,
        crs,
        output,
        band_names=band_names,
        overviews=False,
    )
    output.with_suffix(".json").write_text(
        json.dumps({**metadata, "output": str(output)}, indent=2) + "\n"
    )
    print(f"Saved centered embeddings -> {output}", flush=True)


if __name__ == "__main__":
    main()
