#!/usr/bin/env python3
"""Regenerate a local Prithvi crop with overlap-weighted inference.

The statewide Prithvi rasters were assembled from non-overlapping 224 px
chips. Because transformer embeddings depend on chip context, their 14 x 14
token grids can have visible seams. This script evaluates a local window on a
half-chip stride and blends duplicate tokens with center-weighted averaging.
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
    "/scratch/users/nhendrix/embeddings-health/figures/figure1_prithvi_overlap"
)


def _token_starts(first: int, stop: int, limit: int, stride: int) -> list[int]:
    """Return aligned chip starts whose token intervals overlap [first, stop)."""
    chip_tokens = PRITHVI_CHIP_PX // PRITHVI_PATCH_SIZE
    lo = math.floor((first - chip_tokens + 1) / stride) * stride
    hi = math.floor((stop - 1) / stride) * stride
    return [
        start
        for start in range(lo, hi + 1, stride)
        if start >= 0
        and start + chip_tokens <= limit
        and start < stop
        and start + chip_tokens > first
    ]


def _blend_weights() -> np.ndarray:
    """Return a nonzero center-weighted 14 x 14 blend."""
    chip_tokens = PRITHVI_CHIP_PX // PRITHVI_PATCH_SIZE
    one_d = np.hanning(chip_tokens + 2)[1:-1].astype("float32")
    one_d /= one_d.max()
    return np.outer(one_d, one_d).astype("float32")


def infer_overlap_crop(
    input_paths: list[Path],
    variant: str,
    lat: float,
    lon: float,
    box_km: float,
    batch_size: int,
) -> tuple[np.ndarray, Affine, object, dict]:
    """Return overlap-blended raw embeddings and local georeferencing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Overlap regeneration requires a CUDA GPU")

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
        full_rows = math.ceil(ref.height / PRITHVI_PATCH_SIZE)
        full_cols = math.ceil(ref.width / PRITHVI_PATCH_SIZE)
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
        stride_tokens = chip_tokens // 2
        row_starts = _token_starts(row0, row1, full_rows, stride_tokens)
        col_starts = _token_starts(col0, col1, full_cols, stride_tokens)
        tasks = [(row, col) for row in row_starts for col in col_starts]
        if not tasks:
            raise ValueError("No inference chips cover the requested crop")

        accum = np.zeros((embed_dim, height, width), dtype="float32")
        weight_sum = np.zeros((height, width), dtype="float32")
        weights = _blend_weights()

        print(
            f"{variant}: target={height}x{width} tokens; "
            f"overlap chips={len(tasks)} ({len(row_starts)}x{len(col_starts)}); "
            f"device={device}",
            flush=True,
        )

        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            frames = []
            for src in srcs:
                raw = np.stack(
                    [
                        src.read(
                            window=Window(
                                col * PRITHVI_PATCH_SIZE,
                                row * PRITHVI_PATCH_SIZE,
                                PRITHVI_CHIP_PX,
                                PRITHVI_CHIP_PX,
                            )
                        )
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

            for embedding, (global_row, global_col) in zip(spatial, batch):
                ir0 = max(global_row, row0)
                ic0 = max(global_col, col0)
                ir1 = min(global_row + chip_tokens, row1)
                ic1 = min(global_col + chip_tokens, col1)
                sr0, sc0 = ir0 - global_row, ic0 - global_col
                sr1, sc1 = ir1 - global_row, ic1 - global_col
                dr0, dc0 = ir0 - row0, ic0 - col0
                dr1, dc1 = ir1 - row0, ic1 - col0
                local_weights = weights[sr0:sr1, sc0:sc1]
                accum[:, dr0:dr1, dc0:dc1] += (
                    embedding[sr0:sr1, sc0:sc1, :].transpose(2, 0, 1)
                    * local_weights[None, :, :]
                )
                weight_sum[dr0:dr1, dc0:dc1] += local_weights

            print(
                f"  processed {min(batch_start + batch_size, len(tasks))}/"
                f"{len(tasks)} chips",
                flush=True,
            )

        if np.any(weight_sum == 0):
            missing = int(np.count_nonzero(weight_sum == 0))
            raise RuntimeError(f"Overlap blend left {missing} target cells uncovered")

        blended = accum / weight_sum[None, :, :]
        local_transform = rasterio.windows.transform(target, output_transform)
        metadata = {
            "variant": variant,
            "embedding_dimensions": embed_dim,
            "model_family": model_family,
            "num_frames": num_frames,
            "input_paths": [str(path) for path in paths],
            "chip_pixels": PRITHVI_CHIP_PX,
            "patch_pixels": PRITHVI_PATCH_SIZE,
            "overlap_stride_pixels": stride_tokens * PRITHVI_PATCH_SIZE,
            "overlap_chips": len(tasks),
            "target_shape": [height, width],
            "weight_min": float(weight_sum.min()),
            "weight_max": float(weight_sum.max()),
        }
        return blended, local_transform, ref.crs, metadata
    finally:
        for src in srcs:
            src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("tiny", "300M-TL"), required=True)
    parser.add_argument("--state", default="CA")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument("--box-km", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    inputs = [
        args.data_root / f"s2_{season}_{args.state}_{args.year}_prithvi.tif"
        for season in ("spring", "summer", "fall")
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))

    batch_size = args.batch_size or (8 if args.variant == "tiny" else 1)
    embeddings, transform, crs, metadata = infer_overlap_crop(
        inputs, args.variant, args.lat, args.lon, args.box_km, batch_size
    )

    safe_variant = args.variant.replace("-", "_").lower()
    output = args.output_root / f"prithvi_{safe_variant}_palo_alto_overlap.tif"
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
    print(f"Saved overlap-blended embeddings -> {output}", flush=True)


if __name__ == "__main__":
    main()
