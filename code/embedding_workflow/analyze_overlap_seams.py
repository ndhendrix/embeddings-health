"""Measure embedding discontinuities at retained-chip boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def _summary(parts: list[np.ndarray]) -> dict[str, float | int | None]:
    values = np.concatenate(parts) if parts else np.empty(0, dtype="float32")
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _collect(parts: dict[str, list[np.ndarray]], values: np.ndarray, phases: np.ndarray) -> None:
    seam = phases == 0
    parts["seam"].append(values[:, seam].ravel())
    parts["interior"].append(values[:, ~seam].ravel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band-chunk", type=int, default=16)
    args = parser.parse_args()

    horizontal = {"seam": [], "interior": []}
    vertical = {"seam": [], "interior": []}
    with rasterio.open(args.raster) as src:
        tags = src.tags()
        retained = int(tags["retained_center_pixels"])
        patch = int(tags["patch_pixels"])
        if retained % patch:
            raise ValueError("retained center is not divisible by patch size")
        keep_tokens = retained // patch

        for _, window in src.block_windows(1):
            height, width = int(window.height), int(window.width)
            read_height = min(height + 1, src.height - int(window.row_off))
            read_width = min(width + 1, src.width - int(window.col_off))
            read_window = Window(window.col_off, window.row_off, read_width, read_height)
            h_width = min(width, src.width - int(window.col_off) - 1)
            v_height = min(height, src.height - int(window.row_off) - 1)
            h_sq = np.zeros((height, max(0, h_width)), dtype="float64")
            v_sq = np.zeros((max(0, v_height), width), dtype="float64")
            h_valid = np.ones(h_sq.shape, dtype=bool)
            v_valid = np.ones(v_sq.shape, dtype=bool)

            for start in range(1, src.count + 1, args.band_chunk):
                indexes = list(range(start, min(src.count + 1, start + args.band_chunk)))
                data = src.read(indexes, window=read_window)
                if h_width:
                    left = data[:, :height, :h_width]
                    right = data[:, :height, 1:h_width + 1]
                    valid = np.isfinite(left) & np.isfinite(right)
                    h_valid &= valid.all(axis=0)
                    difference = np.where(valid, right - left, 0.0).astype("float64")
                    h_sq += np.square(difference).sum(axis=0)
                if v_height:
                    top = data[:, :v_height, :width]
                    bottom = data[:, 1:v_height + 1, :width]
                    valid = np.isfinite(top) & np.isfinite(bottom)
                    v_valid &= valid.all(axis=0)
                    difference = np.where(valid, bottom - top, 0.0).astype("float64")
                    v_sq += np.square(difference).sum(axis=0)

            if h_width:
                h_rms = np.sqrt(h_sq / src.count).astype("float32")
                h_rms[~h_valid] = np.nan
                phases = (np.arange(int(window.col_off), int(window.col_off) + h_width) + 1) % keep_tokens
                _collect(horizontal, h_rms, phases)
            if v_height:
                v_rms = np.sqrt(v_sq / src.count).astype("float32")
                v_rms[~v_valid] = np.nan
                phases = (np.arange(int(window.row_off), int(window.row_off) + v_height) + 1) % keep_tokens
                _collect(vertical, v_rms.T, phases)

        result = {
            "raster": str(args.raster),
            "shape": [src.count, src.height, src.width],
            "keep_tokens": keep_tokens,
            "metric": "per-pair root-mean-square embedding difference",
            "horizontal": {key: _summary(value) for key, value in horizontal.items()},
            "vertical": {key: _summary(value) for key, value in vertical.items()},
            "tags": tags,
        }
    for direction in ("horizontal", "vertical"):
        seam = result[direction]["seam"]["mean"]
        interior = result[direction]["interior"]["mean"]
        result[direction]["seam_to_interior_mean_ratio"] = (
            None if seam is None or not interior else seam / interior
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
