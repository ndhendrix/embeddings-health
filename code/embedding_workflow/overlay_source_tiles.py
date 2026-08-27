"""Overlay selected source-composite tile TIFs onto an existing merged composite."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def overlay_tile(base_path: Path, patch_path: Path) -> None:
    with rasterio.open(patch_path) as patch:
        bounds = patch.bounds
        patch_data = patch.read()
        with rasterio.open(base_path, "r+") as base:
            if base.count != patch.count:
                raise ValueError(f"band count mismatch: {base_path} vs {patch_path}")
            if str(base.crs) != str(patch.crs):
                raise ValueError(f"CRS mismatch: {base.crs} vs {patch.crs}")
            window = from_bounds(*bounds, transform=base.transform)
            window = window.round_offsets().round_lengths()
            if int(window.width) != patch.width or int(window.height) != patch.height:
                patch_data = patch.read(out_shape=(patch.count, int(window.height), int(window.width)))
            base_data = base.read(window=window)
            merged = np.where(np.isfinite(patch_data), patch_data, base_data)
            base.write(merged, window=window)
            base.update_tags(patched_source_tiles="finite_only")
    print(f"patched {base_path.name} with {patch_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--tiles", nargs="+", type=int, required=True)
    args = parser.parse_args()
    for tile in args.tiles:
        patch = args.patch_dir / f"s2_annual_{args.state}_{args.year}_olmoearth_tile{tile:03d}.tif"
        if not patch.is_file() or patch.stat().st_size == 0:
            raise FileNotFoundError(patch)
        overlay_tile(args.base, patch)


if __name__ == "__main__":
    main()
