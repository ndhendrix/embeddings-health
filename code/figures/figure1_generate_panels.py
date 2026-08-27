#!/usr/bin/env python3
"""Generate the 12 standalone PCA3 image panels used by Figure 1.

Each task renders one model at one 24 km square location. Final overlap-workflow
tiles are selected from their source geometry without opening every state tile.
The generated PNGs are deliberately clean (no title or axes) so assembly and
typographic styling remain a separate, reproducible step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.image as mpimg
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from figure1_rgb_panel import make_rgb_panel


SCRATCH_ROOT = Path(
    os.environ.get("FIGURE1_SCRATCH_ROOT", "/scratch/users/nhendrix/embeddings-health")
)
OVERLAP_ROOT = SCRATCH_ROOT / "embedding_workflow_overlap_v1"
PRITHVI_ROOT = SCRATCH_ROOT / "prithvi_embeddings"
PRITHVI_FIGURE1_RECONSTRUCTED_ROOT = (
    SCRATCH_ROOT / "figures" / "figure1_prithvi_reconstructed"
)
OUTPUT_ROOT = Path(
    os.environ.get("FIGURE1_OUTPUT_ROOT", str(SCRATCH_ROOT / "figures" / "figure1_panels"))
)
BOX_KM = 24.0

LOCATIONS = {
    "palo_alto": {
        "label": "Palo Alto, CA",
        "lat": 37.4419,
        "lon": -122.1430,
        "states": ("CA",),
        "alphaearth": SCRATCH_ROOT / "figures" / "alphaearth_embeddings_2022_palo_alto.tif",
    },
    "washington_dc": {
        "label": "Washington, DC",
        "lat": 38.9072,
        "lon": -77.0369,
        "states": ("DC", "MD", "VA"),
        "alphaearth": SCRATCH_ROOT / "figures" / "alphaearth_embeddings_2022_washington_dc.tif",
    },
}

MODELS = {
    "alphaearth_foundations": {
        "label": "AlphaEarth Foundations",
        "kind": "alphaearth",
    },
    "olmoearth_v1_2_nano": {
        "label": "OlmoEarth v1.2 Nano",
        "kind": "overlap",
        "directory": "olmoearth-v1.2-nano",
        "stem": "olmoearth-v1.2-nano_overlap-center50",
    },
    "olmoearth_v1_2_base": {
        "label": "OlmoEarth v1.2 Base",
        "kind": "overlap",
        "directory": "olmoearth-v1.2-base",
        "stem": "olmoearth-v1.2-base_overlap-center50",
    },
    "clay_v1_5": {
        "label": "Clay v1.5",
        "kind": "overlap",
        "directory": "clay-1.5",
        "stem": "clay-1.5_overlap-center50",
    },
    "prithvi_eo_2_tiny_tl": {
        "label": "Prithvi-EO-2.0 tiny-TL",
        "kind": "prithvi",
        "variant": "tiny",
        "stem": "prithvi_tiny",
        "raw_suffix": "_raw",
    },
    "prithvi_eo_2_300m_tl": {
        "label": "Prithvi-EO-2.0 300M-TL",
        "kind": "prithvi",
        "variant": "300M-TL",
        "stem": "prithvi_300M-TL",
        "raw_suffix": "_raw",
    },
}

MODEL_ORDER = tuple(MODELS)
LOCATION_ORDER = tuple(LOCATIONS)
TASKS = tuple(
    (location, model)
    for location in LOCATION_ORDER
    for model in MODEL_ORDER
)


def _partitions(pixels: int, stride: int, divisions: int) -> list[tuple[int, int]]:
    """Return source-pixel ownership intervals for an overlap tile axis."""
    blocks = math.ceil(pixels / stride)
    if not 1 <= divisions <= blocks:
        raise ValueError(
            f"invalid tile divisions={divisions} for {blocks} retained blocks"
        )
    base, extra = divmod(blocks, divisions)
    intervals = []
    for index in range(divisions):
        start_block = index * base + min(index, extra)
        stop_block = start_block + base + (index < extra)
        intervals.append(
            (start_block * stride, min(stop_block * stride, pixels))
        )
    return intervals


def _overlap_tiles_for_state(
    model: dict,
    state: str,
    lat: float,
    lon: float,
    box_km: float,
) -> list[Path]:
    directory = OVERLAP_ROOT / model["directory"] / state
    pattern = f'{model["stem"]}_{state}_2022*.tif'
    available = sorted(directory.glob(pattern))
    if not available:
        raise FileNotFoundError(f"No overlap tiles match {directory / pattern}")

    with rasterio.open(available[0]) as first:
        tags = first.tags()
        tile_crs = first.crs
        grid_rows, grid_cols = map(int, tags["tile_grid"].split("x"))

    num_tiles = grid_rows * grid_cols

    def tile_path(row: int, col: int) -> Path:
        index = row * grid_cols + col
        name = (
            f'{model["stem"]}_{state}_2022.tif'
            if num_tiles == 1
            else f'{model["stem"]}_{state}_2022_tile{index:03d}.tif'
        )
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected overlap tile is missing: {path}")
        return path

    cx, cy = Transformer.from_crs(
        "EPSG:4326", tile_crs, always_xy=True
    ).transform(lon, lat)
    half_m = box_km * 500.0
    left, bottom, right, top = (
        cx - half_m,
        cy - half_m,
        cx + half_m,
        cy + half_m,
    )

    # Read one representative tile per grid row/column. This remains fast even
    # when an old source-composite path in the COG tags has since been retired.
    rows = set()
    for row in range(grid_rows):
        with rasterio.open(tile_path(row, 0)) as tile:
            if tile.bounds.top > bottom and tile.bounds.bottom < top:
                rows.add(row)

    cols = set()
    for col in range(grid_cols):
        with rasterio.open(tile_path(0, col)) as tile:
            if tile.bounds.right > left and tile.bounds.left < right:
                cols.add(col)

    if not rows or not cols:
        return []

    # Include one neighboring tile in every direction. This is cheap (normally
    # at most nine files) and protects selection at transformed CRS boundaries.
    rows = {
        candidate
        for row in rows
        for candidate in (row - 1, row, row + 1)
        if 0 <= candidate < grid_rows
    }
    cols = {
        candidate
        for col in cols
        for candidate in (col - 1, col, col + 1)
        if 0 <= candidate < grid_cols
    }

    num_tiles = grid_rows * grid_cols
    selected = []
    for row in sorted(rows):
        for col in sorted(cols):
            index = row * grid_cols + col
            name = (
                f'{model["stem"]}_{state}_2022.tif'
                if num_tiles == 1
                else f'{model["stem"]}_{state}_2022_tile{index:03d}.tif'
            )
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Expected overlap tile is missing: {path}")
            selected.append(path)
    return selected


def source_paths(model_key: str, location_key: str) -> list[Path]:
    model = MODELS[model_key]
    location = LOCATIONS[location_key]

    if model["kind"] == "alphaearth":
        paths = [location["alphaearth"]]
    elif model["kind"] == "overlap":
        paths = []
        for state in location["states"]:
            paths.extend(
                _overlap_tiles_for_state(
                    model,
                    state,
                    location["lat"],
                    location["lon"],
                    BOX_KM,
                )
            )
    elif model["kind"] == "prithvi":
        safe_variant = model["variant"].replace("-", "_").lower()
        local_centered = (
            PRITHVI_FIGURE1_RECONSTRUCTED_ROOT
            / f"prithvi_{safe_variant}_{location_key}_centered.tif"
        )
        if local_centered.is_file():
            return [local_centered]
        paths = []
        for state in location["states"]:
            suffix = model["raw_suffix"]
            candidate = (
                PRITHVI_ROOT
                / model["variant"]
                / state
                / f'{model["stem"]}_{state}_2022{suffix}.tif'
            )
            if not candidate.is_file() and suffix:
                candidate = (
                    PRITHVI_ROOT
                    / model["variant"]
                    / state
                    / f'{model["stem"]}_{state}_2022.tif'
                )
            paths.append(candidate)
    else:
        raise ValueError(f'Unknown source kind: {model["kind"]}')

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source files:\n" + "\n".join(f"  {path}" for path in missing)
        )
    if not paths:
        raise FileNotFoundError(
            f"No source rasters overlap {location_key} for {model_key}"
        )
    return paths


def render_task(
    model_key: str,
    location_key: str,
    force: bool = False,
) -> Path:
    model = MODELS[model_key]
    location = LOCATIONS[location_key]
    output_dir = OUTPUT_ROOT / location_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{model_key}_pca3.png"
    metadata_path = output.with_suffix(".json")

    if output.is_file() and metadata_path.is_file() and not force:
        print(f"Already complete: {output}")
        return output

    paths = source_paths(model_key, location_key)
    print(
        f'Rendering {location["label"]} / {model["label"]} '
        f"from {len(paths)} raster(s)",
        flush=True,
    )
    for path in paths:
        print(f"  {path}", flush=True)

    with rasterio.open(paths[0]) as source:
        embedding_dimensions = source.count
        native_resolution = list(source.res)
        source_crs = str(source.crs)

    rgb, explained = make_rgb_panel(
        paths,
        location["lat"],
        location["lon"],
        BOX_KM,
        max_bands=None,
    )
    mpimg.imsave(output, rgb)

    metadata = {
        "schema": 1,
        "model_key": model_key,
        "model_label": model["label"],
        "location_key": location_key,
        "location_label": location["label"],
        "center_lat": location["lat"],
        "center_lon": location["lon"],
        "box_km": BOX_KM,
        "year": 2022,
        "pca_components": 3,
        "pca_fit_sample_limit": 50000,
        "pca_random_seed": 42,
        "embedding_dimensions_used": embedding_dimensions,
        "pca_explained_variance_sum": float(explained),
        "rgb_scaling": "per-component min-max over valid crop pixels",
        "image_shape": list(rgb.shape),
        "source_crs": source_crs,
        "native_resolution": native_resolution,
        "source_rasters": [str(path) for path in paths],
        "output": str(output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output}", flush=True)
    print(f"Saved {metadata_path}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--location", choices=LOCATION_ORDER)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        for index, (location, model) in enumerate(TASKS):
            print(f"{index:02d} {location} {model}")
        return

    if args.all:
        if args.task_index is not None or args.model or args.location:
            parser.error("--all cannot be combined with task selectors")
        for location_key, model_key in TASKS:
            render_task(model_key, location_key, force=args.force)
        return

    if args.task_index is not None:
        if args.model or args.location:
            parser.error("--task-index cannot be combined with --model/--location")
        try:
            location_key, model_key = TASKS[args.task_index]
        except IndexError:
            parser.error(f"--task-index must be between 0 and {len(TASKS) - 1}")
    elif args.model and args.location:
        model_key = args.model
        location_key = args.location
    else:
        parser.error("provide --task-index or both --model and --location")

    render_task(model_key, location_key, force=args.force)


if __name__ == "__main__":
    main()
