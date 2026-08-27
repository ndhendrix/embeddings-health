"""Plan model-aware overlap tiles from state composite dimensions."""
import argparse
import json
import math
from pathlib import Path

import rasterio

from models import get_model
from overlap_geometry import OverlapGrid
from plan_clay_tasks import choose_rectangular_grid


def validated_output(tile: Path, source: Path, model: str, revision: str, family: str) -> bool:
    validation = tile.with_suffix(".validation.json")
    if not tile.is_file() or not validation.is_file():
        return False
    try:
        report = json.loads(validation.read_text())
        tags = report["tags"]
        if report["model"] != model or tags["model"] != model:
            return False
        if tags["model_revision"] != revision:
            return False
        if Path(tags["source_composite"]).resolve() != source.resolve():
            return False
        source_stat = source.stat()
        if tags.get("source_size") != str(source_stat.st_size):
            return False
        if tags.get("source_mtime_ns") != str(source_stat.st_mtime_ns):
            return False
        if family == "olmoearth":
            return (
                tags.get("workflow") == "overlap-center50-v2"
                and tags.get("input_normalization") == "computed-2std"
            )
        return tags.get("workflow") == "overlap-center50-v1"
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--composites", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--all-task-file", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--target-blocks", type=int, default=20000)
    parser.add_argument("--max-tasks", type=int, default=80)
    parser.add_argument("--states", nargs="*")
    parser.add_argument("--exclude-states", nargs="*", default=[])
    parser.add_argument("--tiling", choices=("rectangular", "rows"), default="rectangular")
    args = parser.parse_args()

    spec = get_model(args.model)
    grid = OverlapGrid(spec.chip_pixels, spec.patch_pixels)
    allowed = set(args.states or [])
    excluded = set(args.exclude_states)
    tasks = []
    all_tasks = []
    state_lines = []
    records = []
    total_output_bytes = 0
    total_tiles = 0
    total_missing_tiles = 0

    pattern = f"s2_annual_*_{args.year}_olmoearth.tif"
    for source in sorted(args.composites.glob(pattern)):
        prefix = "s2_annual_"
        suffix = f"_{args.year}_olmoearth.tif"
        state = source.name[len(prefix) : -len(suffix)]
        if (allowed and state not in allowed) or state in excluded:
            continue

        with rasterio.open(source) as src:
            block_rows = grid.block_count(src.height)
            block_cols = grid.block_count(src.width)
            input_shape = [src.height, src.width]
        if args.tiling == "rows":
            grid_rows = min(
                block_rows,
                max(1, math.ceil(block_rows * block_cols / args.target_blocks)),
            )
            grid_cols = 1
        else:
            grid_rows, grid_cols = choose_rectangular_grid(
                block_rows, block_cols, args.target_blocks
            )

        num_tiles = grid_rows * grid_cols
        stem = f"{args.model}_overlap-center50_{state}_{args.year}"
        directory = args.output_root / args.model / state
        missing = []
        for index in range(num_tiles):
            tile = (
                directory / f"{stem}_tile{index:03d}.tif"
                if num_tiles > 1
                else directory / f"{stem}.tif"
            )
            task = (state, index, num_tiles, grid_rows, grid_cols)
            all_tasks.append(task)
            if not validated_output(tile, source, args.model, spec.revision, spec.family):
                missing.append(index)
                if len(tasks) < args.max_tasks:
                    tasks.append(task)

        output_rows = math.ceil(input_shape[0] / spec.patch_pixels)
        output_cols = math.ceil(input_shape[1] / spec.patch_pixels)
        output_bytes = spec.dimensions * output_rows * output_cols * 4
        max_blocks = math.ceil(block_rows / grid_rows) * math.ceil(block_cols / grid_cols)
        total_output_bytes += output_bytes
        total_tiles += num_tiles
        total_missing_tiles += len(missing)
        state_lines.append((state, num_tiles, grid_rows, grid_cols))
        records.append(
            {
                "state": state,
                "source": str(source.resolve()),
                "input_shape": input_shape,
                "output_shape": [spec.dimensions, output_rows, output_cols],
                "estimated_uncompressed_output_bytes": output_bytes,
                "block_grid": [block_rows, block_cols],
                "tile_grid": [grid_rows, grid_cols],
                "num_tiles": num_tiles,
                "target_blocks": args.target_blocks,
                "max_blocks_per_tile": max_blocks,
                "missing_tiles": missing,
            }
        )

    if allowed:
        found = {record["state"] for record in records}
        missing_sources = sorted(allowed - excluded - found)
        if missing_sources:
            raise FileNotFoundError(f"missing state composites: {', '.join(missing_sources)}")

    args.task_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.task_file.write_text("".join(f"{s} {i} {n} {gr} {gc}\n" for s, i, n, gr, gc in tasks))
    if args.all_task_file:
        args.all_task_file.parent.mkdir(parents=True, exist_ok=True)
        args.all_task_file.write_text(
            "".join(f"{s} {i} {n} {gr} {gc}\n" for s, i, n, gr, gc in all_tasks)
        )
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(
            "".join(f"{s} {n} {gr} {gc}\n" for s, n, gr, gc in state_lines)
        )

    manifest = {
        "schema": 3,
        "model": args.model,
        "repository": spec.repository,
        "revision": spec.revision,
        "chip": grid.chip,
        "stride": grid.stride,
        "patch": grid.patch,
        "dimensions": spec.dimensions,
        "tiling": args.tiling,
        "target_blocks": args.target_blocks,
        "selected_states": len(records),
        "total_tiles": total_tiles,
        "total_missing_tiles": total_missing_tiles,
        "tasks_in_wave": len(tasks),
        "estimated_uncompressed_output_bytes": total_output_bytes,
        "states": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task_file": str(args.task_file),
                "manifest": str(args.manifest),
                "tasks_in_wave": len(tasks),
                "total_missing_tiles": total_missing_tiles,
                "total_tiles": total_tiles,
                "states_considered": len(records),
                "estimated_uncompressed_output_bytes": total_output_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
