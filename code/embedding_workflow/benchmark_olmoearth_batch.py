"""Benchmark OlmoEarth batch sizes using a fixed in-memory chip sample."""
import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "embedding_generation"))
import embed as engine
from models import get_model
from overlap_geometry import OverlapGrid


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def preload_chips(path: Path, grid: OverlapGrid, num_chips: int):
    started = time.perf_counter()
    with rasterio.open(path) as src:
        if src.count != 12:
            raise ValueError(f"expected 12 Sentinel-2 bands, found {src.count}")
        block_rows = grid.block_count(src.height)
        block_cols = grid.block_count(src.width)
        total_blocks = block_rows * block_cols
        if not 1 <= num_chips <= total_blocks:
            raise ValueError(f"num_chips must be in [1, {total_blocks}]")

        # Evenly cover the state rather than benchmarking one unusually coherent area.
        flat_indices = np.linspace(0, total_blocks - 1, num_chips, dtype=np.int64)
        chips = np.empty((num_chips, src.count, grid.chip, grid.chip), dtype=np.float32)
        latlons = np.empty((num_chips, 2), dtype=np.float32)
        for index, flat_index in enumerate(flat_indices):
            block_row, block_col = divmod(int(flat_index), block_cols)
            row = grid.chip_origin(block_row)
            col = grid.chip_origin(block_col)
            chips[index] = src.read(
                window=Window(col, row, grid.chip, grid.chip),
                boundless=True,
                fill_value=np.nan,
                out_dtype="float32",
            )
            latlons[index] = engine.chip_center_latlon(
                src.transform, src.crs, row, col, grid.chip
            )
            if (index + 1) % 256 == 0:
                print(f"preloaded {index + 1}/{num_chips} chips", flush=True)

        metadata = {
            "input_width": src.width,
            "input_height": src.height,
            "input_bands": src.count,
            "available_blocks": total_blocks,
            "sample_strategy": "evenly-spaced-over-flat-overlap-grid",
            "preload_seconds": time.perf_counter() - started,
            "host_sample_bytes": chips.nbytes + latlons.nbytes,
        }
    return chips, latlons, metadata


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def infer_sample(model, chips, latlons, device, year, batch_size):
    first_value = None
    output_shape = None
    for offset in range(0, len(chips), batch_size):
        output = engine.run_olmoearth_batch(
            model,
            chips[offset : offset + batch_size],
            latlons[offset : offset + batch_size],
            device,
            year,
        )
        output_shape = list(output.shape[1:])
        if first_value is None:
            first_value = float(output.flat[0])
        del output
    return output_shape, first_value


def benchmark_batch_size(model, chips, latlons, device, year, batch_size, repeats):
    result = {"batch_size": batch_size, "status": "running", "repeats": repeats}
    try:
        gc.collect()
        torch.cuda.empty_cache()
        infer_sample(
            model,
            chips[:batch_size],
            latlons[:batch_size],
            device,
            year,
            batch_size,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        elapsed_seconds = []
        output_shape = None
        first_value = None
        for repeat in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output_shape, first_value = infer_sample(
                model, chips, latlons, device, year, batch_size
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            elapsed_seconds.append(elapsed)
            print(
                f"batch={batch_size} repeat={repeat + 1}/{repeats} "
                f"seconds={elapsed:.3f} chips_per_second={len(chips) / elapsed:.3f}",
                flush=True,
            )

        median_seconds = float(np.median(elapsed_seconds))
        result.update(
            status="completed",
            elapsed_seconds=elapsed_seconds,
            median_seconds=median_seconds,
            chips_per_second=len(chips) / median_seconds,
            retained_blocks_per_minute=60 * len(chips) / median_seconds,
            estimated_12000_block_minutes=12000 * median_seconds / len(chips) / 60,
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            output_shape=output_shape,
            first_output_value=first_value,
        )
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        result.update(status="oom", error=str(error))
        print(f"batch={batch_size} status=oom: {error}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="olmoearth-v1.2-nano")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--num-chips", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--year", type=int, default=2022)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if any(size < 1 or size > args.num_chips for size in args.batch_sizes):
        raise ValueError("batch sizes must be positive and no larger than num_chips")

    spec = get_model(args.model)
    if spec.family != "olmoearth":
        raise ValueError("this benchmark only supports OlmoEarth models")
    grid = OverlapGrid(spec.chip_pixels, spec.patch_pixels)
    chips, latlons, input_metadata = preload_chips(args.input, grid, args.num_chips)

    device = torch.device("cuda")
    model_started = time.perf_counter()
    model = engine.load_olmoearth(spec.variant).to(device)
    model_load_seconds = time.perf_counter() - model_started
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema": 1,
        "model": args.model,
        "repository": spec.repository,
        "revision": spec.revision,
        "input": str(args.input.resolve()),
        "year": args.year,
        "normalization": engine.OLMOEARTH_NORMALIZATION,
        "chip_pixels": grid.chip,
        "stride_pixels": grid.stride,
        "retained_center_pixels": grid.stride,
        "patch_pixels": grid.patch,
        "num_chips": args.num_chips,
        "model_load_seconds": model_load_seconds,
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        **input_metadata,
        "results": [],
    }
    write_json(args.output, payload)

    for batch_size in args.batch_sizes:
        result = benchmark_batch_size(
            model, chips, latlons, device, args.year, batch_size, args.repeats
        )
        payload["results"].append(result)
        write_json(args.output, payload)

    completed = [row for row in payload["results"] if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("every requested batch size ran out of GPU memory")
    fastest = max(completed, key=lambda row: row["chips_per_second"])
    payload["fastest_completed_batch_size"] = fastest["batch_size"]
    payload["fastest_chips_per_second"] = fastest["chips_per_second"]
    write_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
