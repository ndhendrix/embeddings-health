"""Fit a national PCA from validated overlap embedding tiles."""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.decomposition import PCA
from tqdm import tqdm


def tile_path(output_root: Path, model: str, state: str, year: int, index: int, num_tiles: int) -> Path:
    stem = f"{model}_overlap-center50_{state}_{year}"
    directory = output_root / model / state
    if num_tiles == 1:
        return directory / f"{stem}.tif"
    return directory / f"{stem}_tile{index:03d}.tif"


def read_task_file(path: Path, output_root: Path, model: str, year: int, require_validation: bool) -> dict[str, list[Path]]:
    by_state: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        state, index, num_tiles, *_ = raw.split()
        tile = tile_path(output_root, model, state, year, int(index), int(num_tiles))
        if tile in seen:
            continue
        if not tile.is_file() or tile.stat().st_size == 0:
            raise FileNotFoundError(f"missing tile: {tile}")
        validation = tile.with_suffix(".validation.json")
        if require_validation and (not validation.is_file() or validation.stat().st_size == 0):
            raise FileNotFoundError(f"missing validation: {validation}")
        by_state[state].append(tile)
        seen.add(tile)
    return dict(sorted(by_state.items()))


def sample_raster(tif_path: Path, n_samples: int, rng: np.random.Generator, block_size: int) -> np.ndarray:
    samples: list[np.ndarray] = []
    remaining = n_samples
    with rasterio.open(tif_path) as src:
        rows = list(range(0, src.height, block_size))
        cols = list(range(0, src.width, block_size))
        blocks = [(r, c) for r in rows for c in cols]
        rng.shuffle(blocks)
        for row, col in blocks:
            if remaining <= 0:
                break
            height = min(block_size, src.height - row)
            width = min(block_size, src.width - col)
            block = src.read(window=Window(col, row, width, height)).astype(np.float32)
            flat = block.reshape(src.count, -1).T
            valid = np.isfinite(flat).all(axis=1)
            flat = flat[valid]
            if len(flat) == 0:
                continue
            take = min(remaining, len(flat))
            choice = rng.choice(len(flat), take, replace=False)
            samples.append(flat[choice])
            remaining -= take
    if not samples:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(samples)


def sample_state(
    tif_paths: list[Path],
    n_samples: int,
    rng: np.random.Generator,
    block_size: int,
    max_rasters: int | None,
) -> np.ndarray:
    areas = []
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            areas.append(src.height * src.width)
    areas_array = np.asarray(areas, dtype=np.float64)
    if max_rasters is not None and len(tif_paths) > max_rasters:
        probabilities = areas_array / areas_array.sum()
        chosen = np.sort(rng.choice(len(tif_paths), size=max_rasters, replace=False, p=probabilities))
        tif_paths = [tif_paths[int(i)] for i in chosen]
        areas_array = areas_array[chosen]

    raw_allocations = n_samples * areas_array / float(areas_array.sum())
    allocations = [int(value) for value in raw_allocations]
    remainder = n_samples - sum(allocations)
    order = np.argsort([value - int(value) for value in raw_allocations])[::-1]
    for idx in order[:remainder]:
        allocations[int(idx)] += 1

    pieces = [
        sample_raster(path, n, rng, block_size)
        for path, n in zip(tif_paths, allocations)
        if n > 0
    ]
    pieces = [piece for piece in pieces if piece.ndim == 2 and len(piece) > 0]
    if not pieces:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--total-samples", type=int, default=500_000)
    parser.add_argument("--min-samples-per-state", type=int, default=500)
    parser.add_argument("--max-rasters-per-state", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--no-require-validation", action="store_true")
    args = parser.parse_args()

    by_state = read_task_file(
        args.task_file,
        args.output_root,
        args.model,
        args.year,
        require_validation=not args.no_require_validation,
    )
    if not by_state:
        raise SystemExit("ERROR: no tiles found in task file")

    state_areas = {}
    state_dims = {}
    for state, paths in by_state.items():
        area = 0
        dims = set()
        for path in paths:
            with rasterio.open(path) as src:
                area += src.height * src.width
                dims.add(src.count)
        if len(dims) != 1:
            raise ValueError(f"mixed band counts for {state}: {sorted(dims)}")
        state_areas[state] = area
        state_dims[state] = next(iter(dims))
    dimensions = set(state_dims.values())
    if len(dimensions) != 1:
        raise ValueError(f"mixed band counts across states: {sorted(dimensions)}")

    total_area = sum(state_areas.values())
    samples_per_state = {
        state: max(args.min_samples_per_state, int(args.total_samples * state_areas[state] / total_area))
        for state in by_state
    }

    print(f"Model: {args.model}")
    print(f"States: {len(by_state)}")
    print(f"Tiles: {sum(len(paths) for paths in by_state.values())}")
    print(f"Input dimensions: {next(iter(dimensions))}")
    print(f"Components: {args.n_components}")
    print(f"Target samples: {args.total_samples:,}")
    print(f"Allocated samples: {sum(samples_per_state.values()):,}")
    print(f"Max rasters/state: {args.max_rasters_per_state}")

    rng = np.random.default_rng(42)
    all_samples = []
    for state in tqdm(sorted(by_state), desc="Sampling states", unit="state"):
        pixels = sample_state(
            by_state[state],
            samples_per_state[state],
            rng,
            args.block_size,
            args.max_rasters_per_state,
        )
        if pixels.ndim < 2 or len(pixels) == 0:
            tqdm.write(f"  WARNING: no valid pixels in {state}; skipping")
            continue
        all_samples.append(pixels)
        tqdm.write(
            f"  {state}: {len(pixels):,} px (target {samples_per_state[state]:,}) "
            f"from {len(by_state[state])} tiles"
        )

    if not all_samples:
        raise SystemExit("ERROR: no valid pixels collected; cannot fit PCA")
    matrix = np.vstack(all_samples)
    print(f"Total pixels for PCA fit: {matrix.shape[0]:,} x {matrix.shape[1]}")
    print(f"Fitting PCA({args.n_components}, randomized SVD)...")
    pca = PCA(n_components=args.n_components, random_state=42, svd_solver="randomized")
    pca.fit(matrix)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"Explained variance at {args.n_components} components: {explained[-1]:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(pca, handle)
    print(f"PCA model saved to: {args.output}")


if __name__ == "__main__":
    main()
