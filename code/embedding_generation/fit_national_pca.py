"""
Fit a single national PCA across all complete state raw embeddings.

Samples pixels size-proportionally from each state's raw embedding TIF,
then fits a PCA on the combined pool.  The resulting .pkl is used by
aggregate.py (--pca-model) so that every state's tract statistics share
a consistent, nationally-comparable embedding space.

TIF preference (prithvi only):
  *_raw.tif  — written by older runs that also produced a PCA-reduced .tif
  *.tif      — written by newer runs using --no-pca (raw is the primary output)
Clay has no variant subdirectory or raw/plain distinction — one file per state.

Usage:
  python fit_national_pca.py --model prithvi \\
    --variant tiny --year 2022 \\
    --embed-dir $SCRATCH/embeddings-health/prithvi_embeddings \\
    --output    $SCRATCH/embeddings-health/prithvi_aggregated/national_pca/prithvi_tiny_national_pca.pkl

  python fit_national_pca.py --model clay --year 2022 \\
    --embed-dir $SCRATCH/embeddings-health/clay_embeddings \\
    --output    $SCRATCH/embeddings-health/clay_aggregated/national_pca/clay_v1.5_national_pca.pkl
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.decomposition import PCA
from tqdm import tqdm


def _safe_variant(variant: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", variant)


def find_raw_tif(variant_dir: Path, state: str, safe_var: str, year: int, min_bands: int = 65) -> Path | None:
    """Return the raw (full-dim) embedding TIF for this Prithvi state, or None.

    Prefers *_raw.tif (older PCA-enabled runs).  Falls back to *.tif and
    verifies the band count exceeds min_bands to exclude PCA-compressed files.
    """
    raw = variant_dir / state / f"prithvi_{safe_var}_{state}_{year}_raw.tif"
    if raw.is_file() and raw.stat().st_size > 0:
        return raw
    plain = variant_dir / state / f"prithvi_{safe_var}_{state}_{year}.tif"
    if plain.is_file() and plain.stat().st_size > 0:
        with rasterio.open(plain) as src:
            if src.count >= min_bands:
                return plain
    return None


def find_clay_tifs(embed_dir: Path, state: str, year: int) -> list[Path]:
    """Return raw Clay embedding TIFs for this state.

    Supports both the older one-state-one-TIF layout and the newer overlap tile
    layout. Overlap tiles are non-overlapping retained-center products, so they
    are valid sampling units for the national PCA fit.
    """
    state_dir = embed_dir / state
    legacy = state_dir / f"clay_v1.5_{state}_{year}.tif"
    if legacy.is_file() and legacy.stat().st_size > 0:
        return [legacy]

    overlap_tiles = sorted(
        p for p in state_dir.glob(f"clay-1.5_overlap-center50_{state}_{year}_tile*.tif")
        if p.is_file() and p.stat().st_size > 0
        and p.with_suffix(".validation.json").is_file()
        and p.with_suffix(".validation.json").stat().st_size > 0
    )
    if overlap_tiles:
        return overlap_tiles

    overlap_plain = state_dir / f"clay-1.5_overlap-center50_{state}_{year}.tif"
    overlap_plain_validation = overlap_plain.with_suffix(".validation.json")
    if (
        overlap_plain.is_file()
        and overlap_plain.stat().st_size > 0
        and overlap_plain_validation.is_file()
        and overlap_plain_validation.stat().st_size > 0
    ):
        return [overlap_plain]

    return sorted(
        p for p in state_dir.glob(f"clay-1.5_overlap-center50_{state}_{year}_tile*.tif")
        if p.is_file() and p.stat().st_size > 0
    )


def sample_raster(tif_path: Path, n_samples: int, rng, block_size: int = 256) -> np.ndarray:
    """Sample up to n_samples valid pixels via random block reads (memory-efficient)."""
    samples = []
    remaining = n_samples
    with rasterio.open(tif_path) as src:
        H, W, D = src.height, src.width, src.count
        row_offs = list(range(0, H, block_size))
        col_offs = list(range(0, W, block_size))
        blocks = [(r, c) for r in row_offs for c in col_offs]
        rng.shuffle(blocks)
        for r, c in blocks:
            if remaining <= 0:
                break
            h = min(block_size, H - r)
            w = min(block_size, W - c)
            block = src.read(window=Window(c, r, w, h)).astype(np.float32)  # (D, h, w)
            flat = block.reshape(D, -1).T                                    # (n_px, D)
            valid = ~np.isnan(flat).any(axis=1)
            flat = flat[valid]
            if len(flat) == 0:
                continue
            n_take = min(remaining, len(flat))
            idx = rng.choice(len(flat), n_take, replace=False)
            samples.append(flat[idx])
            remaining -= n_take
    if not samples:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(samples)


def sample_state(
    tif_paths: list[Path],
    n_samples: int,
    rng,
    block_size: int = 256,
    max_rasters: int | None = None,
) -> np.ndarray:
    """Sample a state represented by one or more raw embedding TIFs.

    For overlap products, visiting every tile in large states is slow and adds
    little to a national PCA sample. Select a bounded, area-weighted subset of
    tiles per state, then allocate that state's sample quota across those tiles.
    """
    if not tif_paths:
        return np.empty((0,), dtype=np.float32)

    areas = []
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            areas.append(src.height * src.width)
    areas = np.asarray(areas, dtype=np.float64)

    if max_rasters is not None and len(tif_paths) > max_rasters:
        probabilities = areas / areas.sum()
        chosen = np.sort(rng.choice(len(tif_paths), size=max_rasters, replace=False, p=probabilities))
        tif_paths = [tif_paths[int(i)] for i in chosen]
        areas = areas[chosen]

    total_area = float(areas.sum())
    raw_allocations = [n_samples * float(area) / total_area for area in areas]
    allocations = [int(value) for value in raw_allocations]
    remainder = n_samples - sum(allocations)
    order = np.argsort([value - int(value) for value in raw_allocations])[::-1]
    for idx in order[:remainder]:
        allocations[int(idx)] += 1

    samples = [
        sample_raster(tif_path, n, rng, block_size=block_size)
        for tif_path, n in zip(tif_paths, allocations)
        if n > 0
    ]
    samples = [sample for sample in samples if sample.ndim == 2 and len(sample) > 0]
    if not samples:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["prithvi", "clay"], default="prithvi",
                        help="Determines directory layout and filename pattern.")
    parser.add_argument("--variant", default=None,
                        help="Model variant name (e.g. tiny, 300M-TL). Required for "
                             "--model prithvi; unused for --model clay (one variant).")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--embed-dir", type=Path, required=True,
                        help="Root embeddings directory: parent of tiny/, 300M-TL/ etc. "
                             "for prithvi, or parent of per-state dirs directly for clay.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for the fitted PCA .pkl")
    parser.add_argument("--n-components", type=int, default=64,
                        help="Number of PCA components (default 64)")
    parser.add_argument("--total-samples", type=int, default=500_000,
                        help="Target total pixels across all states (default 500000)")
    parser.add_argument("--min-samples-per-state", type=int, default=500,
                        help="Floor on per-state sample count (default 500)")
    parser.add_argument("--max-rasters-per-state", type=int, default=16,
                        help="For tiled Clay overlap inputs, sample at most this many GeoTIFFs per state (default 16)")
    args = parser.parse_args()

    if args.model == "prithvi" and not args.variant:
        sys.exit("ERROR: --variant is required for --model prithvi")

    if args.model == "prithvi":
        safe_var = _safe_variant(args.variant)
        scan_dir = args.embed_dir / args.variant
    else:
        scan_dir = args.embed_dir

    if not scan_dir.is_dir():
        sys.exit(f"ERROR: directory not found: {scan_dir}")

    # Discover states with a valid raw TIF
    tif_paths: dict[str, list[Path]] = {}
    tif_areas: dict[str, int] = {}  # H×W as proxy for state size
    for state_dir in sorted(scan_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        state = state_dir.name
        if args.model == "prithvi":
            tif = find_raw_tif(scan_dir, state, safe_var, args.year)
            tifs = [tif] if tif is not None else []
        else:
            tifs = find_clay_tifs(scan_dir, state, args.year)
        if not tifs:
            continue
        area = 0
        for tif in tifs:
            with rasterio.open(tif) as src:
                area += src.height * src.width
        tif_paths[state] = tifs
        tif_areas[state] = area

    if not tif_paths:
        sys.exit("ERROR: no suitable raw embedding TIFs found. "
                 "Check --embed-dir (and --variant for prithvi), and ensure at least "
                 "some states are complete.")

    print(f"Model:          {args.model}" + (f"  (variant: {args.variant})" if args.variant else ""))
    print(f"States found:   {len(tif_paths)}")
    print(f"Target samples: {args.total_samples:,}")
    print(f"Components:     {args.n_components}")
    if args.model == "clay":
        print(f"Max rasters/state: {args.max_rasters_per_state}")
    print()

    # Allocate samples proportionally by TIF area, with a per-state floor
    total_area = sum(tif_areas.values())
    samples_per_state = {
        state: max(
            args.min_samples_per_state,
            int(args.total_samples * tif_areas[state] / total_area),
        )
        for state in tif_paths
    }
    print(f"Total allocated samples: {sum(samples_per_state.values()):,}")
    print()

    # Sample from each state
    rng = np.random.default_rng(42)
    all_samples: list[np.ndarray] = []
    for state in tqdm(sorted(tif_paths), desc="Sampling states", unit="state"):
        max_rasters = args.max_rasters_per_state if args.model == "clay" else None
        pixels = sample_state(tif_paths[state], samples_per_state[state], rng, max_rasters=max_rasters)
        if pixels.ndim < 2 or len(pixels) == 0:
            print(f"  WARNING: no valid pixels in {state} — skipping")
            continue
        all_samples.append(pixels)
        source_desc = (
            tif_paths[state][0].name
            if len(tif_paths[state]) == 1
            else f"{len(tif_paths[state])} tiled GeoTIFFs"
        )
        tqdm.write(f"  {state}: {len(pixels):,} px  (target {samples_per_state[state]:,})  "
                   f"from {source_desc}")

    if not all_samples:
        sys.exit("ERROR: no valid pixels collected — cannot fit PCA.")

    X = np.vstack(all_samples)
    print(f"\nTotal pixels for PCA fit: {X.shape[0]:,} × {X.shape[1]} dims")

    print(f"Fitting PCA({args.n_components}, randomized SVD)…")
    pca = PCA(n_components=args.n_components, random_state=42, svd_solver="randomized")
    pca.fit(X)

    cum_var = pca.explained_variance_ratio_.cumsum()
    print(f"Explained variance — PC1: {pca.explained_variance_ratio_[0]:.1%}  "
          f"PC{args.n_components}: {cum_var[-1]:.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
