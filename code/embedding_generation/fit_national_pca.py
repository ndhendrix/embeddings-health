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


def find_clay_tif(embed_dir: Path, state: str, year: int) -> Path | None:
    """Return the (only, always-raw) Clay embedding TIF for this state, or None."""
    tif = embed_dir / state / f"clay_v1.5_{state}_{year}.tif"
    return tif if tif.is_file() and tif.stat().st_size > 0 else None


def sample_state(tif_path: Path, n_samples: int, rng, block_size: int = 256) -> np.ndarray:
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
    tif_paths: dict[str, Path] = {}
    tif_areas: dict[str, int] = {}  # H×W as proxy for state size
    for state_dir in sorted(scan_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        state = state_dir.name
        if args.model == "prithvi":
            tif = find_raw_tif(scan_dir, state, safe_var, args.year)
        else:
            tif = find_clay_tif(scan_dir, state, args.year)
        if tif is None:
            continue
        with rasterio.open(tif) as src:
            tif_paths[state] = tif
            tif_areas[state] = src.height * src.width

    if not tif_paths:
        sys.exit("ERROR: no suitable raw embedding TIFs found. "
                 "Check --embed-dir (and --variant for prithvi), and ensure at least "
                 "some states are complete.")

    print(f"Model:          {args.model}" + (f"  (variant: {args.variant})" if args.variant else ""))
    print(f"States found:   {len(tif_paths)}")
    print(f"Target samples: {args.total_samples:,}")
    print(f"Components:     {args.n_components}")
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
        pixels = sample_state(tif_paths[state], samples_per_state[state], rng)
        if pixels.ndim < 2 or len(pixels) == 0:
            print(f"  WARNING: no valid pixels in {state} — skipping")
            continue
        all_samples.append(pixels)
        tqdm.write(f"  {state}: {len(pixels):,} px  (target {samples_per_state[state]:,})  "
                   f"from {tif_paths[state].name}")

    if not all_samples:
        sys.exit("ERROR: no valid pixels collected — cannot fit PCA.")

    X = np.vstack(all_samples)
    print(f"\nTotal pixels for PCA fit: {X.shape[0]:,} × {X.shape[1]} dims")

    print(f"Fitting PCA({args.n_components})…")
    pca = PCA(n_components=args.n_components, random_state=42)
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
