"""
Validate a Prithvi embedding TIF for completeness and consistency.

Checks:
  1. File opens and has the expected band count (192 for tiny, 1024 for 300M-TL).
  2. Zero-pixel coverage: pixels where all sampled bands are zero indicate
     chips that were never written (gaps from interrupted checkpointing).
  3. NaN coverage: any NaN in the embedding bands.
  4. Band statistics sample: mean/std for a handful of bands, optionally
     compared against a reference TIF.

Uses COG overviews for speed — overview level 4 (~1/16 resolution) is
sufficient to locate any zero-patch larger than a few chips.

Usage (sh_dev or small normal-partition job):
    uv run python validate_embedding.py <tif_path> [--ref <ref_tif>]

Example:
    uv run python validate_embedding.py \
        $SCRATCH/embeddings-health/prithvi_embeddings/300M-TL/TX/prithvi_300M-TL_TX_2022.tif \
        --ref $SCRATCH/embeddings-health/prithvi_embeddings/300M-TL/TN/prithvi_300M-TL_TN_2022.tif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


SAMPLE_BANDS = [0, 63, 127, 255, 511, 767, 1023]  # indices, not 1-based; clipped to actual count

# Known valid band counts:
#   192  = Prithvi tiny raw embedding
#  1024  = Prithvi 300M-TL raw embedding
#    64  = PCA-reduced 300M-TL (old pipeline; file has _raw.tif sibling with 1024 bands)
VALID_BAND_COUNTS = {192, 1024, 64}


def _read_overview(src, band_indices, overview_level=4):
    """Read a set of bands at a COG overview level (power-of-2 downscale)."""
    factor = 2 ** overview_level
    out_h = max(1, src.height // factor)
    out_w = max(1, src.width  // factor)
    data = src.read(
        indexes=[i + 1 for i in band_indices],  # rasterio is 1-indexed
        out_shape=(len(band_indices), out_h, out_w),
        resampling=Resampling.average,
    )
    return data  # (len(band_indices), out_h, out_w)


def _check_zero_pixels(data, label="overview"):
    """Return count and fraction of pixels where ALL bands are zero."""
    all_zero = np.all(data == 0, axis=0)
    n_zero = int(all_zero.sum())
    n_total = all_zero.size
    pct = 100.0 * n_zero / n_total if n_total else 0
    print(f"  Zero pixels ({label}): {n_zero:,} / {n_total:,}  ({pct:.2f}%)")
    return n_zero, n_total


def _check_nan_pixels(data, label="overview"):
    any_nan = np.any(np.isnan(data), axis=0)
    n_nan = int(any_nan.sum())
    if n_nan:
        print(f"  NaN pixels  ({label}): {n_nan:,}  *** WARNING ***")
    else:
        print(f"  NaN pixels  ({label}): 0  OK")
    return n_nan


def _band_stats(data, band_indices):
    """Per-band mean and std, printed as a table."""
    print(f"  {'Band':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    for idx, bi in enumerate(band_indices):
        if bi >= data.shape[0]:
            continue
        row = data[idx].ravel()
        row = row[~np.isnan(row)]
        if len(row) == 0:
            print(f"  {bi:>6}  (all NaN)")
            continue
        print(f"  {bi:>6}  {row.mean():>10.4f}  {row.std():>10.4f}"
              f"  {row.min():>10.4f}  {row.max():>10.4f}")


def validate(tif_path: Path, ref_path: Path | None = None, overview_level: int = 4):
    print(f"\n{'='*60}")
    print(f"Validating: {tif_path.name}")
    print(f"{'='*60}")

    if not tif_path.exists():
        print("ERROR: file does not exist.")
        return False

    ok = True

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        height  = src.height
        width   = src.width
        crs     = src.crs
        transform = src.transform
        overviews = src.overviews(1)

        print(f"\nMetadata")
        print(f"  Bands:     {n_bands}  (expect 192=tiny or 1024=300M-TL)")
        print(f"  Size:      {width} × {height} chips")
        print(f"  CRS:       {crs.to_epsg() if crs else 'None'}")
        print(f"  Pixel size:{transform.a:.1f} m  (West edge x: {transform.c:.0f})")
        print(f"  Overviews: {overviews}")
        print(f"  File size: {tif_path.stat().st_size / 1e9:.1f} GB")

        if n_bands == 64:
            print(f"  NOTE: 64 bands = PCA-reduced file (old pipeline). "
                  f"Use _raw.tif sibling for raw-embedding comparison.")
        elif n_bands not in VALID_BAND_COUNTS:
            print(f"  WARNING: unexpected band count {n_bands}")
            ok = False

        # Clip sample bands to actual band count
        sample = [b for b in SAMPLE_BANDS if b < n_bands]
        if n_bands == 192:
            sample = [0, 47, 95, 143, 191]

        print(f"\nCoverage check (overview level {overview_level}, "
              f"~1/{2**overview_level} resolution)")
        data = _read_overview(src, sample, overview_level)
        nz, nt = _check_zero_pixels(data, f"level-{overview_level} overview")
        _check_nan_pixels(data, f"level-{overview_level} overview")

        if nz / nt > 0.01:
            print("  WARNING: >1% zero pixels — possible chip gaps from checkpointing")
            ok = False

        print(f"\nBand statistics (sampled at overview level {overview_level})")
        _band_stats(data, sample)

        # Full-resolution check on a small centre crop (~50×50 chips) to catch
        # any overview-averaging artefacts hiding individual zero chips.
        cx, cy = width // 2, height // 2
        crop_r = 25
        window = rasterio.windows.Window(
            max(0, cx - crop_r), max(0, cy - crop_r),
            min(width,  2 * crop_r), min(height, 2 * crop_r),
        )
        print(f"\nFull-resolution centre crop ({window.width}×{window.height} chips)")
        crop_data = src.read(
            indexes=[b + 1 for b in sample],
            window=window,
        )
        nzc, ntc = _check_zero_pixels(crop_data, "centre crop")
        _check_nan_pixels(crop_data, "centre crop")
        if nzc / ntc > 0.01:
            print("  WARNING: >1% zero pixels in centre crop")
            ok = False

    # Reference comparison
    if ref_path and ref_path.exists():
        print(f"\nReference comparison: {ref_path.name}")
        with rasterio.open(ref_path) as ref:
            ref_sample = [b for b in sample if b < ref.count]
            ref_data = _read_overview(ref, ref_sample, overview_level)

        print(f"  {'Band':>6}  {'Target mean':>12}  {'Ref mean':>12}  {'Ratio':>8}")
        for idx, bi in enumerate(sample):
            if bi >= data.shape[0] or bi >= ref_data.shape[0]:
                continue
            t_mean = float(np.nanmean(data[idx]))
            r_mean = float(np.nanmean(ref_data[idx]))
            ratio  = t_mean / r_mean if r_mean != 0 else float("nan")
            flag   = "  ***" if not (0.1 < abs(ratio) < 10) else ""
            print(f"  {bi:>6}  {t_mean:>12.4f}  {r_mean:>12.4f}  {ratio:>8.3f}{flag}")

    print(f"\n{'='*60}")
    print(f"Result: {'PASS' if ok else 'FAIL  (see warnings above)'}")
    print(f"{'='*60}\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tif", type=Path)
    parser.add_argument("--ref", type=Path, default=None,
                        help="Reference TIF for band-statistics comparison")
    parser.add_argument("--overview-level", type=int, default=4,
                        help="COG overview level to use for coverage scan (default 4 = 1/16)")
    args = parser.parse_args()

    ok = validate(args.tif, args.ref, args.overview_level)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
