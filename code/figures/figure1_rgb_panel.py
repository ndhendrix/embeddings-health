"""
RGB embedding visualization for a geographic window.

Reads one or more multi-band embedding COGs, crops to a 24 km × 24 km square
centered on a given coordinate, fits PCA(3) on the valid pixels within that
window, and returns an RGB image where each channel is one principal component
scaled to [0, 1].

When multiple TIFs are provided they are mosaicked on-the-fly (e.g. DC + MD +
VA for a cross-state bounding box).  Files with a different CRS are reprojected
to match the first file's CRS via WarpedVRT.

Usage:
  python figure1_rgb_panel.py \
    --tif $SCRATCH/embeddings-health/olmoearth_nano_embeddings/CT/olmoearth_Nano_CT_2022.tif \
    --lat 41.5582 --lon -73.0515 \
    --label "Waterbury CT — OlmoEarth Nano" \
    --out $SCRATCH/embeddings-health/figures/waterbury_nano_rgb.png
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def make_rgb_panel(
    tif_paths,
    center_lat: float,
    center_lon: float,
    box_km: float = 24.0,
    max_bands: int | None = 64,
) -> tuple[np.ndarray, float]:
    """Return an (H, W, 3) uint8 RGB array and PCA explained variance (3 components).

    Parameters
    ----------
    tif_paths:
        A single path or a list of paths to multi-band float32 embedding COGs.
        When multiple files are given they are spatially mosaicked; files whose
        CRS differs from the first file are reprojected on-the-fly.
    center_lat, center_lon:
        WGS-84 centre of the desired window.
    box_km:
        Side length of the square window in km (default 24).
    max_bands:
        Maximum number of evenly spaced embedding dimensions to read. Pass
        ``None`` to use every dimension (recommended for final figures). The
        default of 64 preserves the historical, faster exploratory behavior.
    """
    if isinstance(tif_paths, (str, Path)):
        tif_paths = [Path(tif_paths)]
    else:
        tif_paths = [Path(p) for p in tif_paths]

    half_m = (box_km * 1000) / 2.0

    # Reference CRS and resolution from the first file
    with rasterio.open(tif_paths[0]) as ref:
        target_crs = ref.crs
        total_bands = ref.count
        res = abs(ref.transform.a)

    # Convert center to target CRS
    to_crs = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    cx, cy = to_crs.transform(center_lon, center_lat)

    left   = cx - half_m
    right  = cx + half_m
    bottom = cy - half_m
    top    = cy + half_m

    # Evenly-spaced band subset to cap random I/O (one seek per band-tile)
    if max_bands is not None and max_bands < 3:
        raise ValueError("max_bands must be at least 3 or None")
    if max_bands is not None and total_bands > max_bands:
        step = total_bands / max_bands
        indexes = [int(round(i * step)) + 1 for i in range(max_bands)]
    else:
        indexes = list(range(1, total_bands + 1))
    D = len(indexes)

    # Output canvas in target CRS pixels
    out_h = round((top - bottom) / res)
    out_w = round((right - left) / res)
    out = np.full((D, out_h, out_w), np.nan, dtype="float32")

    output_transform = from_origin(left, top, res, res)

    for tif_path in tif_paths:
        t0 = time.time()
        with rasterio.open(tif_path) as src:
            if src.crs != target_crs:
                warped_bounds = transform_bounds(
                    src.crs, target_crs, *src.bounds, densify_pts=21
                )
                if (
                    warped_bounds[2] <= left
                    or warped_bounds[0] >= right
                    or warped_bounds[3] <= bottom
                    or warped_bounds[1] >= top
                ):
                    continue

                # Reproject directly into the crop grid. WarpedVRT's implicit
                # whole-tile target grid caused pathological repeated reads for
                # cross-zone, band-interleaved Virginia tiles.
                chunk = np.full((D, out_h, out_w), np.nan, dtype="float32")
                reproject(
                    source=rasterio.band(src, indexes),
                    destination=chunk,
                    src_nodata=src.nodata,
                    dst_transform=output_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    num_threads=4,
                    warp_mem_limit=1024,
                    init_dest_nodata=True,
                )
                valid = np.isfinite(chunk)
                out[valid] = chunk[valid]
                print(
                    f"    {tif_path.name}: {chunk.shape} reprojected "
                    f"in {time.time()-t0:.1f}s",
                    flush=True,
                )
                continue

            fl = max(left, src.bounds.left)
            fr = min(right, src.bounds.right)
            fb = max(bottom, src.bounds.bottom)
            ft = min(top, src.bounds.top)

            if fl >= fr or fb >= ft:
                continue

            win = from_bounds(fl, fb, fr, ft, src.transform)
            chunk = src.read(indexes=indexes, window=win)  # (D, h, w)

        print(f"    {tif_path.name}: {chunk.shape} in {time.time()-t0:.1f}s", flush=True)

        # Pixel offset of this chunk within the output canvas
        dst_col = round((fl - left) / res)
        dst_row = round((top - ft) / res)
        ch, cw = chunk.shape[1], chunk.shape[2]
        er = min(dst_row + ch, out_h)
        ec = min(dst_col + cw, out_w)
        uh, uw = er - dst_row, ec - dst_col
        if uh <= 0 or uw <= 0:
            continue

        region = out[:, dst_row:er, dst_col:ec]
        piece = chunk[:, :uh, :uw]
        # AlphaEarth uses -inf as nodata, while the locally generated COGs use
        # NaN. Treat every non-finite value as missing so neither encoding can
        # leak into the PCA fit.
        valid = np.isfinite(piece)
        region[valid] = piece[valid]

    # ── PCA projection ──────────────────────────────────────────────────────
    t0 = time.time()
    flat = out.reshape(D, -1).T                 # (N, D)
    # Avoid allocating a second (N, D) boolean array for large 10 m crops.
    valid_flat = np.isfinite(out[0]).ravel()
    for band in range(1, D):
        valid_flat &= np.isfinite(out[band]).ravel()
    X          = flat[valid_flat]
    print(f"    mask+flatten: {time.time()-t0:.1f}s  valid_px={len(X)}", flush=True)

    if len(X) < 3:
        raise ValueError(f"PCA requires at least 3 valid pixels; found {len(X)}")

    t0    = time.time()
    rng   = np.random.default_rng(42)
    n_fit = min(50_000, len(X))
    idx   = rng.choice(len(X), n_fit, replace=False)
    pca   = PCA(n_components=3, random_state=42, svd_solver="randomized", iterated_power=2)
    pca.fit(X[idx])
    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA fit ({n_fit} samples): {time.time()-t0:.1f}s", flush=True)

    t0        = time.time()
    rgb_valid = MinMaxScaler().fit_transform(pca.transform(X))
    print(f"    transform+scale: {time.time()-t0:.1f}s", flush=True)

    rgb            = np.zeros((out_h * out_w, 3), dtype="float32")
    rgb[valid_flat] = rgb_valid
    rgb_image      = (rgb.reshape(out_h, out_w, 3) * 255).astype("uint8")
    return rgb_image, explained


def plot_panel(
    rgb_image: np.ndarray,
    label: str,
    out_path: Path | None = None,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(label, fontsize=10, pad=6)
    fig.tight_layout(pad=0.5)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tif",   required=True, help="Embedding COG path")
    parser.add_argument("--lat",   type=float, required=True, help="Centre latitude (WGS-84)")
    parser.add_argument("--lon",   type=float, required=True, help="Centre longitude (WGS-84)")
    parser.add_argument("--box-km", type=float, default=24.0, help="Window side length in km (default 24)")
    parser.add_argument("--label", default="", help="Panel title")
    parser.add_argument("--out",   type=Path, default=None, help="Output PNG path (omit to show interactively)")
    parser.add_argument("--dpi",   type=int, default=150)
    args = parser.parse_args()

    print(f"Reading {args.tif}")
    rgb, explained = make_rgb_panel(args.tif, args.lat, args.lon, args.box_km)
    print(f"  Image shape: {rgb.shape}  PCA explained variance: {explained:.1%}")

    label = args.label or Path(args.tif).stem
    plot_panel(rgb, label, out_path=args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
