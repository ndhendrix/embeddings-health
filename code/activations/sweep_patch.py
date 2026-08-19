"""
Track ONE patch of ground as it moves from one edge of the chip to the other.

Runs alongside extract_activations.py -- imports the model, hook and forward-pass
helpers from it unchanged, and adds only what that script cannot do: offset the
chip by a whole number of patches.

For each of GRID positions:
  slide the chip one patch (PATCH px) west, so the target ground sits one patch
  further east inside the chip, run the encoder, and keep that one patch's
  embedding.

Output
  patch_sweep.npz          raw embeddings, (n_positions, n_features)
  patch_sweep_raw.png      one row per position, single colour scale
  patch_sweep_bycol.png    same data, each feature column scaled on its own
"""

import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

from extract_activations import (
    LAT, LON, DATE, SIZE, PATCH,
    load_model, find_encoder_blocks, register_hooks, run_model, remove_hooks,
)
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy


# ---------------------------------------------------------------- config

LAYER = 3        # which encoder block, 0-indexed. nano has 4, so 0..3.
GRID = SIZE // PATCH          # patches per chip side -> number of positions
SAVE_DTYPE = np.float32


# ---------------------------------------------------------------- data

def load_region(lat, lon, height, width, row_off, col_off):
    """
    Read a height x width pixel region from the .SAFE scene, positioned so that
    the pixel at (lat, lon) lands at local offset (row_off, col_off).

    Same band-order iteration and WarpedVRT resampling as load_tile in
    extract_activations.py -- the only difference is that the window is
    rectangular and its placement is given explicitly rather than centred, so
    chips can be sliced out of it at controlled offsets.
    """
    fnames = []
    for band_name in Modality.SENTINEL2_L2A.band_order:
        matches = glob.glob(f"*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2")
        if not matches:
            raise FileNotFoundError(
                f"No file for band {band_name}. Is the .SAFE folder in this dir?"
            )
        fnames.append(matches[0])

    with rasterio.open(fnames[0]) as src:
        crs, transform = src.crs, src.transform
        scene_w, scene_h = src.width, src.height
        xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
        row_t, col_t = src.index(xs[0], ys[0])

    row0, col0 = row_t - row_off, col_t - col_off
    if not (0 <= row0 and row0 + height <= scene_h
            and 0 <= col0 and col0 + width <= scene_w):
        raise ValueError(
            f"The {height}x{width} px region needed for this sweep falls outside "
            "the scene. Pick a location further from the scene edge."
        )
    window = rasterio.windows.Window(col0, row0, width, height)

    image = np.zeros((len(fnames), height, width), dtype=np.int32)
    for i, fname in enumerate(fnames):
        with rasterio.open(fname) as src:
            with WarpedVRT(src, crs=crs, transform=transform,
                           width=scene_w, height=scene_h,
                           resampling=Resampling.bilinear) as vrt:
                image[i] = vrt.read(1, window=window)

    image = image.transpose(1, 2, 0)[None, :, :, None, :]
    print(f"region loaded, raw shape {image.shape}")
    return image, (row0, col0), (row_t, col_t)


def normalize_region(image):
    """
    Normalise the whole region ONCE, before any chip is sliced from it.

    extract_activations.py normalises each tile on its own, which is correct
    when tiles are independent. Here 32 overlapping chips share ground, and if
    Strategy.COMPUTED derives any statistic from the array it is given, the same
    ground pixel would enter the model as a different number in each chip. That
    difference would appear in the heatmap and look exactly like an edge effect.
    Normalising the region first rules it out; if the strategy uses fixed
    dataset constants this is identical to per-chip normalisation anyway.
    """
    return Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, image)


# ---------------------------------------------------------------- sweep

def sweep(model, device, region, block_names, store):
    """
    Returns (GRID, n_features): row k is the target patch's embedding from the
    run where it sat at local patch column k.

    Geometry. The region is laid out so the target pixel is at local row
    SIZE//2 and local column PATCH*(GRID-1). Chip k starts at local column
    PATCH*(GRID-1-k), which puts the target pixel at local column PATCH*k inside
    that chip -- i.e. local patch column k, for k = 0 .. GRID-1.

    The row never changes: the target stays at local patch row GRID//2 in every
    run. Only the column varies, which is what makes this a clean single-axis
    sweep.
    """
    layer_name = block_names[LAYER]
    patch_row = (SIZE // 2) // PATCH
    rows = None

    for k in range(GRID):
        col_start = PATCH * (GRID - 1 - k)
        chip = region[:, :, col_start:col_start + SIZE, :, :]

        store.clear()
        run_model(model, device, chip, DATE)

        tok = store[layer_name].squeeze(0).numpy()     # (n_tokens, n_features)
        if tok.shape[0] != GRID * GRID:
            raise ValueError(
                f"{tok.shape[0]} tokens but expected {GRID*GRID} for a "
                f"{GRID}x{GRID} patch grid. Token layout is not what this "
                "script assumes."
            )
        # Tokens are laid out in reading order across the patch grid, so the
        # patch at (row, col) is token row*GRID + col.
        idx = patch_row * GRID + k

        if rows is None:
            rows = np.zeros((GRID, tok.shape[1]), dtype=np.float32)
        rows[k] = tok[idx]

    print(f"swept {GRID} positions, embedding width {rows.shape[1]}")
    return rows


def sanity_check(rows):
    """
    Neighbouring positions should be similar but not identical: the ground is
    the same, only the surroundings the model could see have changed. A cosine
    of exactly 1.0 everywhere would mean the chip never actually moved; a very
    low cosine would mean the wrong token is being pulled out.
    """
    a, b = rows[:-1], rows[1:]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                             * np.linalg.norm(b, axis=-1) + 1e-8)
    print(f"adjacent-position cosine: min {cos.min():.4f}, "
          f"max {cos.max():.4f}, mean {cos.mean():.4f}")
    return cos


# ---------------------------------------------------------------- figures

def plot_raw(rows, path="patch_sweep_raw.png"):
    """
    One colour scale across every feature. This is the honest picture, and it is
    usually hard to read: the 128 features sit at very different baselines, so
    the image is dominated by which feature a column is rather than by position.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(rows, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("patch position within chip")
    ax.set_yticks(range(0, rows.shape[0], 2))
    ax.set_title(f"Target patch embedding by position in chip, layer {LAYER}\n"
                 "raw values, one colour scale for all features")
    fig.colorbar(im, ax=ax, fraction=.02, pad=.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_by_column(rows, path="patch_sweep_bycol.png"):
    """
    Each feature scaled against its own range across the positions.

    This makes change DOWN a column visible, which is the effect being looked
    for. The cost is that a feature which barely moves gets stretched to full
    contrast too, so a dramatic-looking column may be amplified noise. The raw
    range of every column is saved to the npz so those can be identified.
    """
    lo = rows.min(axis=0, keepdims=True)
    hi = rows.max(axis=0, keepdims=True)
    norm = (rows - lo) / np.where(hi - lo == 0, 1.0, hi - lo)

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(norm, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("patch position within chip")
    ax.set_yticks(range(0, rows.shape[0], 2))
    ax.set_title(f"Target patch embedding by position in chip, layer {LAYER}\n"
                 "NORMALISED BY COLUMN: each feature scaled to its own min/max")
    fig.colorbar(im, ax=ax, fraction=.02, pad=.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return (hi - lo).ravel()


# ---------------------------------------------------------------- main

def main():
    # Region geometry, in pixels. Wide enough for the chip to slide GRID-1
    # patches; only as tall as the chip, since the sweep is along one axis.
    height = SIZE
    width = SIZE + PATCH * (GRID - 1)
    row_off = SIZE // 2
    col_off = PATCH * (GRID - 1)

    model, device = load_model()
    block_names = find_encoder_blocks(model)
    if not 0 <= LAYER < len(block_names):
        raise ValueError(
            f"LAYER={LAYER} but this model has {len(block_names)} blocks "
            f"(valid: 0..{len(block_names)-1})."
        )
    print(f"tracking layer {LAYER} ({block_names[LAYER]}) "
          f"across {GRID} positions")

    region, origin, target = load_region(LAT, LON, height, width,
                                         row_off, col_off)
    region = normalize_region(region)

    store = {}
    handles = register_hooks(model, block_names, store)
    try:
        rows = sweep(model, device, region, block_names, store)
    finally:
        remove_hooks(handles)

    cos = sanity_check(rows)

    plot_raw(rows)
    col_range = plot_by_column(rows)

    np.savez(
        "patch_sweep.npz",
        rows=rows.astype(SAVE_DTYPE),        # (position, feature)
        column_range=col_range,              # raw spread of each feature
        adjacent_cosine=cos,
        layer=LAYER, layer_name=block_names[LAYER],
        positions=np.arange(GRID),
        patch_row=(SIZE // 2) // PATCH,
        region_origin_rowcol=np.array(origin),
        target_pixel_rowcol=np.array(target),
        lat=LAT, lon=LON, date=np.array(DATE),
        chip=SIZE, patch=PATCH, grid=GRID,
    )
    print("saved patch_sweep.npz")


if __name__ == "__main__":
    main()