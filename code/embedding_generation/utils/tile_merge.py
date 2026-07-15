"""Shared tile-mosaic helper.

Merges per-tile GeoTIFFs into one output file via windowed writes (no
full-scene RAM allocation), resumable via a `.merge_ckpt` sidecar that
records which tiles are already written — an interrupted merge (e.g. a
Slurm timeout on a huge state) resumes instead of restarting.

Used by composite.py (composite tiles) and embed.py (embedding tiles).
"""
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows


def merge_tiles(tile_paths: list[Path], band_names: list[str], out_path: Path) -> None:
    """Mosaic tile TIFs via windowed writes — no full-scene RAM allocation."""
    datasets = [rasterio.open(p) for p in tile_paths]

    out_left   = min(ds.bounds.left   for ds in datasets)
    out_bottom = min(ds.bounds.bottom for ds in datasets)
    out_right  = max(ds.bounds.right  for ds in datasets)
    out_top    = max(ds.bounds.top    for ds in datasets)

    res_x, res_y = datasets[0].res
    out_transform = rasterio.transform.from_origin(out_left, out_top, res_x, res_y)
    out_width  = round((out_right  - out_left)   / res_x)
    out_height = round((out_top    - out_bottom) / res_y)

    profile = datasets[0].profile.copy()
    profile.update(
        height=out_height, width=out_width, transform=out_transform,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        nodata=np.nan, BIGTIFF="YES",
    )

    tmp_path = out_path.with_suffix(".tmp.tif")
    ckpt_path = out_path.with_suffix(".merge_ckpt")

    # Resume an interrupted merge if both tmp file and checkpoint exist.
    resuming = tmp_path.exists() and ckpt_path.exists()
    already_merged: set[str] = set()
    dst_ctx = None
    if resuming:
        try:
            already_merged = set(ckpt_path.read_text().splitlines())
            dst_ctx = rasterio.open(tmp_path, "r+")
        except Exception as exc:
            # A prior run (e.g. OOM-killed mid-write) can leave a truncated/corrupt
            # tmp file that will never open. Without this, every retry re-hits the
            # same open failure forever instead of starting a fresh merge.
            print(f"      Resume checkpoint unreadable ({exc}) — deleting and starting fresh")
            resuming = False
            already_merged = set()

    if dst_ctx is not None:
        print(f"      Resuming merge: {len(already_merged)}/{len(tile_paths)} tiles already written")
    else:
        tmp_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        dst_ctx = rasterio.open(tmp_path, "w", **profile)

    write_error: Exception | None = None
    with dst_ctx as dst:
        if not resuming:
            for i, band_name in enumerate(band_names, 1):
                dst.set_band_description(i, band_name)
        for idx, ds in enumerate(datasets):
            tile_key = str(tile_paths[idx])
            if tile_key in already_merged:
                ds.close()
                continue
            tile_window = rasterio.windows.from_bounds(
                ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top,
                transform=out_transform,
            ).round_offsets().round_lengths()
            try:
                # Read one native block at a time (all bands together), not
                # band-by-band. These embeddings run to hundreds of bands and
                # are stored pixel-interleaved, so a block's bands are only
                # ever decompressed together — reading band-by-band forces
                # GDAL to re-decompress the same blocks once per band (its
                # cache is far smaller than a tile's full decompressed size),
                # which is what was driving merges past the 8h walltime.
                for _, block_window in ds.block_windows(1):
                    data = ds.read(window=block_window)
                    dest_window = rasterio.windows.Window(
                        col_off=tile_window.col_off + block_window.col_off,
                        row_off=tile_window.row_off + block_window.row_off,
                        width=block_window.width,
                        height=block_window.height,
                    )
                    dst.write(data, window=dest_window)
            except Exception as exc:
                ds.close()
                write_error = exc
                break
            ds.close()
            already_merged.add(tile_key)
            ckpt_path.write_text("\n".join(sorted(already_merged)))

    if write_error is not None:
        # The tmp file is likely corrupted (e.g. a block was partially written when
        # a previous job was killed mid-write). Delete both so the next run starts
        # fresh rather than hitting the same corrupt block again.
        tmp_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Tile write failed — deleted corrupted tmp+checkpoint so next run starts fresh. "
            f"Cause: {write_error}"
        )

    tmp_path.rename(out_path)
    ckpt_path.unlink(missing_ok=True)
    for p in tile_paths:
        p.unlink()
        print(f"      Removed tile: {p.name}")
