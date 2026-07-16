"""Shared tile-mosaic helper.

Merges per-tile GeoTIFFs into one output file via windowed writes (no
full-scene RAM allocation), resumable via a `.merge_ckpt` sidecar that
records which tiles are already written — an interrupted merge (e.g. a
Slurm timeout on a huge state) resumes instead of restarting.

Used by composite.py (composite tiles) and embed.py (embedding tiles).
"""
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows

# Intermittent GDAL/libtiff read failures and vanishing tmp files have been
# observed on this filesystem when reopening/finalising very large merge
# outputs (not reliably reproducible in isolation -- looks like a transient
# close-to-open consistency hiccup, the same class of issue found in
# write_cog()). Retrying costs seconds; failing outright throws away a
# multi-hour merge, so both are worth a bounded retry before giving up.
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_S = 5
_RENAME_RETRIES = 3
_RENAME_RETRY_DELAY_S = 10


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
    dst = None
    if resuming:
        try:
            already_merged = set(ckpt_path.read_text().splitlines())
            dst = rasterio.open(tmp_path, "r+")
        except Exception as exc:
            # A prior run (e.g. OOM-killed mid-write) can leave a truncated/corrupt
            # tmp file that will never open. Without this, every retry re-hits the
            # same open failure forever instead of starting a fresh merge.
            print(f"      Resume checkpoint unreadable ({exc}) — deleting and starting fresh")
            resuming = False
            already_merged = set()

    if dst is not None:
        print(f"      Resuming merge: {len(already_merged)}/{len(tile_paths)} tiles already written")
    else:
        tmp_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        dst = rasterio.open(tmp_path, "w", **profile)
        if band_names:
            for i, band_name in enumerate(band_names, 1):
                dst.set_band_description(i, band_name)

    write_error: Exception | None = None
    try:
        for idx, ds in enumerate(datasets):
            tile_key = str(tile_paths[idx])
            if tile_key in already_merged:
                ds.close()
                continue
            tile_window = rasterio.windows.from_bounds(
                ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top,
                transform=out_transform,
            ).round_offsets().round_lengths()

            tile_error: Exception | None = None
            for attempt in range(1, _WRITE_RETRIES + 2):
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
                    tile_error = None
                    break
                except Exception as exc:
                    tile_error = exc
                    if attempt > _WRITE_RETRIES:
                        break
                    print(f"      Tile write failed ({exc}) — retrying "
                          f"({attempt}/{_WRITE_RETRIES}) after reopening")
                    if not dst.closed:
                        dst.close()
                    time.sleep(_WRITE_RETRY_DELAY_S)
                    dst = rasterio.open(tmp_path, "r+")

            if tile_error is not None:
                ds.close()
                write_error = tile_error
                break

            ds.close()
            already_merged.add(tile_key)
            ckpt_path.write_text("\n".join(sorted(already_merged)))
            # Close and reopen so a checkpointed tile is actually durable on
            # disk, not just sitting in GDAL's dirty block cache -- otherwise
            # a kill right after this checkpoint could leave a later resume
            # picking up from a tile whose data was never truly flushed.
            dst.close()
            if len(already_merged) < len(tile_paths):
                dst = rasterio.open(tmp_path, "r+")
    except Exception as exc:
        write_error = exc
    finally:
        if dst is not None and not dst.closed:
            dst.close()

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

    for attempt in range(1, _RENAME_RETRIES + 2):
        try:
            tmp_path.rename(out_path)
            break
        except FileNotFoundError:
            if attempt > _RENAME_RETRIES:
                raise
            print(f"      tmp file missing at rename time (attempt {attempt}/{_RENAME_RETRIES}) "
                  f"— possible transient filesystem hiccup, retrying after a pause")
            time.sleep(_RENAME_RETRY_DELAY_S)

    ckpt_path.unlink(missing_ok=True)
    for p in tile_paths:
        p.unlink()
        print(f"      Removed tile: {p.name}")
