"""Write multi-band numpy arrays as Cloud-Optimized GeoTIFFs."""
import time

import numpy as np
import rasterio
import rasterio.windows
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from pathlib import Path

# Resuming a partially-written file has been observed to intermittently hit a
# GDAL/libtiff read failure the moment a *different* process writes into a
# block the original session never touched (not reliably reproducible in
# isolation -- looks like a transient close-to-open consistency hiccup rather
# than a hard incompatibility). Module level (not function-local) so tests
# can shrink the delay instead of actually sleeping.
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_S = 5

# Minimum real time between checkpoints (close + reopen), regardless of how
# many block boundaries have passed. Module level so tests can force it to 0
# to exercise the intermediate-checkpoint path deterministically instead of
# waiting on a wall-clock timer.
_MIN_CHECKPOINT_INTERVAL_S = 300


def write_cog(
    arr: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    path: Path,
    band_names: list[str] | None = None,
    compress: str = "zstd",
    nodata: float | None = np.nan,
    overviews: bool = True,
) -> None:
    """Write (C, H, W) float32 array to a tiled GeoTIFF, optionally with COG overviews.

    Resumable via a `.cog_ckpt` sidecar recording the next unwritten row: the
    strip-write loop has no per-chip checkpoint of its own (unlike inference),
    so a Slurm walltime kill mid-write previously meant re-writing and
    re-compressing the entire array from scratch on every retry -- for large,
    high-band-count states this alone can exceed the walltime, an infinite
    retry loop that never finishes. Resuming skips straight to the next
    unwritten strip instead.

    Args:
        arr: (C, H, W) numpy array.
        transform: Affine geotransform.
        crs: Coordinate reference system (EPSG string or rasterio CRS).
        path: Output file path.
        band_names: Optional list of band name strings for metadata tags.
        compress: Compression codec (zstd, lzw, deflate). Defaults to zstd
            with a floating-point predictor (see _compression_options below):
            LZW is a dictionary/byte-pattern codec built for repeated values
            (8-bit categorical imagery) and is both slower and a worse ratio
            than zstd+predictor on continuous float32 embedding data, which
            has little byte-level repetition for LZW to exploit.
        nodata: Nodata value; use np.nan for float data.
        overviews: Build overview pyramids and copy as COG. Set False for
            high-band-count arrays (e.g. embeddings) where overview
            generation across hundreds of bands would take hours.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(crs, str):
        crs = CRS.from_string(crs)

    n_bands, height, width = arr.shape

    # Rows per write strip. Keeps peak RAM at strip_rows × width × n_bands × 4 B,
    # which is a few GB even for 1024-dim Clay embeddings over Texas.
    # Materialising the full array at once OOMs on large states (LA: 137 GB,
    # TX: ~1 TB) because arr may be a disk-backed memmap.
    _STRIP_ROWS = 64
    _BLOCK_ROWS = 512

    # rasterio's DatasetWriter has no public flush() (checked: rasterio 1.3.11
    # only exposes close()/closed) -- and strips are narrower than the file's
    # own 512-row tile blocks, so GDAL holds several strips in its dirty block
    # cache before a block is complete enough to write out. Checkpointing
    # after every strip without forcing a real flush would let a kill leave
    # the checkpoint claiming rows are safe when they're still only in GDAL's
    # cache, silently corrupting the resumed output. Instead, close (which
    # does force a full flush) and reopen -- but only at a block boundary
    # (for a clean flush) *and* only after _MIN_CHECKPOINT_INTERVAL_S has
    # elapsed since the last one. A fixed row-count interval doesn't scale:
    # 512 rows is instant for a small/fast state (measured close+reopen
    # overhead alone was enough to erase zstd's speed edge over lzw on a
    # 2048-row test array) but is a small fraction of a multi-hour write for
    # a huge one. Gating on elapsed time means fast writes get ~0 extra
    # close/reopen cycles while slow ones still get checkpointed often enough
    # to bound how much a kill has to redo.
    assert _BLOCK_ROWS % _STRIP_ROWS == 0

    # Write to a temporary in-memory file first, then copy as COG.
    # The copy step reorganises internal tiling and adds overviews.
    tmp_path = path.with_suffix(".tmp.tif")
    ckpt_path = path.with_suffix(".cog_ckpt")

    resuming = tmp_path.exists() and ckpt_path.exists()
    resume_row = 0
    dst = None
    if resuming:
        try:
            resume_row = int(ckpt_path.read_text().strip())
            reopened = rasterio.open(tmp_path, "r+")
            if (reopened.height, reopened.width, reopened.count) == (height, width, n_bands):
                dst = reopened
            else:
                reopened.close()
        except Exception as exc:
            # A prior run (e.g. killed mid-write) can leave a truncated/corrupt
            # tmp file or an unparseable checkpoint. Without this, every retry
            # re-hits the same failure forever instead of starting fresh.
            print(f"      COG resume checkpoint unreadable ({exc}) — deleting and starting fresh")

    if dst is not None:
        print(f"      Resuming COG write: {resume_row}/{height} rows already written")
    else:
        tmp_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        resume_row = 0
        dst = rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_bands,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=compress,
            nodata=nodata,
            tiled=True,
            blockxsize=512,
            blockysize=_BLOCK_ROWS,
            BIGTIFF="IF_SAFER",
            **_compression_options(compress),
        )
        if band_names:
            for i, name in enumerate(band_names, 1):
                dst.update_tags(i, name=name)

    write_error: Exception | None = None
    last_ckpt_row = resume_row
    last_ckpt_time = time.monotonic()
    row_start = resume_row
    try:
        while row_start < height:
            row_end = min(row_start + _STRIP_ROWS, height)
            strip = np.array(arr[:, row_start:row_end, :], dtype="float32")
            win = rasterio.windows.Window(0, row_start, width, row_end - row_start)

            for attempt in range(1, _WRITE_RETRIES + 2):
                try:
                    dst.write(strip, window=win)
                    break
                except Exception as exc:
                    if attempt > _WRITE_RETRIES:
                        raise
                    print(f"      COG strip write failed ({exc}) — retrying "
                          f"({attempt}/{_WRITE_RETRIES}) after reopening")
                    if not dst.closed:
                        dst.close()
                    time.sleep(_WRITE_RETRY_DELAY_S)
                    dst = rasterio.open(tmp_path, "r+")

            at_block_boundary = row_end - last_ckpt_row >= _BLOCK_ROWS
            due_for_checkpoint = time.monotonic() - last_ckpt_time >= _MIN_CHECKPOINT_INTERVAL_S
            if row_end == height or (at_block_boundary and due_for_checkpoint):
                dst.close()  # forces GDAL to flush its dirty block cache
                ckpt_path.write_text(str(row_end))
                last_ckpt_row = row_end
                last_ckpt_time = time.monotonic()
                if row_end < height:
                    dst = rasterio.open(tmp_path, "r+")
            row_start = row_end
    except Exception as exc:
        write_error = exc
    finally:
        if dst is not None and not dst.closed:
            dst.close()

    if write_error is not None:
        # The tmp file is likely corrupted (e.g. a block was partially written
        # when a previous run was killed mid-write). Delete both so the next
        # run starts fresh rather than hitting the same corrupt block again.
        tmp_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"COG write failed — deleted corrupted tmp+checkpoint so next run starts fresh. "
            f"Cause: {write_error}"
        )

    ckpt_path.unlink(missing_ok=True)
    try:
        if overviews:
            _add_overviews_and_copy_as_cog(tmp_path, path, compress)
        else:
            tmp_path.rename(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _compression_options(compress: str) -> dict:
    """Extra GDAL creation options for a given codec, tuned for continuous
    float32 embedding data rather than 8-bit categorical imagery.

    predictor=3 (floating-point prediction) differences neighboring pixel
    values before compression -- it applies to LZW, DEFLATE, and ZSTD alike
    and improves both speed and ratio for continuous data. zstd_level=1
    trades ratio for speed: embeddings are high-entropy floats where higher
    zstd levels buy little extra compression for much more CPU time.
    """
    opts = {"predictor": 3}
    if compress == "zstd":
        opts["zstd_level"] = 1
    return opts


def _add_overviews_and_copy_as_cog(src_path: Path, dst_path: Path, compress: str) -> None:
    """Add internal overviews then re-write as a proper COG."""
    overview_levels = [2, 4, 8, 16, 32]

    with rasterio.open(src_path, "r+") as src:
        src.build_overviews(overview_levels, Resampling.average)
        src.update_tags(ns="rio_overview", resampling="average")

    from rasterio.shutil import copy as rio_copy
    rio_copy(
        src_path,
        dst_path,
        driver="GTiff",
        compress=compress,
        copy_src_overviews=True,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
        **_compression_options(compress),
    )
