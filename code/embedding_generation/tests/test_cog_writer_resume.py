"""Plain assert-based tests for write_cog()'s resumability (no pytest in this
repo). A Slurm walltime kill during the compression pass has no per-chip
checkpoint of its own (unlike inference) -- these verify a `.cog_ckpt`
sidecar lets a killed write resume instead of re-writing the whole array from
scratch, and that a corrupted resume state self-heals rather than looping
forever.

Run: uv run --python 3.11 python tests/test_cog_writer_resume.py
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.transform import from_origin

from utils.cog_writer import write_cog


def _make_test_array(n_bands=2, height=600, width=64):
    # height=600 spans more than one internal block (_BLOCK_ROWS=512), so
    # a normal write already exercises the periodic close/reopen cycle.
    rng = np.random.default_rng(7)
    return rng.standard_normal((n_bands, height, width)).astype("float32")


def test_write_cog_spans_multiple_block_checkpoints():
    """A normal (uninterrupted) write taller than one internal block must
    still round-trip correctly across the close/reopen checkpoint boundary."""
    arr = _make_test_array()
    transform = from_origin(0.0, 6000.0, 10, 10)
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "spans.tif"
        write_cog(arr, transform, "EPSG:32610", path, overviews=False)
        with rasterio.open(path) as src:
            written = src.read()
            assert np.allclose(arr, written, atol=1e-5), \
                "values differ across the close/reopen checkpoint boundary"
        print("test_write_cog_spans_multiple_block_checkpoints: PASS")
    finally:
        shutil.rmtree(tmp)


def test_write_cog_resumes_from_checkpoint():
    """Simulate a kill partway through: a valid tmp.tif with some rows
    already flushed plus a matching checkpoint. Resuming should pick up
    from the checkpointed row and produce the fully correct final array."""
    arr = _make_test_array()
    transform = from_origin(0.0, 6000.0, 10, 10)
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "resume.tif"
        tmp_path = path.with_suffix(".tmp.tif")
        ckpt_path = path.with_suffix(".cog_ckpt")

        n_bands, height, width = arr.shape
        resume_at = 512  # exactly one full block, matching _BLOCK_ROWS
        with rasterio.open(
            tmp_path, "w", driver="GTiff", dtype="float32", count=n_bands,
            height=height, width=width, crs="EPSG:32610", transform=transform,
            nodata=np.nan, tiled=True, blockxsize=512, blockysize=512,
            compress="zstd", predictor=3, zstd_level=1,
        ) as dst:
            dst.write(arr[:, :resume_at, :], window=rasterio.windows.Window(0, 0, width, resume_at))
        ckpt_path.write_text(str(resume_at))

        write_cog(arr, transform, "EPSG:32610", path, overviews=False)

        assert path.exists(), "merged output was not written"
        assert not tmp_path.exists() and not ckpt_path.exists(), \
            "tmp file and checkpoint should be cleaned up after a successful write"
        with rasterio.open(path) as src:
            written = src.read()
            assert np.allclose(arr, written, atol=1e-5), \
                "resumed write does not match source array"
        print("test_write_cog_resumes_from_checkpoint: PASS")
    finally:
        shutil.rmtree(tmp)


def test_write_cog_recovers_from_corrupt_checkpoint():
    """A checkpoint pointing at a truncated/unopenable tmp file (e.g. killed
    mid-write, before any block was ever flushed) should trigger a fresh
    restart instead of failing forever on the same unreadable tmp file."""
    arr = _make_test_array()
    transform = from_origin(0.0, 6000.0, 10, 10)
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "corrupt.tif"
        tmp_path = path.with_suffix(".tmp.tif")
        ckpt_path = path.with_suffix(".cog_ckpt")
        tmp_path.touch()  # 0 bytes -- not a valid GeoTIFF
        ckpt_path.write_text("64")  # plausible-looking, but tmp can't back it up

        write_cog(arr, transform, "EPSG:32610", path, overviews=False)

        assert path.exists()
        assert not tmp_path.exists() and not ckpt_path.exists()
        with rasterio.open(path) as src:
            written = src.read()
            assert np.allclose(arr, written, atol=1e-5)
        print("test_write_cog_recovers_from_corrupt_checkpoint: PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_write_cog_spans_multiple_block_checkpoints()
    test_write_cog_resumes_from_checkpoint()
    test_write_cog_recovers_from_corrupt_checkpoint()
    print("ALL PASSED")
