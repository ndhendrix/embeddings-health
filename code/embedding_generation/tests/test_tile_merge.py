"""Plain assert-based test for utils.tile_merge.merge_tiles (no pytest in this repo).

Run: uv run --python 3.11 python tests/test_tile_merge.py
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.tile_merge import merge_tiles


def make_tile(path, value, top, left, width, height, n_bands=2):
    transform = from_origin(left, top, 10, 10)
    profile = dict(
        driver="GTiff", dtype="float32", count=n_bands,
        height=height, width=width, crs="EPSG:32610", transform=transform,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(1, n_bands + 1):
            dst.write(np.full((height, width), value * b, dtype="float32"), indexes=b)


def test_merge_reproduces_expected_mosaic():
    tmp = Path(tempfile.mkdtemp())
    try:
        # Two vertically-stacked tiles: tile0 on top (higher y), tile1 below.
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"
        make_tile(tile0, value=1.0, top=100.0, left=0.0, width=4, height=4)
        make_tile(tile1, value=2.0, top=60.0,  left=0.0, width=4, height=4)

        out_path = tmp / "emb.tif"
        merge_tiles([tile0, tile1], ["B01", "B02"], out_path)

        assert out_path.exists(), "merged output was not written"
        assert not tile0.exists(), "tile0 should be deleted after merge"
        assert not tile1.exists(), "tile1 should be deleted after merge"

        with rasterio.open(out_path) as merged:
            assert merged.height == 8, f"expected height 8, got {merged.height}"
            assert merged.width == 4, f"expected width 4, got {merged.width}"
            band1 = merged.read(1)
            # Top 4 rows come from tile0 (value 1.0 * band1 = 1.0)
            assert np.allclose(band1[:4, :], 1.0), "top rows should be tile0's values"
            # Bottom 4 rows come from tile1 (value 2.0 * band1 = 2.0)
            assert np.allclose(band1[4:, :], 2.0), "bottom rows should be tile1's values"
            assert merged.descriptions == ("B01", "B02")

        print("test_merge_reproduces_expected_mosaic: PASS")
    finally:
        shutil.rmtree(tmp)


def test_merge_resumes_from_checkpoint():
    tmp = Path(tempfile.mkdtemp())
    try:
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"
        make_tile(tile0, value=1.0, top=100.0, left=0.0, width=4, height=4)
        make_tile(tile1, value=2.0, top=60.0,  left=0.0, width=4, height=4)
        out_path = tmp / "emb.tif"

        # Simulate a partial merge: tmp file + checkpoint recording tile0 done.
        tmp_path = out_path.with_suffix(".tmp.tif")
        ckpt_path = out_path.with_suffix(".merge_ckpt")
        transform = from_origin(0.0, 100.0, 10, 10)
        profile = dict(
            driver="GTiff", dtype="float32", count=2, height=8, width=4,
            crs="EPSG:32610", transform=transform, nodata=np.nan,
            compress="deflate", tiled=True, blockxsize=512, blockysize=512,
            BIGTIFF="YES",
        )
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(np.full((4, 4), 1.0, dtype="float32"), indexes=1, window=rasterio.windows.Window(0, 0, 4, 4))
            dst.write(np.full((4, 4), 2.0, dtype="float32"), indexes=2, window=rasterio.windows.Window(0, 0, 4, 4))
        ckpt_path.write_text(str(tile0))

        merge_tiles([tile0, tile1], ["B01", "B02"], out_path)

        assert out_path.exists()
        assert not tile1.exists(), "tile1 should be merged and deleted on resume"
        with rasterio.open(out_path) as merged:
            band1 = merged.read(1)
            assert np.allclose(band1[4:, :], 2.0), "resumed merge should have written tile1's rows"

        print("test_merge_resumes_from_checkpoint: PASS")
    finally:
        shutil.rmtree(tmp)


def test_merge_recovers_from_corrupt_checkpoint():
    """A truncated/corrupt tmp file (e.g. OOM-killed mid-write) should trigger a
    fresh restart instead of failing forever on the same unreadable tmp file."""
    tmp = Path(tempfile.mkdtemp())
    try:
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"
        make_tile(tile0, value=1.0, top=100.0, left=0.0, width=4, height=4)
        make_tile(tile1, value=2.0, top=60.0,  left=0.0, width=4, height=4)
        out_path = tmp / "emb.tif"

        # Simulate the corruption left behind by a process killed mid-write:
        # zero-byte tmp file, zero-byte checkpoint.
        tmp_path = out_path.with_suffix(".tmp.tif")
        ckpt_path = out_path.with_suffix(".merge_ckpt")
        tmp_path.touch()
        ckpt_path.touch()

        merge_tiles([tile0, tile1], ["B01", "B02"], out_path)

        assert out_path.exists(), "merge should have restarted fresh and completed"
        assert not tile0.exists() and not tile1.exists()
        with rasterio.open(out_path) as merged:
            band1 = merged.read(1)
            assert np.allclose(band1[:4, :], 1.0)
            assert np.allclose(band1[4:, :], 2.0)

        print("test_merge_recovers_from_corrupt_checkpoint: PASS")
    finally:
        shutil.rmtree(tmp)


def test_merge_handles_multiblock_multiband_tiles():
    """Regression test for the block-windowed read path: pixel-interleaved
    tiles spanning multiple internal blocks, with more than one band, must
    still assemble correctly (each block read once, all bands together)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"

        def make_tiled_tile(path, value, top, left, width, height, n_bands, blocksize):
            transform = from_origin(left, top, 10, 10)
            profile = dict(
                driver="GTiff", dtype="float32", count=n_bands,
                height=height, width=width, crs="EPSG:32610", transform=transform,
                nodata=np.nan, tiled=True, blockxsize=blocksize, blockysize=blocksize,
                interleave="pixel", compress="zstd",
            )
            with rasterio.open(path, "w", **profile) as dst:
                for b in range(1, n_bands + 1):
                    dst.write(np.full((height, width), value * b, dtype="float32"), indexes=b)

        # 32x32 pixels over 16x16 blocks (2x2 block grid, GTiff's minimum
        # block size) with 3 bands -- spans multiple blocks in both
        # dimensions and multiple bands per block.
        make_tiled_tile(tile0, value=1.0, top=400.0, left=0.0, width=32, height=32, n_bands=3, blocksize=16)
        make_tiled_tile(tile1, value=2.0, top=80.0,  left=0.0, width=32, height=32, n_bands=3, blocksize=16)

        out_path = tmp / "emb.tif"
        merge_tiles([tile0, tile1], ["B01", "B02", "B03"], out_path)

        assert out_path.exists(), "merged output was not written"
        assert not tile0.exists() and not tile1.exists()

        with rasterio.open(out_path) as merged:
            assert merged.height == 64, f"expected height 64, got {merged.height}"
            assert merged.width == 32, f"expected width 32, got {merged.width}"
            for b in range(1, 4):
                band = merged.read(b)
                assert np.allclose(band[:32, :], 1.0 * b), f"band {b} top rows should be tile0's values"
                assert np.allclose(band[32:, :], 2.0 * b), f"band {b} bottom rows should be tile1's values"

        print("test_merge_handles_multiblock_multiband_tiles: PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_merge_reproduces_expected_mosaic()
    test_merge_resumes_from_checkpoint()
    test_merge_recovers_from_corrupt_checkpoint()
    test_merge_handles_multiblock_multiband_tiles()
    print("ALL PASSED")
