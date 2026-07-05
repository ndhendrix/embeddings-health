"""Plain assert-based tests for write_cog()'s compression settings (no pytest
in this repo). Verifies the default codec is both correct (round-trips
float32 values and NaN nodata exactly) and faster than the legacy
LZW-without-predictor settings for continuous float32 data (embeddings),
per the rationale in cog_writer.py's write_cog() docstring.

Run: uv run --python 3.11 python tests/test_cog_writer_compression.py
"""
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.cog_writer import write_cog


def _make_test_array(n_bands=32, height=2048, width=2048):
    # Continuous float32 data, not integer/categorical -- behaves like real
    # model-embedding output for compression-timing purposes, unlike e.g.
    # zeros or a repeating pattern which would compress trivially fast under
    # any codec and hide the LZW-vs-zstd difference this test exists to catch.
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_bands, height, width)).astype("float32")


def test_written_values_round_trip_correctly():
    arr = _make_test_array(n_bands=4, height=256, width=256)
    arr[0, 0, 0] = np.nan  # exercise nodata handling
    transform = from_origin(0.0, 1000.0, 10, 10)
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "roundtrip.tif"
        write_cog(arr, transform, "EPSG:32610", path, overviews=False)
        with rasterio.open(path) as src:
            written = src.read()
            assert np.isnan(written[0, 0, 0]), "NaN nodata value was not preserved"
            valid = ~np.isnan(arr)
            assert np.allclose(arr[valid], written[valid], atol=1e-5), \
                "compressed round-trip values differ from source beyond float32 tolerance"
    finally:
        shutil.rmtree(tmp)
    print("test_written_values_round_trip_correctly: PASS")


def test_default_compression_is_faster_than_legacy_lzw():
    arr = _make_test_array()
    transform = from_origin(0.0, 1000.0, 10, 10)
    tmp = Path(tempfile.mkdtemp())
    try:
        legacy_path = tmp / "legacy_lzw.tif"
        new_path = tmp / "new_default.tif"

        start = time.monotonic()
        write_cog(arr, transform, "EPSG:32610", legacy_path,
                  compress="lzw", overviews=False)
        legacy_elapsed = time.monotonic() - start
        assert legacy_path.exists()

        start = time.monotonic()
        write_cog(arr, transform, "EPSG:32610", new_path, overviews=False)
        new_elapsed = time.monotonic() - start
        assert new_path.exists()

        print(f"legacy lzw: {legacy_elapsed:.1f}s   new default (zstd+predictor): {new_elapsed:.1f}s")
        assert new_elapsed < legacy_elapsed, (
            f"expected the new zstd+predictor default to be faster than legacy "
            f"lzw, got new={new_elapsed:.1f}s vs legacy={legacy_elapsed:.1f}s"
        )
    finally:
        shutil.rmtree(tmp)
    print("test_default_compression_is_faster_than_legacy_lzw: PASS")


if __name__ == "__main__":
    test_written_values_round_trip_correctly()
    test_default_compression_is_faster_than_legacy_lzw()
    print("ALL PASSED")
