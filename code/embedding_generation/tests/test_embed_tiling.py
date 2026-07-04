"""Plain assert-based tests for embed.py's tiling helpers (no pytest in this repo).

Run: uv run --python 3.11 python tests/test_embed_tiling.py
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from embed import tile_row_bounds, iter_chips


def test_tile_row_bounds_partitions_evenly():
    # 12 rows split into 3 tiles -> 4 rows each, no gaps/overlaps.
    bounds = [tile_row_bounds(12, i, 3) for i in range(3)]
    assert bounds == [(0, 4), (4, 8), (8, 12)], bounds
    print("test_tile_row_bounds_partitions_evenly: PASS")


def test_tile_row_bounds_distributes_remainder():
    # 10 rows split into 3 tiles -> remainder rows go to the first tiles.
    bounds = [tile_row_bounds(10, i, 3) for i in range(3)]
    assert bounds == [(0, 4), (4, 7), (7, 10)], bounds
    total_rows = sum(end - start for start, end in bounds)
    assert total_rows == 10
    # No gaps or overlaps: each tile's end == next tile's start.
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]
    print("test_tile_row_bounds_distributes_remainder: PASS")


def test_tile_row_bounds_single_tile_covers_everything():
    assert tile_row_bounds(37, 0, 1) == (0, 37)
    print("test_tile_row_bounds_single_tile_covers_everything: PASS")


def test_tile_row_bounds_rejects_out_of_range():
    try:
        tile_row_bounds(10, 3, 3)
        raise AssertionError("expected ValueError for tile_index >= num_tiles")
    except ValueError:
        pass
    try:
        tile_row_bounds(10, 0, 20)
        raise AssertionError("expected ValueError for num_tiles > n_row_chips")
    except ValueError:
        pass
    print("test_tile_row_bounds_rejects_out_of_range: PASS")


def test_iter_chips_row_bounds_restricts_iteration():
    # 4x4 chip grid of 2x2-px chips = 8x8 px raster.
    arr = np.arange(64, dtype="float32").reshape(1, 8, 8)
    transform = from_origin(0.0, 8.0, 1, 1)
    profile = dict(driver="GTiff", dtype="float32", count=1, height=8, width=8,
                    crs="EPSG:32610", transform=transform)
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mktemp(suffix=".tif"))
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr)
        with rasterio.open(tmp) as src:
            # Full raster: 4x4 = 16 chips.
            all_chips = list(iter_chips(src, chip_px=2))
            assert len(all_chips) == 16, len(all_chips)

            # Restrict to rows [2, 4) (pixel rows, i.e. chip-row 1 of the 4-row grid)
            # -> should yield exactly the 4 chips in that row (row_off == 2), covering
            # all 4 columns (width=8, chip_px=2 -> col_off in {0, 2, 4, 6}).
            restricted = list(iter_chips(src, chip_px=2, row_px_bounds=(2, 4)))
            assert len(restricted) == 4, len(restricted)
            assert all(row_off == 2 for row_off, col_off, win, data in restricted)
            cols = sorted(col_off for row_off, col_off, win, data in restricted)
            assert cols == [0, 2, 4, 6], cols
    finally:
        tmp.unlink(missing_ok=True)
    print("test_iter_chips_row_bounds_restricts_iteration: PASS")


if __name__ == "__main__":
    test_tile_row_bounds_partitions_evenly()
    test_tile_row_bounds_distributes_remainder()
    test_tile_row_bounds_single_tile_covers_everything()
    test_tile_row_bounds_rejects_out_of_range()
    test_iter_chips_row_bounds_restricts_iteration()
    print("ALL PASSED")
