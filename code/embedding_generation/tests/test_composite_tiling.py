"""Plain assert-based tests for composite.py's tile-count planning (no pytest
in this repo).

Regression coverage for a bug where the Slurm submit script silently derived
its expected tile count from a land-area heuristic instead of the real tile
grid, permanently under-submitting tiles for large/coastal states (TX, CA)
whose bounding-box area is far larger than their land area. See
--print-tile-count in composite.py and get_tile_count() in
slurm/submit_olmoearth_composite_parallel.sh.

Run: uv run --python 3.11 python tests/test_composite_tiling.py
"""
import math
import subprocess
import sys
from pathlib import Path

from composite import STATE_BBOXES, bbox_to_utm_epsg, split_bbox_into_tiles

COMPOSITE_PY = Path(__file__).resolve().parent.parent / "composite.py"


def true_tile_count(state: str, max_tile_km: float) -> int:
    bbox = STATE_BBOXES[state]
    crs = bbox_to_utm_epsg(bbox)
    return len(split_bbox_into_tiles(bbox, crs, max_tile_km))


def old_area_estimate(area_km2: float) -> int:
    """The land-area heuristic the submit script used to fall back to."""
    return max(1, math.ceil(area_km2 / 10000 * 1.3))


def test_split_bbox_into_tiles_matches_expected_grid():
    # A ~250km x ~150km bbox at max_tile_km=100 must grid to 3x2 = 6 tiles.
    # Chosen with a ~50km safety margin either side of the 200/300km and
    # 100/200km breakpoints so UTM projection distortion can't flip the
    # ceil() result.
    bbox = (-100.0, 30.0, -97.39, 31.35)
    crs = bbox_to_utm_epsg(bbox)
    tiles = split_bbox_into_tiles(bbox, crs, max_tile_km=100)
    assert len(tiles) == 6, f"expected 6 tiles, got {len(tiles)}: {tiles}"
    lon_min = min(t[0] for t in tiles)
    lon_max = max(t[2] for t in tiles)
    lat_min = min(t[1] for t in tiles)
    lat_max = max(t[3] for t in tiles)
    assert math.isclose(lon_min, bbox[0], abs_tol=1e-6)
    assert math.isclose(lon_max, bbox[2], abs_tol=1e-6)
    assert math.isclose(lat_min, bbox[1], abs_tol=1e-6)
    assert math.isclose(lat_max, bbox[3], abs_tol=1e-6)
    print("test_split_bbox_into_tiles_matches_expected_grid: PASS")


def test_area_heuristic_undercounts_tx_and_ca():
    # Locks in the root cause: for large coastal/irregular states, the bbox
    # (which split_bbox_into_tiles grids) is much bigger than the land area
    # (which the old heuristic used), so the heuristic undercounts badly
    # enough to permanently drop coastal tiles from submission.
    tx_real = true_tile_count("TX", max_tile_km=100)
    tx_estimate = old_area_estimate(676587)
    assert tx_real > tx_estimate * 1.3, (
        f"TX: real={tx_real} estimate={tx_estimate} — expected the heuristic "
        f"to undercount by a wide margin"
    )

    ca_real = true_tile_count("CA", max_tile_km=100)
    ca_estimate = old_area_estimate(403466)
    assert ca_real > ca_estimate * 1.3, (
        f"CA: real={ca_real} estimate={ca_estimate} — expected the heuristic "
        f"to undercount by a wide margin"
    )
    print(f"test_area_heuristic_undercounts_tx_and_ca: PASS "
          f"(TX real={tx_real} vs estimate={tx_estimate}; "
          f"CA real={ca_real} vs estimate={ca_estimate})")


def test_print_tile_count_cli_matches_internal_count_with_no_network():
    # --print-tile-count must never construct a pystac_client (no network
    # call), and must agree exactly with split_bbox_into_tiles().
    for state in ("TX", "CA", "RI"):
        expected = true_tile_count(state, max_tile_km=100)
        result = subprocess.run(
            [sys.executable, str(COMPOSITE_PY), "--state", state,
             "--max-tile-km", "100", "--print-tile-count"],
            capture_output=True, text=True, timeout=30, cwd=COMPOSITE_PY.parent,
        )
        assert result.returncode == 0, (
            f"{state}: --print-tile-count exited {result.returncode}: {result.stderr}"
        )
        printed = int(result.stdout.strip())
        assert printed == expected, (
            f"{state}: CLI printed {printed}, split_bbox_into_tiles says {expected}"
        )
    print("test_print_tile_count_cli_matches_internal_count_with_no_network: PASS")


if __name__ == "__main__":
    test_split_bbox_into_tiles_matches_expected_grid()
    test_area_heuristic_undercounts_tx_and_ca()
    test_print_tile_count_cli_matches_internal_count_with_no_network()
    print("ALL PASSED")
