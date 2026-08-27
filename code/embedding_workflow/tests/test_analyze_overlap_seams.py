"""Regression test for retained-chip seam diagnostics."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_smooth_field_has_no_seam_excess(tmp_path: Path):
    raster = tmp_path / "smooth.tif"
    output = tmp_path / "seams.json"
    y, x = np.mgrid[:32, :32]
    data = np.stack([(x + y + band).astype("float32") for band in range(4)])
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="float32",
        crs="EPSG:32619",
        transform=from_origin(0, 320, 10, 10),
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(data)
        dst.update_tags(retained_center_pixels="8", patch_pixels="2")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "analyze_overlap_seams.py"),
            "--raster",
            str(raster),
            "--output",
            str(output),
            "--band-chunk",
            "2",
        ],
        check=True,
    )
    result = json.loads(output.read_text())

    assert result["keep_tokens"] == 4
    np.testing.assert_allclose(
        result["horizontal"]["seam_to_interior_mean_ratio"], 1.0
    )
    np.testing.assert_allclose(
        result["vertical"]["seam_to_interior_mean_ratio"], 1.0
    )
