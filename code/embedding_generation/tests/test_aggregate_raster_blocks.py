"""Regression test for blockwise sufficient-statistic aggregation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_raster_blocks import aggregate_block

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "embedding_workflow"))
from aggregate_tiles import reduce_partials


def check() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raster_path = root / "embeddings.tif"
        tracts_path = root / "tracts.geojson"

        base = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
        data = np.stack([base + 1000 * i for i in range(4)])
        data[:, 0, 0] = np.nan
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=32,
            height=16,
            count=4,
            dtype="float32",
            crs="EPSG:3857",
            transform=from_origin(0, 16, 1, 1),
            tiled=True,
            blockxsize=16,
            blockysize=16,
            nodata=np.nan,
        ) as dst:
            dst.write(data)

        tracts = gpd.GeoDataFrame(
            {"GEOID": ["00000000001", "00000000002"]},
            geometry=[box(0, 8, 32, 16), box(0, 0, 32, 8)],
            crs="EPSG:3857",
        )
        tracts.to_file(tracts_path, driver="GeoJSON")

        partials = []
        for block_col in range(2):
            partial = root / f"block_{block_col}.npz"
            aggregate_block(
                raster_path,
                tracts_path,
                partial,
                block_row=0,
                block_col=block_col,
                expected_dim=4,
                band_chunk=2,
            )
            partials.append(partial)

        output = root / "tracts.csv"
        qa_json = root / "tracts.validation.json"
        reduce_partials(
            partials,
            output,
            year=2022,
            prefix="PR",
            tracts_path=tracts_path,
            qa_json=qa_json,
        )

        actual = pd.read_csv(output, dtype={"GEOID": str}).set_index("GEOID")
        for geoid, row_slice in [
            ("00000000001", slice(0, 8)),
            ("00000000002", slice(8, 16)),
        ]:
            expected = data[:, row_slice, :].reshape(4, -1)
            expected = expected[:, np.isfinite(expected).all(axis=0)]
            assert actual.loc[geoid, "pixel_count"] == expected.shape[1]
            for dimension in range(4):
                prefix = f"PR{dimension:04d}"
                np.testing.assert_allclose(
                    actual.loc[geoid, f"{prefix}_MEAN"], expected[dimension].mean()
                )
                np.testing.assert_allclose(
                    actual.loc[geoid, f"{prefix}_MINIMUM"], expected[dimension].min()
                )
                np.testing.assert_allclose(
                    actual.loc[geoid, f"{prefix}_MAXIMUM"], expected[dimension].max()
                )
                np.testing.assert_allclose(
                    actual.loc[geoid, f"{prefix}_STD"], expected[dimension].std()
                )


if __name__ == "__main__":
    check()
    print("PASS: blockwise aggregation matches direct tract statistics")
