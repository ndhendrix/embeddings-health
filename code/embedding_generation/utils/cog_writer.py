"""Write multi-band numpy arrays as Cloud-Optimized GeoTIFFs."""
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from pathlib import Path


def write_cog(
    arr: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    path: Path,
    band_names: list[str] | None = None,
    compress: str = "lzw",
    nodata: float | None = np.nan,
) -> None:
    """Write (C, H, W) float32 array to a COG GeoTIFF with overview pyramids.

    Args:
        arr: (C, H, W) numpy array.
        transform: Affine geotransform.
        crs: Coordinate reference system (EPSG string or rasterio CRS).
        path: Output file path.
        band_names: Optional list of band name strings for metadata tags.
        compress: Compression codec (lzw, deflate, zstd).
        nodata: Nodata value; use np.nan for float data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(crs, str):
        crs = CRS.from_string(crs)

    n_bands, height, width = arr.shape

    # Write to a temporary in-memory file first, then copy as COG.
    # The copy step reorganises internal tiling and adds overviews.
    tmp_path = path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(
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
            blockysize=512,
        ) as dst:
            dst.write(arr.astype("float32"))
            if band_names:
                for i, name in enumerate(band_names, 1):
                    dst.update_tags(i, name=name)

        # Build overview levels and copy as COG
        _add_overviews_and_copy_as_cog(tmp_path, path, compress)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
    )
