"""Write multi-band numpy arrays as Cloud-Optimized GeoTIFFs."""
import numpy as np
import rasterio
import rasterio.windows
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
    compress: str = "zstd",
    nodata: float | None = np.nan,
    overviews: bool = True,
) -> None:
    """Write (C, H, W) float32 array to a tiled GeoTIFF, optionally with COG overviews.

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
            BIGTIFF="IF_SAFER",
            **_compression_options(compress),
        ) as dst:
            for row_start in range(0, height, _STRIP_ROWS):
                row_end = min(row_start + _STRIP_ROWS, height)
                strip = np.array(arr[:, row_start:row_end, :], dtype="float32")
                win = rasterio.windows.Window(0, row_start, width, row_end - row_start)
                dst.write(strip, window=win)
            if band_names:
                for i, name in enumerate(band_names, 1):
                    dst.update_tags(i, name=name)

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
