"""Memory-bounded GeoTIFF copy and mosaic helpers for OlmoEarth downloads."""
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge
from rasterio.vrt import WarpedVRT


COG_PROFILE = {
    "driver": "GTiff",
    "compress": "deflate",
    "predictor": 2,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "bigtiff": "IF_SAFER",
}


def write_cog_from_tifs(tif_paths: list[Path], out_path: Path, mem_limit_mb: int = 256) -> tuple:
    """Write one or more source TIFs to out_path without materializing the mosaic.

    Returns the output shape as (bands, height, width).
    """
    if not tif_paths:
        raise ValueError("No TIF paths supplied")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")
    tmp_path.unlink(missing_ok=True)

    try:
        if len(tif_paths) == 1:
            shape = _copy_one_tif(tif_paths[0], tmp_path)
        else:
            shape = _merge_tifs_streaming(tif_paths, tmp_path, mem_limit_mb)
        tmp_path.rename(out_path)
        return shape
    finally:
        tmp_path.unlink(missing_ok=True)


def is_readable_tif(path: Path) -> bool:
    """Return True if Rasterio can open the TIF and read its basic metadata."""
    try:
        with rasterio.open(path) as src:
            return src.count > 0 and src.width > 0 and src.height > 0
    except Exception:
        return False


def _copy_one_tif(src_path: Path, out_path: Path) -> tuple:
    """Copy a single TIF block-by-block into the standard compressed profile."""
    with rasterio.open(src_path) as src:
        profile = {**src.profile, **COG_PROFILE}
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(src.read(window=window), window=window)
        return (profile["count"], profile["height"], profile["width"])


def _merge_tifs_streaming(tif_paths: list[Path], out_path: Path, mem_limit_mb: int) -> tuple:
    """Merge multiple TIFs using Rasterio's windowed dst_path writer."""
    datasets = [rasterio.open(p) for p in tif_paths]
    vrts = []
    try:
        target_crs = datasets[0].crs
        crs_set = {str(ds.crs) for ds in datasets}
        if len(crs_set) > 1:
            print(
                f"    Reprojecting {len(crs_set)} source CRSs to {target_crs} "
                "for mosaic"
            )
            sources = [
                WarpedVRT(ds, crs=target_crs, resampling=Resampling.nearest)
                for ds in datasets
            ]
            vrts.extend(sources)
        else:
            sources = datasets

        profile = {**datasets[0].profile, "crs": target_crs, **COG_PROFILE}
        rasterio_merge(
            sources,
            dst_path=out_path,
            dst_kwds=profile,
            mem_limit=mem_limit_mb,
        )
        with rasterio.open(out_path) as dst:
            return (dst.count, dst.height, dst.width)
    finally:
        for vrt in vrts:
            vrt.close()
        for ds in datasets:
            ds.close()
