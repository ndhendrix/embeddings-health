"""
Produce an annual (or seasonal) Sentinel-2 L2A cloud-free composite.

Usage:
  python composite.py --state RI --year 2022 --model olmoearth
  python composite.py --state RI --year 2022 --model prithvi

For OlmoEarth: writes one annual-median GeoTIFF (12 bands).
For Prithvi:   writes three seasonal-median GeoTIFFs (spring/summer/fall, 6 bands each).
Output files land in --output-dir.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import pystac_client
import odc.stac

from utils.cloud_mask import mask_s2_l2a


STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# AWS Element84 STAC uses common names as asset keys (not B01/B02/... codes).
# Band ordering follows olmoearth_pretrain.data.constants.Modality.SENTINEL2_L2A band sets:
#   BandSet 0 (10m):  B02 B03 B04 B08
#   BandSet 1 (20m):  B05 B06 B07 B8A B11 B12
#   BandSet 2 (60m):  B01 B09
OLMOEARTH_BANDS = [
    "blue",      # B02 — BandSet 0
    "green",     # B03
    "red",       # B04
    "nir",       # B08
    "rededge1",  # B05 — BandSet 1
    "rededge2",  # B06
    "rededge3",  # B07
    "nir08",     # B8A
    "swir16",    # B11
    "swir22",    # B12
    "coastal",   # B01 — BandSet 2
    "nir09",     # B09
]

# 6 HLS-compatible bands: Blue, Green, Red, Narrow NIR, SWIR-1, SWIR-2
# TODO: verify order against ibm-nasa-geospatial/Prithvi-EO-2.0-300M config.json
PRITHVI_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Seasonal windows for Prithvi's 3-timestep input
SEASONS = {
    "spring": (f"03-01", "05-31"),
    "summer": (f"06-01", "08-31"),
    "fall":   (f"09-01", "11-30"),
}

# Approximate bounding boxes for US states (lon_min, lat_min, lon_max, lat_max).
# Add more states as needed; used only when --state is given without --bbox.
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "RI": (-71.908, 41.146, -71.075, 42.018),
    "CT": (-73.728, 40.980, -71.787, 42.050),
    "DE": (-75.789, 38.451, -74.984, 39.839),
    "MA": (-73.508, 41.237, -69.928, 42.887),
}


def bbox_to_utm_epsg(bbox: tuple[float, float, float, float]) -> str:
    """Return the UTM EPSG code for the bbox midpoint (northern hemisphere)."""
    lon_mid = (bbox[0] + bbox[2]) / 2
    zone = int((lon_mid + 180) / 6) + 1
    return f"EPSG:{32600 + zone}"


def load_and_composite(
    client: pystac_client.Client,
    bbox: tuple[float, float, float, float],
    datetime_str: str,
    bands: list[str],
    crs: str,
    resolution: int,
    max_cloud_cover: int = 80,
) -> np.ndarray | None:
    """Query STAC, cloud-mask, and return pixel-wise median as (C, H, W) float32.

    Returns None if no usable scenes are found.
    """
    search = client.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=datetime_str,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.item_collection())
    print(f"    {len(items)} scenes found for {datetime_str}")
    if not items:
        return None, None, None

    all_bands = bands + ["scl"]
    ds = odc.stac.load(
        items,
        bands=all_bands,
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        groupby="solar_day",
        chunks={"time": 1, "x": 2048, "y": 2048},
    )

    # Mask clouds using SCL
    masked = mask_s2_l2a(ds, scl_band="scl")

    # Pixel-wise temporal median; drop time dim
    print("    Computing temporal median (this may take a moment)...")
    median = masked.median(dim="time").compute()

    # Stack into (C, H, W)
    arr = np.stack([median[b].values for b in bands], axis=0).astype("float32")

    # Extract geotransform from odc-stac Dataset
    geobox = ds.odc.geobox
    transform = geobox.transform
    out_crs = geobox.crs.to_wkt()

    return arr, transform, out_crs


def save_tif(arr: np.ndarray, transform, crs: str, bands: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a .tmp file first, then rename — ensures the output is either
    # complete or absent (never a partial TIF that looks done).
    tmp = path.with_suffix(".tmp.tif")
    n, h, w = arr.shape
    with rasterio.open(
        tmp, "w",
        driver="GTiff",
        height=h, width=w, count=n,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr)
        dst.update_tags(band_names=",".join(bands))
    tmp.rename(path)
    print(f"    Saved → {path}  shape={arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="RI",
                        help="Two-letter state abbreviation (must be in STATE_BBOXES)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                        help="Override state bbox: W S E N in WGS84")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], default="olmoearth")
    parser.add_argument("--resolution", type=int, default=10,
                        help="Output pixel resolution in metres (default 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/composites"))
    args = parser.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.state in STATE_BBOXES:
        bbox = STATE_BBOXES[args.state]
    else:
        raise SystemExit(f"State '{args.state}' not in STATE_BBOXES. Use --bbox W S E N.")

    crs = bbox_to_utm_epsg(bbox)
    print(f"State: {args.state}  Year: {args.year}  Model: {args.model}  CRS: {crs}")

    client = pystac_client.Client.open(STAC_ENDPOINT)

    if args.model == "olmoearth":
        bands = OLMOEARTH_BANDS
        datetime_str = f"{args.year}-01-01/{args.year}-12-31"
        out_path = args.output_dir / f"s2_annual_{args.state}_{args.year}_olmoearth.tif"
        if out_path.exists():
            print(f"Annual composite already exists, skipping: {out_path}")
        else:
            print(f"Annual composite ({len(bands)} bands)…")
            arr, transform, out_crs = load_and_composite(
                client, bbox, datetime_str, bands, crs, args.resolution
            )
            if arr is not None:
                save_tif(arr, transform, out_crs, bands, out_path)

    elif args.model == "prithvi":
        bands = PRITHVI_BANDS
        for season, (start, end) in SEASONS.items():
            out_path = args.output_dir / f"s2_{season}_{args.state}_{args.year}_prithvi.tif"
            if out_path.exists():
                print(f"  Season {season} already exists, skipping: {out_path}")
                continue
            datetime_str = f"{args.year}-{start}/{args.year}-{end}"
            print(f"Season: {season}  ({datetime_str}, {len(bands)} bands)…")
            arr, transform, out_crs = load_and_composite(
                client, bbox, datetime_str, bands, crs, args.resolution
            )
            if arr is not None:
                save_tif(arr, transform, out_crs, bands, out_path)


if __name__ == "__main__":
    main()
