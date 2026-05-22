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
import traceback
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import pystac_client
import odc.stac

from utils.cloud_mask import mask_s2_l2a

# Treat NotGeoreferencedWarning as a printed warning only — never as an error.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


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

# Approximate bounding boxes for all 48 contiguous US states
# (lon_min, lat_min, lon_max, lat_max) in WGS84.
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AL": (-88.473,  30.144,  -84.889,  35.008),
    "AR": (-94.617,  33.004,  -89.644,  36.500),
    "AZ": (-114.818, 31.332, -109.045,  37.004),
    "CA": (-124.409, 32.534, -114.131,  42.009),
    "CO": (-109.060, 36.992, -102.042,  41.003),
    "CT": (-73.728,  40.980,  -71.787,  42.050),
    "DE": (-75.789,  38.451,  -74.984,  39.839),
    "FL": (-87.634,  24.396,  -79.974,  31.001),
    "GA": (-85.605,  30.356,  -80.840,  35.001),
    "IA": (-96.639,  40.376,  -90.140,  43.501),
    "ID": (-117.243, 41.988, -111.043,  49.001),
    "IL": (-91.513,  36.970,  -87.020,  42.508),
    "IN": (-88.097,  37.772,  -84.784,  41.761),
    "KS": (-102.051, 36.993,  -94.588,  40.003),
    "KY": (-89.571,  36.497,  -81.965,  39.148),
    "LA": (-94.043,  28.928,  -88.817,  33.020),
    "MA": (-73.508,  41.237,  -69.928,  42.887),
    "MD": (-79.487,  37.886,  -75.048,  39.723),
    "ME": (-71.083,  43.058,  -66.949,  47.460),
    "MI": (-90.418,  41.697,  -82.122,  48.306),
    "MN": (-97.239,  43.500,  -89.491,  49.384),
    "MO": (-95.774,  35.996,  -89.099,  40.614),
    "MS": (-91.655,  30.174,  -88.098,  35.008),
    "MT": (-116.049, 44.358, -104.040,  49.001),
    "NC": (-84.322,  33.842,  -75.460,  36.588),
    "ND": (-104.049, 45.935,  -96.554,  49.001),
    "NE": (-104.053, 39.999,  -95.308,  43.001),
    "NH": (-72.557,  42.697,  -70.610,  45.305),
    "NJ": (-75.563,  38.928,  -73.893,  41.358),
    "NM": (-109.050, 31.332, -103.002,  37.000),
    "NV": (-120.005, 35.002, -114.039,  42.002),
    "NY": (-79.762,  40.496,  -71.856,  45.015),
    "OH": (-84.820,  38.403,  -80.518,  41.978),
    "OK": (-103.002, 33.616,  -94.431,  37.002),
    "OR": (-124.566, 41.992, -116.463,  46.236),
    "PA": (-80.519,  39.720,  -74.690,  42.269),
    "RI": (-71.908,  41.146,  -71.075,  42.018),
    "SC": (-83.354,  32.034,  -78.541,  35.215),
    "SD": (-104.058, 42.480,  -96.436,  45.945),
    "TN": (-90.310,  34.983,  -81.647,  36.678),
    "TX": (-106.646, 25.837,  -93.508,  36.501),
    "UT": (-114.053, 36.998, -109.041,  42.001),
    "VA": (-83.675,  36.541,  -75.242,  39.466),
    "VT": (-73.437,  42.727,  -71.465,  45.017),
    "WA": (-124.848, 45.544, -116.916,  49.002),
    "WI": (-92.889,  42.492,  -86.249,  47.309),
    "WV": (-82.644,  37.202,  -77.719,  40.638),
    "WY": (-111.056, 40.995, -104.052,  45.006),
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
        chunks={"time": 1, "x": 512, "y": 512},
    )

    # Mask clouds using SCL
    masked = mask_s2_l2a(ds, scl_band="scl")

    # Pixel-wise temporal median; drop time dim
    print("    Computing temporal median (this may take a moment)...")
    try:
        median = masked.median(dim="time").compute()
    except Exception:
        print("    ERROR during dask compute:")
        traceback.print_exc()
        return None, None, None

    print(f"    Median dataset dims: {dict(median.dims)}")

    # Stack into (C, H, W)
    try:
        arr = np.stack([median[b].values for b in bands], axis=0).astype("float32")
    except Exception:
        print("    ERROR stacking bands:")
        traceback.print_exc()
        return None, None, None

    print(f"    Array shape: {arr.shape}  dtype={arr.dtype}  "
          f"nan_frac={np.isnan(arr).mean():.1%}")

    if arr.shape[1] == 0 or arr.shape[2] == 0:
        print("    WARNING: empty spatial extent — skipping.")
        return None, None, None

    # Extract geotransform from odc-stac Dataset
    try:
        geobox = ds.odc.geobox
        transform = geobox.transform
        out_crs = geobox.crs.to_wkt()
    except Exception:
        print("    ERROR extracting geobox:")
        traceback.print_exc()
        return None, None, None

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


def process_state(
    client: pystac_client.Client,
    state: str,
    bbox: tuple[float, float, float, float],
    model: str,
    year: int,
    resolution: int,
    output_dir: Path,
) -> None:
    """Run composite generation for a single state."""
    crs = bbox_to_utm_epsg(bbox)
    print(f"\n{'='*60}")
    print(f"State: {state}  Year: {year}  Model: {model}  CRS: {crs}")

    if model == "olmoearth":
        bands = OLMOEARTH_BANDS
        datetime_str = f"{year}-01-01/{year}-12-31"
        out_path = output_dir / f"s2_annual_{state}_{year}_olmoearth.tif"
        if out_path.exists():
            print(f"  Annual composite already exists, skipping: {out_path}")
            return
        print(f"  Annual composite ({len(bands)} bands)…")
        arr, transform, out_crs = load_and_composite(
            client, bbox, datetime_str, bands, crs, resolution
        )
        if arr is not None:
            save_tif(arr, transform, out_crs, bands, out_path)

    elif model == "prithvi":
        bands = PRITHVI_BANDS
        for season, (start, end) in SEASONS.items():
            out_path = output_dir / f"s2_{season}_{state}_{year}_prithvi.tif"
            if out_path.exists():
                print(f"  Season {season} already exists, skipping: {out_path}")
                continue
            datetime_str = f"{year}-{start}/{year}-{end}"
            print(f"  Season: {season}  ({datetime_str}, {len(bands)} bands)…")
            arr, transform, out_crs = load_and_composite(
                client, bbox, datetime_str, bands, crs, resolution
            )
            if arr is not None:
                save_tif(arr, transform, out_crs, bands, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=None,
                        help="Two-letter state abbreviation (must be in STATE_BBOXES). "
                             "Ignored when --all-states is set.")
    parser.add_argument("--all-states", action="store_true",
                        help="Process all 48 contiguous US states sequentially, "
                             "skipping any whose output files already exist.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                        help="Override state bbox: W S E N in WGS84. Single state only.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], default="olmoearth")
    parser.add_argument("--resolution", type=int, default=10,
                        help="Output pixel resolution in metres (default 10)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/composites"))
    args = parser.parse_args()

    client = pystac_client.Client.open(STAC_ENDPOINT)

    if args.all_states:
        states = list(STATE_BBOXES.items())
        print(f"Running all {len(states)} CONUS states  year={args.year}  model={args.model}")
        for state, bbox in states:
            process_state(client, state, bbox, args.model, args.year,
                          args.resolution, args.output_dir)
        print("\nAll states complete.")
    else:
        state = args.state or "RI"
        if args.bbox:
            bbox = tuple(args.bbox)
        elif state in STATE_BBOXES:
            bbox = STATE_BBOXES[state]
        else:
            raise SystemExit(f"State '{state}' not in STATE_BBOXES. Use --bbox W S E N.")
        process_state(client, state, bbox, args.model, args.year,
                      args.resolution, args.output_dir)


if __name__ == "__main__":
    main()
