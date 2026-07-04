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
import math
import os
import traceback
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
import pystac_client
import odc.stac
from dask.diagnostics import ProgressBar

from utils.cloud_mask import mask_s2_l2a
from utils.tile_merge import merge_tiles

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

# Approximate bounding boxes for the 48 contiguous US states and DC
# (lon_min, lat_min, lon_max, lat_max) in WGS84.
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AL": (-88.473,  30.144,  -84.889,  35.008),
    "AR": (-94.617,  33.004,  -89.644,  36.500),
    "AZ": (-114.818, 31.332, -109.045,  37.004),
    "CA": (-124.409, 32.534, -114.131,  42.009),
    "CO": (-109.060, 36.992, -102.042,  41.003),
    "CT": (-73.728,  40.980,  -71.787,  42.050),
    "DC": (-77.119,  38.791,  -76.909,  38.995),
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


def sample_scenes_per_month(items: list, max_per_month: int) -> list:
    """Return at most max_per_month scenes per calendar month.

    Within each month, scenes are ranked by ascending cloud cover so the
    clearest observations are always preferred. This keeps the dask task
    graph small while preserving full temporal coverage across the year.
    """
    from collections import defaultdict

    by_month: dict = defaultdict(list)
    for item in items:
        key = (item.datetime.year, item.datetime.month)
        by_month[key].append(item)

    sampled = []
    for key in sorted(by_month):
        month_items = sorted(
            by_month[key],
            key=lambda i: i.properties.get("eo:cloud_cover", 100),
        )
        sampled.extend(month_items[:max_per_month])

    return sampled


def load_and_composite(
    client: pystac_client.Client,
    bbox: tuple[float, float, float, float],
    datetime_str: str,
    bands: list[str],
    crs: str,
    resolution: int,
    max_cloud_cover: int = 30,
    max_scenes_per_month: int | None = 2,
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

    if max_scenes_per_month is not None:
        items = sample_scenes_per_month(items, max_scenes_per_month)
        print(f"    → {len(items)} scenes after sampling "
              f"({max_scenes_per_month}/month max)")

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
    print("    Computing temporal median…")
    try:
        with ProgressBar():
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
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(arr)
        dst.update_tags(band_names=",".join(bands))
    tmp.rename(path)
    print(f"    Saved → {path}  shape={arr.shape}")


def split_bbox_into_tiles(
    bbox: tuple[float, float, float, float],
    crs: str,
    max_tile_km: float = 200.0,
) -> list[tuple[float, float, float, float]]:
    """Split a WGS84 bbox into a grid of tiles at most max_tile_km on each side.

    Tile edges are computed in the projected CRS so each tile has a uniform
    metric size. Returns a flat list of WGS84 (lon_min, lat_min, lon_max, lat_max)
    sub-bboxes. Returns [bbox] unchanged when no tiling is needed.
    """
    from pyproj import Transformer

    t_fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    t_inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    x_min, y_min = t_fwd.transform(lon_min, lat_min)
    x_max, y_max = t_fwd.transform(lon_max, lat_max)

    max_tile_m = max_tile_km * 1_000
    n_x = max(1, math.ceil((x_max - x_min) / max_tile_m))
    n_y = max(1, math.ceil((y_max - y_min) / max_tile_m))

    if n_x == 1 and n_y == 1:
        return [bbox]

    xs = np.linspace(x_min, x_max, n_x + 1)
    ys = np.linspace(y_min, y_max, n_y + 1)

    tiles = []
    for i in range(n_x):
        for j in range(n_y):
            lon1, lat1 = t_inv.transform(xs[i],     ys[j])
            lon2, lat2 = t_inv.transform(xs[i + 1], ys[j + 1])
            tiles.append((
                min(lon1, lon2), min(lat1, lat2),
                max(lon1, lon2), max(lat1, lat2),
            ))

    return tiles


def _run_composite(
    client: pystac_client.Client,
    tiles: list[tuple[float, float, float, float]],
    bands: list[str],
    crs: str,
    datetime_str: str,
    out_path: Path,
    resolution: int,
    max_cloud_cover: int,
    max_scenes_per_month: int | None = 2,
    label: str = "",
    force: bool = False,
    tile_index: int | None = None,
    merge_only: bool = False,
) -> None:
    """Composite one time window over a list of tiles, merging if necessary.

    Tile TIFs are named <out_path.stem>_tile###.tif and deleted after merging.
    Individual tiles that already exist are skipped, enabling resume after
    interruption mid-merge. Pass force=True to delete existing outputs and
    reprocess from scratch.

    Per-tile parallel mode:
      tile_index=N  — process only tile N; write tile TIF, do not merge.
      merge_only=True — merge all existing tile TIFs into final output; exit
                        with a warning if any tiles are still missing.
    """
    prefix = f"  [{label}]" if label else " "
    n = len(tiles)

    # ------------------------------------------------------------------
    # merge-only: assemble the final mosaic from already-written tile TIFs
    # ------------------------------------------------------------------
    if merge_only:
        if out_path.exists() and not force:
            print(f"{prefix} Already exists, skipping merge: {out_path.name}")
            return
        tile_paths = sorted(out_path.parent.glob(f"{out_path.stem}_tile*.tif"))
        max_missing = max(0, n // 20)  # tolerate up to 5% missing (ocean / no-coverage tiles)
        if len(tile_paths) < n - max_missing:
            print(f"{prefix} SKIP merge: expected {n} tiles, found {len(tile_paths)} "
                  f"({n - len(tile_paths)} still missing, max allowed {max_missing})")
            return
        if len(tile_paths) < n:
            print(f"{prefix} WARNING: {n - len(tile_paths)}/{n} tiles missing "
                  f"(likely ocean/no-coverage) — those areas will be nodata")
        print(f"{prefix} Merging {len(tile_paths)}/{n} tiles → {out_path.name}…")
        merge_tiles(tile_paths, bands, out_path)
        return

    # ------------------------------------------------------------------
    # per-tile mode: process exactly one tile, write its TIF, do not merge
    # ------------------------------------------------------------------
    if tile_index is not None:
        if tile_index < 0 or tile_index >= n:
            raise SystemExit(f"--tile-index {tile_index} is out of range [0, {n})")
        if n == 1:
            # Single-tile state: tile_index=0 writes the final output directly
            if out_path.exists() and not force:
                print(f"{prefix} Already exists, skipping: {out_path.name}")
                return
            print(f"{prefix} Compositing ({len(bands)} bands)…")
            arr, transform, out_crs = load_and_composite(
                client, tiles[0], datetime_str, bands, crs, resolution,
                max_cloud_cover, max_scenes_per_month,
            )
            if arr is not None:
                save_tif(arr, transform, out_crs, bands, out_path)
            return
        tile_path = out_path.with_name(f"{out_path.stem}_tile{tile_index:03d}.tif")
        if tile_path.exists() and not force:
            print(f"{prefix} Tile {tile_index + 1}/{n} already exists, skipping")
            return
        print(f"{prefix} Tile {tile_index + 1}/{n} ({len(bands)} bands)…")
        arr, transform, out_crs = load_and_composite(
            client, tiles[tile_index], datetime_str, bands, crs, resolution,
            max_cloud_cover, max_scenes_per_month,
        )
        if arr is None:
            print(f"{prefix} WARNING: tile {tile_index + 1}/{n} returned no data")
            return
        save_tif(arr, transform, out_crs, bands, tile_path)
        return

    # ------------------------------------------------------------------
    # original sequential mode (all tiles in one job, then merge)
    # ------------------------------------------------------------------
    if out_path.exists():
        if not force:
            print(f"{prefix} Already exists, skipping: {out_path.name}")
            return
        print(f"{prefix} --force: removing existing {out_path.name}")
        out_path.unlink()
        # Also remove any leftover tile files from a previous run
        for stale in out_path.parent.glob(f"{out_path.stem}_tile*.tif"):
            stale.unlink()

    def _load(tile_bbox: tuple) -> tuple:
        return load_and_composite(
            client, tile_bbox, datetime_str, bands, crs, resolution,
            max_cloud_cover, max_scenes_per_month,
        )

    out_bands = bands

    if n == 1:
        print(f"{prefix} Compositing ({len(out_bands)} bands)…")
        arr, transform, out_crs = _load(tiles[0])
        if arr is not None:
            save_tif(arr, transform, out_crs, out_bands, out_path)
        return

    # --- tiled path ---
    tile_paths = []
    for idx, tile_bbox in enumerate(tiles):
        tile_path = out_path.with_name(f"{out_path.stem}_tile{idx:03d}.tif")
        if tile_path.exists():
            print(f"{prefix} Tile {idx + 1}/{n} already exists, skipping")
        else:
            print(f"{prefix} Tile {idx + 1}/{n} ({len(out_bands)} bands)…")
            arr, transform, out_crs = _load(tile_bbox)
            if arr is None:
                print(f"{prefix} WARNING: tile {idx + 1}/{n} returned no data, skipping")
                continue
            save_tif(arr, transform, out_crs, out_bands, tile_path)
        tile_paths.append(tile_path)

    if not tile_paths:
        print(f"{prefix} WARNING: no tiles produced data for {out_path.name}")
        return

    print(f"{prefix} Merging {len(tile_paths)}/{n} tiles → {out_path.name}…")
    merge_tiles(tile_paths, out_bands, out_path)


def process_state(
    client: pystac_client.Client,
    state: str,
    bbox: tuple[float, float, float, float],
    model: str,
    year: int,
    resolution: int,
    output_dir: Path,
    max_cloud_cover: int = 30,
    max_tile_km: float = 200.0,
    max_scenes_per_month: int | None = 2,
    force: bool = False,
    tile_index: int | None = None,
    merge_only: bool = False,
) -> None:
    """Run composite generation for a single state, tiling automatically if needed."""
    crs = bbox_to_utm_epsg(bbox)
    tiles = split_bbox_into_tiles(bbox, crs, max_tile_km)
    n_tiles = len(tiles)

    print(f"\n{'='*60}")
    tile_info = f"  {n_tiles} tiles" if n_tiles > 1 else ""
    mode_info = (f"  tile-index={tile_index}" if tile_index is not None else
                 "  merge-only" if merge_only else "")
    print(f"State: {state}  Year: {year}  Model: {model}  "
          f"Source: Element84  CRS: {crs}{tile_info}{mode_info}")

    if model == "olmoearth":
        _run_composite(
            client, tiles, OLMOEARTH_BANDS, crs,
            f"{year}-01-01/{year}-12-31",
            output_dir / f"s2_annual_{state}_{year}_olmoearth.tif",
            resolution, max_cloud_cover, max_scenes_per_month,
            force=force, tile_index=tile_index, merge_only=merge_only,
        )

    elif model == "prithvi":
        for season, (start, end) in SEASONS.items():
            _run_composite(
                client, tiles, PRITHVI_BANDS, crs,
                f"{year}-{start}/{year}-{end}",
                output_dir / f"s2_{season}_{state}_{year}_prithvi.tif",
                resolution, max_cloud_cover, max_scenes_per_month,
                label=season, force=force,
                tile_index=tile_index, merge_only=merge_only,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=None,
                        help="Two-letter state abbreviation (must be in STATE_BBOXES). "
                             "Ignored when --all-states is set.")
    parser.add_argument("--all-states", action="store_true",
                        help="Process all 48 contiguous US states and DC sequentially, "
                             "skipping any whose output files already exist.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                        help="Override state bbox: W S E N in WGS84. Single state only.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], default="olmoearth")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Output pixel resolution in metres. Defaults to 10 m for "
                             "OlmoEarth and 30 m for Prithvi. Override only if you have "
                             "a specific reason to.")
    parser.add_argument("--max-cloud-cover", type=int, default=30,
                        help="Maximum scene-level cloud cover %% to include (default 30). "
                             "Lower values reduce memory usage by filtering more scenes.")
    parser.add_argument("--max-tile-km", type=float, default=200.0,
                        help="Maximum tile side length in km (default 200). States wider "
                             "than this are split into a grid of tiles and merged after "
                             "compositing. Reduce if you still hit memory limits.")
    parser.add_argument("--max-scenes-per-month", type=int, default=2,
                        help="Keep at most N scenes per calendar month, choosing the "
                             "clearest (lowest cloud cover) first (default 3). "
                             "Use 0 to disable sampling and use all available scenes.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files instead of skipping them. "
                             "Also removes any leftover tile files from previous runs.")
    parser.add_argument("--tile-index", type=int, default=None,
                        help="Process only this tile (0-based index). Writes the tile TIF "
                             "and exits without merging. Use with --merge-only in a "
                             "separate job after all tiles complete.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip downloading; merge existing tile TIFs into the final "
                             "output. Exits with a warning if any tiles are still missing.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/composites"))
    args = parser.parse_args()

    # --merge-only reads only from disk; skip the network round-trip to Element84.
    client = None if args.merge_only else pystac_client.Client.open(STAC_ENDPOINT)

    # Per-model resolution defaults: OlmoEarth=10m (native S2), Prithvi=30m (S2 resampled)
    resolution = args.resolution or (10 if args.model == "olmoearth" else 30)
    max_scenes = None if args.max_scenes_per_month == 0 else args.max_scenes_per_month

    if args.all_states:
        states = list(STATE_BBOXES.items())
        print(f"Running all {len(states)} CONUS states  year={args.year}  "
              f"model={args.model}  source=Element84  resolution={resolution}m  "
              f"max_cloud={args.max_cloud_cover}%  "
              f"max_tile={args.max_tile_km:.0f}km  "
              f"max_scenes/month={max_scenes or 'unlimited'}")
        for state, bbox in states:
            process_state(client, state, bbox, args.model, args.year,
                          resolution, args.output_dir,
                          args.max_cloud_cover, args.max_tile_km, max_scenes, args.force)
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
                      resolution, args.output_dir,
                      args.max_cloud_cover, args.max_tile_km, max_scenes, args.force,
                      tile_index=args.tile_index, merge_only=args.merge_only)


if __name__ == "__main__":
    main()
