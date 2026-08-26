"""Land-cover labels from ESA WorldCover.

Why a raster rather than OpenStreetMap
--------------------------------------
OSM land cover is volunteer-drawn and its density tracks population. A chip of
Maryland farmland came back with 25% of patches carrying any label at all while
urban chips reached 92% -- not because the ground is unlabelled in reality, but
because nobody hand-drew those fields. Correlating against that produces a
confident-looking negative result for exactly the rural classes, which is the
worst kind of wrong.

WorldCover is wall-to-wall: every pixel on Earth carries a class, so a rural chip
is labelled as completely as a city one. It is also **10 m**, the same grid as
Sentinel-2, so it aligns with the chips without resampling artefacts.

The circularity to disclose
---------------------------
``worldcover`` was one of OlmoEarth's pretraining modalities. Correlating an
embedding against it therefore partly asks "did the model learn its own training
target", and a strong result is less surprising than it appears. That is fine for
interpreting what a dimension means -- the goal here -- but it is weaker evidence
about generalisation than an independent label would be, and it belongs in any
writeup. The same caveat applies to the OSM ``building``/``highway``/``parking``
rasters, so it is not avoidable by switching source.

Access
------
Public S3, no credentials. Tiles are 3x3 degrees, named by their south-west
corner on a 3-degree grid. Read through GDAL's virtual filesystem so only the
chip's window crosses the network -- a full tile is hundreds of megabytes and
none of it is needed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
DEFAULT_VERSION = "v200"
DEFAULT_YEAR = "2021"

# WorldCover class codes. Water folds in herbaceous wetland: at 40 m a wetland
# patch reads as water to any reflectance-based model, and separating them would
# create two sparse labels instead of one usable one.
CLASSES: dict[str, tuple[int, ...]] = {
    "canopy": (10,),
    "shrub": (20,),
    "grass": (30,),
    "farmland": (40,),
    "built": (50,),
    "bare": (60,),
    "water": (80, 90),
    # Derived, and deliberately overlapping `grass` and `farmland`. WorldCover
    # puts row crops in class 40 but pasture, hay and meadow in class 30, so a
    # dairy county reads as 8% "farmland" and 42% "grassland" while looking
    # entirely agricultural from the air. Anyone asking "how much of this chip is
    # farmed" wants both classes, and gets a misleading answer from either alone.
    #
    # For crop-type detail -- hay versus corn versus soy -- WorldCover cannot
    # help at all; that needs USDA CDL.
    "agriculture": (30, 40),
}

CLASS_NAMES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}


def tile_name(lat: float, lon: float) -> str:
    """Return the WorldCover tile name containing a coordinate.

    Tiles are named by their south-west corner snapped down to a 3-degree grid,
    latitude two digits and longitude three: ``N36W078``.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        The tile name.
    """
    tile_lat = int(math.floor(lat / 3.0) * 3)
    tile_lon = int(math.floor(lon / 3.0) * 3)
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"


def tile_url(
    name: str, version: str = DEFAULT_VERSION, year: str = DEFAULT_YEAR
) -> str:
    """Return the public URL for a tile.

    Args:
        name: A tile name from :func:`tile_name`.
        version: Product version.
        year: Product year.

    Returns:
        An https URL to a Cloud-Optimized GeoTIFF.
    """
    return (
        f"{BUCKET}/{version}/{year}/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{name}_Map.tif"
    )


def tiles_for_bbox(bounds: tuple[float, float, float, float]) -> list[str]:
    """Return every tile name a lat/lon box touches.

    A 2.5 km chip almost always sits inside one 3-degree tile, but a chip near a
    tile edge straddles two and reading only one would leave part of it unlabelled
    -- indistinguishable from genuinely unclassified ground.

    Args:
        bounds: ``(latmin, lonmin, latmax, lonmax)``.

    Returns:
        Tile names, in a stable order.
    """
    latmin, lonmin, latmax, lonmax = bounds
    names: list[str] = []
    lat = math.floor(latmin / 3.0) * 3
    while lat <= latmax:
        lon = math.floor(lonmin / 3.0) * 3
        while lon <= lonmax:
            name = tile_name(lat + 0.5, lon + 0.5)
            if name not in names:
                names.append(name)
            lon += 3
        lat += 3
    return names


def read_class_grid(
    source: str,
    crs: Any,
    affine: Any,
    row0: int,
    col0: int,
    chip_px: int,
    scene_height: int,
    scene_width: int,
) -> np.ndarray:
    """Read WorldCover classes onto a chip's own pixel grid.

    Reprojects from WorldCover's geographic grid onto the scene's UTM grid so a
    class value lands on the same pixel as the reflectance it describes.
    Resampling is **nearest** -- these are category codes, and interpolating
    between "cropland" (40) and "built-up" (50) would invent "bare" (45).

    Args:
        source: Tile URL or local path.
        crs: The scene's CRS.
        affine: The scene's affine transform.
        row0: Chip top pixel row in the scene.
        col0: Chip left pixel column in the scene.
        chip_px: Chip side in pixels.
        scene_height: Scene height in pixels.
        scene_width: Scene width in pixels.

    Returns:
        ``(chip_px, chip_px)`` uint8 class codes. 0 means no data.

    Raises:
        RuntimeError: If the tile cannot be opened.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    try:
        opened = rasterio.open(source)
    except Exception as exc:  # noqa: BLE001 - rasterio raises several types here
        raise RuntimeError(
            f"could not open {source}: {exc}\nIf this is a network error, GDAL "
            f"may lack curl support in this environment -- download the tile "
            f"and pass a local path instead."
        ) from exc

    with opened as src:
        with WarpedVRT(
            src,
            crs=crs,
            transform=affine,
            width=scene_width,
            height=scene_height,
            resampling=Resampling.nearest,
        ) as vrt:
            # Only this window is materialised, so the rest of the tile is never
            # fetched.
            return vrt.read(
                1, window=Window(col0, row0, chip_px, chip_px)
            ).astype(np.uint8)


def coverage_grids(
    classes: np.ndarray, wanted: list[str], patch_px: int
) -> dict[str, np.ndarray]:
    """Turn a class raster into per-patch coverage fractions.

    Args:
        classes: ``(chip_px, chip_px)`` class codes.
        wanted: Label names, keys of :data:`CLASSES`.
        patch_px: Pixels per patch.

    Returns:
        Label name to ``(side, side)`` float32 fractions in [0, 1].

    Raises:
        KeyError: If a label is not a WorldCover class group.
    """
    side = classes.shape[0] // patch_px
    out: dict[str, np.ndarray] = {}
    for label in wanted:
        codes = CLASSES[label]
        mask = np.isin(classes, codes).astype(np.float32)
        out[label] = mask.reshape(side, patch_px, side, patch_px).mean(axis=(1, 3))
    return out


def describe(classes: np.ndarray) -> list[tuple[str, float]]:
    """Summarise which classes a chip contains, most common first.

    Useful as a sanity check independent of the label set: a chip that reads as
    90% cropland is farmland whatever OSM does or does not say about it.

    Args:
        classes: Class codes.

    Returns:
        ``(class name, fraction)`` pairs, descending, omitting anything under 1%.
    """
    total = classes.size
    found: list[tuple[str, float]] = []
    for code in np.unique(classes):
        fraction = float((classes == code).sum()) / total
        if fraction < 0.01:
            continue
        name = CLASS_NAMES.get(int(code), f"class {int(code)}")
        found.append((name, fraction))
    return sorted(found, key=lambda pair: -pair[1])


def local_path(cache_dir: Path, name: str, version: str, year: str) -> Path:
    """Return the local cache path for a tile."""
    return cache_dir / f"ESA_WorldCover_10m_{year}_{version}_{name}_Map.tif"
