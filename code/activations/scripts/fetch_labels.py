#!/usr/bin/env python
"""Build per-patch ground-truth labels for an encoded chip, from two sources.

Runs *after* encoding and *before* correlation. It never touches the model, the
GPU, or the saved embeddings -- it only answers "what is actually on the ground
in each patch", so that "does any dimension respond to parks" becomes a question
about two aligned arrays.

Two backends, split by what each is good at
-------------------------------------------
**ESA WorldCover** (raster) for land cover: canopy, grass, farmland, water,
built. Wall-to-wall, 10 m, one HTTP range read per chip, no rate limit. This
exists because OSM land cover is volunteer-drawn and its density tracks
population: a Maryland farmland chip came back with 25% of patches carrying any
label while urban chips reached 92%, not because the fields are unlabelled in
reality but because nobody drew them. Correlating against that yields a
confident-looking negative for exactly the rural classes.

**OpenStreetMap** (vector) for discrete built features: buildings, roads,
parking, walkability, park, recreation, food_retail, healthcare. No raster
distinguishes a parking apron from a warehouse roof, so these have no
alternative source. Three geometry types are handled -- polygons filled, lines
widened, and points expanded to a disc -- because the same group routinely
contains all three.

The names are the contract. ``canopy`` means the WorldCover tree class;
``osm_canopy`` means OSM woodland polygons. Both are available, and the source of
every label is recorded in the output so a cached array from one backend is never
silently reused as if it came from the other.

How a polygon becomes a number per patch
----------------------------------------
1. Reconstruct the exact chip window from the scene, so patch (i, j) maps to a
   known block of pixels. This is read from the imagery's own CRS and affine
   transform rather than approximated from the centre coordinate -- the tiles sit
   on a UTM grid that is slightly rotated relative to lat/lon, and at 2.5 km that
   rotation is worth about a patch. A systematic one-patch offset would attenuate
   every correlation computed downstream.
2. Rasterise each label's polygons at full 10 m pixel resolution.
3. Mean-pool each patch_px x patch_px block, giving *coverage fraction* per
   patch rather than a yes/no.

The fraction matters. A 40 m patch rarely lands neatly inside or outside a park
boundary, and the embedding of a half-park patch is genuinely a blend, so a hard
threshold would add noise at exactly the boundaries that carry the most
information.

What to expect from each label
------------------------------
``canopy``, ``water``, ``farmland``, ``grass``, ``built``
    Land cover, directly measurable from reflectance. If these show nothing the
    method is at fault rather than the target, which is why they lead. ``water``
    doubles as a control: it should correlate with *something*, and if it does
    not, stop and debug rather than interpreting the rest.
``roads``
    Lines, not areas, so rasterised with a width. OSM ``highway`` was a
    pretraining band, and street-grid texture is visible by eye in the dimension
    maps.
``buildings``, ``parking``
    Both were OSM pretraining rasters, so there is reason to expect a signal --
    but they are dense and are the groups that make Overpass throttle.
``walkability``
    Pedestrian ways: footways, paths, steps, pedestrian streets, cycleways.
    **This is not a walkability index.** Walkability as the literature means it
    combines intersection density, land-use mix and destination access, and no
    single OSM tag expresses it. What this measures is the presence of mapped
    pedestrian infrastructure, which is one input to walkability and is also
    heavily confounded with how thoroughly an area has been mapped. For a real
    index at tract level -- which is where the health modelling happens anyway --
    use EPA's National Walkability Index rather than this.
``food_retail``, ``healthcare``, ``hospitals``
    Land *use*, and mostly point-mapped: a pharmacy in a strip mall is one OSM
    node with no outline, so these rely on ``point_radius_px`` to have any extent
    at all. Never labelled functionally during pretraining, so reachable at best
    through building morphology. Expect little. ``hospitals`` is separated out
    because a hospital campus usually does have a real outline, unlike a clinic,
    making it the only one of the three with enough footprint to stand a chance.

Coverage governs all of this more than the model does. Empirically, labels above
~0.28 mean coverage transferred to held-out locations and labels at or below 0.03
did not, whatever they were. ``healthcare`` and ``food_retail`` will land far
below that, so a null result for them says "too rare in seven chips to measure",
not "the model is blind to it".

A disclosure worth keeping: ``worldcover``, ``cdl`` and the OSM
``building``/``highway``/``parking`` rasters were all OlmoEarth pretraining
modalities. Correlating an embedding against them partly asks whether the model
learned its own training target, so a strong result is less surprising than it
looks. Adequate for interpreting what a dimension means; weaker evidence about
generalisation.

Examples:
    python scripts/fetch_labels.py --location rock_creek
    python scripts/fetch_labels.py --all --labels canopy,farmland,water,built
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from olmoearth_activations import locations as loc
from olmoearth_activations import osm, osm_local, worldcover
from olmoearth_activations.config import RunConfig

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

# Line labels get a width; area labels are filled. Widths are deliberately
# modest -- a residential street is roughly one 10 m pixel wide, and inflating it
# would smear the label across patches that contain no road.
LABEL_GROUPS: dict[str, dict[str, Any]] = {
    "park": {"specs": ["leisure=park", "leisure=garden", "leisure=nature_reserve"]},
    "recreation": {
        "specs": [
            "leisure=pitch",
            "leisure=sports_centre",
            "leisure=track",
            "leisure=swimming_pool",
            "leisure=playground",
        ]
    },
    # Land-cover equivalents of these now come from WorldCover instead -- see
    # WORLDCOVER_LABELS. Kept available under an osm_ prefix for comparing the
    # two sources against each other, which is the only way to see how much the
    # volunteer-mapping gap actually costs.
    "osm_canopy": {"specs": ["natural=wood", "landuse=forest", "natural=scrub"]},
    "osm_water": {
        "specs": ["natural=water", "waterway=riverbank", "landuse=reservoir"]
    },
    "osm_farmland": {
        "specs": ["landuse=farmland", "landuse=meadow", "landuse=orchard"]
    },
    # ``tiles: n`` splits the chip into an n x n grid of separate queries. A
    # dense urban chip holds thousands of building footprints, and asking for
    # all of them with full geometry in one request makes the public Overpass
    # instance give up with a 504. Smaller boxes each return quickly. Features
    # straddling a sub-box boundary come back in both and are de-duplicated by
    # element id.
    # n means n x n queries, so keep it small: 2 is four sub-queries, 4 would be
    # sixteen. Buildings and roads are the only groups dense enough to need it,
    # and the retry-with-backoff in osm.run covers the occasional 504 that still
    # slips through.
    "roads": {"specs": ["highway"], "line_width_px": 1.5, "tiles": 2},
    "buildings": {"specs": ["building"], "tiles": 2},
    "parking": {"specs": ["amenity=parking", "parking"]},
    # Pedestrian infrastructure. NOT a walkability index -- see the module
    # docstring. Dense in cities, so it is tiled like roads.
    "walkability": {
        "specs": [
            "highway=footway",
            "highway=pedestrian",
            "highway=path",
            "highway=steps",
            "highway=living_street",
            "highway=cycleway",
        ],
        "line_width_px": 1.0,
        "tiles": 2,
    },
    # Zoning polygons rather than physical cover. They blanket large areas and
    # help account for ground the other groups leave unlabelled, but OSM's
    # coverage of them is patchy and inconsistent between cities -- so they are
    # available rather than default.
    "developed": {
        "specs": [
            "landuse=residential",
            "landuse=commercial",
            "landuse=industrial",
            "landuse=retail",
        ]
    },
    "osm_grass": {
        "specs": ["landuse=grass", "natural=grassland", "landuse=village_green"]
    },
    # ---------------------------------------------------------------------
    # Landmark groups, organised by the pathway each plausibly acts through to
    # a PLACES outcome. Cheap to add once labels come from a local extract, so
    # the set is deliberately broad -- but breadth is not evidence. Anything
    # under ~0.03 mean coverage cannot be probed from seven chips whatever the
    # model encodes, and most POI groups will land far below that. Read
    # `label_mean_coverage` before reading any score.
    #
    # ``point_radius_px`` is what makes these work at all. Most pharmacies,
    # clinics and shops exist in OSM only as a single tagged point with no
    # outline, and a point has no area -- without a radius the group returns a
    # confident zero, which reads as "the model cannot see it" when the truth is
    # "the label was never built". Features that DO have outlines keep their
    # real geometry; only bare nodes get the disc.
    # ---------------------------------------------------------------------
    # Healthcare access -> checkups, screening, vaccination, insurance
    "hospitals": {"specs": ["amenity=hospital"], "point_radius_px": 6.0},
    "clinics": {
        "specs": [
            "amenity=clinic",
            "amenity=doctors",
            "healthcare=centre",
            "healthcare=clinic",
            "healthcare=doctor",
        ],
        "point_radius_px": 3.0,
    },
    "pharmacies": {
        "specs": ["amenity=pharmacy", "healthcare=pharmacy"],
        "point_radius_px": 2.0,
    },
    "dentists": {
        "specs": ["amenity=dentist", "healthcare=dentist"],
        "point_radius_px": 2.0,
    },
    "healthcare": {
        "specs": [
            "amenity=hospital",
            "amenity=clinic",
            "amenity=doctors",
            "amenity=pharmacy",
            "amenity=dentist",
            "healthcare=centre",
        ],
        "point_radius_px": 3.0,
    },
    "eldercare": {
        "specs": [
            "amenity=nursing_home",
            "social_facility=nursing_home",
            "social_facility=assisted_living",
            "amenity=social_facility",
        ],
        "point_radius_px": 4.0,
    },
    # Food environment -> diet, obesity, diabetes, food insecurity
    "supermarkets": {
        "specs": ["shop=supermarket", "shop=greengrocer", "shop=wholesale"],
        "point_radius_px": 4.0,
    },
    "convenience": {
        "specs": ["shop=convenience", "shop=kiosk"],
        "point_radius_px": 2.0,
    },
    "fast_food": {"specs": ["amenity=fast_food"], "point_radius_px": 2.0},
    "restaurants": {
        "specs": ["amenity=restaurant", "amenity=cafe"],
        "point_radius_px": 2.0,
    },
    "food_retail": {
        "specs": ["shop=supermarket", "shop=greengrocer", "shop=convenience"],
        "point_radius_px": 3.0,
    },
    "food_growing": {
        "specs": ["landuse=allotments", "amenity=marketplace"],
        "point_radius_px": 3.0,
    },
    # Substance availability -> binge drinking, smoking
    "alcohol_tobacco": {
        "specs": [
            "amenity=bar",
            "amenity=pub",
            "amenity=nightclub",
            "shop=alcohol",
            "shop=tobacco",
            "shop=wine",
        ],
        "point_radius_px": 2.0,
    },
    # Physical activity infrastructure -> inactivity, obesity, CHD
    "playgrounds": {"specs": ["leisure=playground"], "point_radius_px": 2.0},
    "sports": {
        "specs": [
            "leisure=pitch",
            "leisure=sports_centre",
            "leisure=track",
            "leisure=stadium",
            "leisure=swimming_pool",
            "leisure=golf_course",
        ],
        "point_radius_px": 3.0,
    },
    "fitness": {
        "specs": ["leisure=fitness_centre", "leisure=fitness_station"],
        "point_radius_px": 2.0,
    },
    "trails": {
        "specs": ["highway=path", "highway=bridleway", "highway=cycleway"],
        "line_width_px": 1.0,
        "tiles": 2,
    },
    # Traffic and industry -> asthma, COPD, CHD via air quality and noise
    "major_roads": {
        "specs": [
            "highway=motorway",
            "highway=trunk",
            "highway=primary",
            "highway=secondary",
        ],
        "line_width_px": 2.5,
    },
    "railways": {
        "specs": ["railway=rail", "railway=light_rail", "railway=subway"],
        "line_width_px": 1.5,
    },
    "industrial": {
        "specs": ["landuse=industrial", "man_made=works", "power=plant"],
        "point_radius_px": 4.0,
    },
    "waste_sites": {
        "specs": [
            "landuse=landfill",
            "landuse=quarry",
            "man_made=wastewater_plant",
        ],
        "point_radius_px": 4.0,
    },
    "airports": {"specs": ["aeroway=aerodrome", "aeroway=runway"]},
    # Transit -> access barriers, and a strong correlate of density
    "transit": {
        "specs": [
            "highway=bus_stop",
            "railway=station",
            "railway=tram_stop",
            "public_transport=station",
        ],
        "point_radius_px": 2.0,
    },
    # Social infrastructure -> mental health, social isolation
    "schools": {
        "specs": ["amenity=school", "amenity=kindergarten"],
        "point_radius_px": 4.0,
    },
    "higher_ed": {
        "specs": ["amenity=college", "amenity=university"],
        "point_radius_px": 5.0,
    },
    "civic": {
        "specs": [
            "amenity=library",
            "amenity=community_centre",
            "amenity=social_centre",
            "amenity=place_of_worship",
            "amenity=post_office",
        ],
        "point_radius_px": 3.0,
    },
    # Land in transition or reserved -> disadvantage proxies, and green space
    # that is not publicly usable
    "vacant_land": {
        "specs": ["landuse=brownfield", "landuse=greenfield", "landuse=construction"]
    },
    "cemeteries": {"specs": ["landuse=cemetery", "amenity=grave_yard"]},
}

# Land-cover labels served from the WorldCover raster. Wall-to-wall coverage, one
# HTTP range read per chip, no rate limit -- and no volunteer-mapping gap, which
# is what made the OSM versions unusable on rural chips.
WORLDCOVER_LABELS = tuple(worldcover.CLASSES)

# The hybrid default. Land cover from WorldCover, discrete built features from
# OSM, because no raster distinguishes a parking apron from a warehouse roof.
# This is also what keeps OSM usable: three groups instead of eight means nine
# sub-queries per chip rather than fourteen, comfortably inside fair use.
#
# These layers overlap and do not partition the chip -- a patch can be both
# canopy and park -- and WorldCover assigns every pixel a class, so its groups
# together do cover the whole chip while the OSM ones do not.
DEFAULT_LABELS = (
    "canopy,grass,farmland,agriculture,water,built,"
    "buildings,roads,parking,walkability,food_retail,healthcare"
)


def _chip_geometry(
    scene_dir: Path, lat: float, lon: float, chip_px: int
) -> tuple[Any, Any, int, int, int, int]:
    """Locate the chip window inside a scene.

    Args:
        scene_dir: Directory holding one ``*.SAFE``.
        lat: Chip centre latitude.
        lon: Chip centre longitude.
        chip_px: Chip side in pixels.

    Returns:
        ``(crs, transform, row0, col0, height, width)`` -- the scene's CRS and
        affine, the chip's top-left pixel, and the scene's full pixel extent.
        The extent is needed to reproject other rasters onto this same grid.

    Raises:
        typer.BadParameter: If no B02 band is found, or the chip falls outside.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    matches = sorted(scene_dir.glob("*.SAFE/GRANULE/*/IMG_DATA/*/*_B02_*.jp2"))
    if not matches:
        raise typer.BadParameter(
            f"no B02 band under {scene_dir}. Fetch a scene first: "
            f"python scripts/fetch_scene.py --location <name>"
        )
    # B02 at 10 m defines the reference grid, matching tiles.SafeSceneSource.
    with rasterio.open(matches[0]) as src:
        crs, affine = src.crs, src.transform
        height, width = src.height, src.width
        xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])

    row0, col0 = int(row) - chip_px // 2, int(col) - chip_px // 2
    if not (0 <= row0 and row0 + chip_px <= height):
        raise typer.BadParameter(f"chip rows {row0}..{row0 + chip_px} outside scene")
    if not (0 <= col0 and col0 + chip_px <= width):
        raise typer.BadParameter(f"chip cols {col0}..{col0 + chip_px} outside scene")
    return crs, affine, row0, col0, height, width


def _chip_bounds_lonlat(
    crs: Any, affine: Any, row0: int, col0: int, chip_px: int
) -> tuple[float, float, float, float]:
    """Return the chip's ``(latmin, lonmin, latmax, lonmax)`` for an OSM query.

    All four corners are projected, not just two: the chip is axis-aligned in
    UTM, so in lat/lon it is a slightly rotated quadrilateral and taking two
    corners would clip real features near the edges.
    """
    from rasterio.warp import transform as warp_transform

    corners = [
        affine * (col0, row0),
        affine * (col0 + chip_px, row0),
        affine * (col0, row0 + chip_px),
        affine * (col0 + chip_px, row0 + chip_px),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
    return min(lats), min(lons), max(lats), max(lons)


def _to_local_pixels(
    ring: list[tuple[float, float]], crs: Any, affine: Any, row0: int, col0: int
) -> list[tuple[float, float]]:
    """Convert a lat/lon ring to chip-local fractional pixel coordinates."""
    from rasterio.warp import transform as warp_transform

    lats = [p[0] for p in ring]
    lons = [p[1] for p in ring]
    xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
    inverse = ~affine
    out: list[tuple[float, float]] = []
    for x, y in zip(xs, ys, strict=True):
        col, row = inverse * (x, y)
        out.append((col - col0, row - row0))
    return out


def _disc(col: float, row: float, radius: float, vertices: int = 12) -> list[tuple]:
    """Approximate a circle in pixel coordinates, for a point feature."""
    step = 2.0 * np.pi / vertices
    ring = [
        (col + radius * np.cos(i * step), row + radius * np.sin(i * step))
        for i in range(vertices)
    ]
    ring.append(ring[0])
    return ring


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Thicken a mask by a pixel radius.

    Lines rasterise one pixel wide whatever width they represent. A footway is
    narrower than a 10 m pixel so that is roughly right, but a dual carriageway
    is not, and ``line_width_px`` was previously read and then never applied.
    Padded shifts rather than ``np.roll`` so the chip edge does not wrap onto the
    opposite side.

    Args:
        mask: Binary mask.
        radius: Dilation radius in pixels. Below 1 the mask is returned as is.

    Returns:
        The dilated mask.
    """
    r = int(round(radius))
    if r < 1:
        return mask
    padded = np.pad(mask, r, mode="constant")
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            window = padded[
                r + dy : r + dy + mask.shape[0], r + dx : r + dx + mask.shape[1]
            ]
            out = np.maximum(out, window)
    return out


def _rasterize(
    shapes: list[dict[str, Any]],
    chip_px: int,
    patch_px: int,
    dilate_px: float = 0.0,
) -> np.ndarray:
    """Burn shapes at pixel resolution, then pool to per-patch fractions.

    Args:
        shapes: GeoJSON-like geometries in chip-local pixel coordinates.
        chip_px: Chip side in pixels.
        patch_px: Pixels per patch.

    Returns:
        ``(side, side)`` float32 coverage fractions in [0, 1].
    """
    from affine import Affine
    from rasterio.features import rasterize

    side = chip_px // patch_px
    if not shapes:
        return np.zeros((side, side), dtype=np.float32)

    # Identity transform: the geometries are already in pixel space.
    burned = rasterize(
        [(shape, 1) for shape in shapes],
        out_shape=(chip_px, chip_px),
        transform=Affine.identity(),
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    if dilate_px >= 1.0:
        burned = _dilate(burned, dilate_px)
    blocks = burned.reshape(side, patch_px, side, patch_px)
    return blocks.mean(axis=(1, 3)).astype(np.float32)


def _shapes_for(
    elements: list[dict[str, Any]],
    group: dict[str, Any],
    crs: Any,
    affine: Any,
    row0: int,
    col0: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convert Overpass elements into pixel-space geometries.

    Returns:
        ``(shapes, feature_count)``.
    """
    width = group.get("line_width_px")
    point_radius = group.get("point_radius_px")
    shapes: list[dict[str, Any]] = []
    used = 0
    for element in elements:
        # A group can legitimately contain all three geometries at once:
        # `healthcare` matches hospital campuses (polygons) and pharmacies
        # (points), and dropping either would misrepresent the label.
        if element.get("type") == "node":
            if point_radius is None:
                continue
            lat, lon = element.get("lat"), element.get("lon")
            if lat is None or lon is None:
                continue
            col, row = _to_local_pixels(
                [(float(lat), float(lon))], crs, affine, row0, col0
            )[0]
            used += 1
            shapes.append(
                {"type": "Polygon", "coordinates": [_disc(col, row, point_radius)]}
            )
            continue

        if width is None:
            found = osm.rings(element)
            kind = "Polygon"
        else:
            found = osm.lines(element)
            kind = "LineString"
        if not found:
            continue
        used += 1
        for ring in found:
            local = _to_local_pixels(ring, crs, affine, row0, col0)
            if kind == "Polygon":
                if local[0] != local[-1]:
                    local.append(local[0])
                shapes.append({"type": "Polygon", "coordinates": [local]})
            else:
                shapes.append({"type": "LineString", "coordinates": local})
    return shapes, used


def _split_bbox(
    bounds: tuple[float, float, float, float], n: int
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into an n x n grid of sub-boxes.

    Args:
        bounds: ``(latmin, lonmin, latmax, lonmax)``.
        n: Divisions per axis. 1 returns the original box.

    Returns:
        Sub-boxes, each slightly overlapped so a feature exactly on an internal
        boundary is not missed by both neighbours.
    """
    if n <= 1:
        return [bounds]
    latmin, lonmin, latmax, lonmax = bounds
    dlat = (latmax - latmin) / n
    dlon = (lonmax - lonmin) / n
    pad_lat, pad_lon = dlat * 0.01, dlon * 0.01
    out: list[tuple[float, float, float, float]] = []
    for i in range(n):
        for j in range(n):
            out.append(
                (
                    latmin + i * dlat - pad_lat,
                    lonmin + j * dlon - pad_lon,
                    latmin + (i + 1) * dlat + pad_lat,
                    lonmin + (j + 1) * dlon + pad_lon,
                )
            )
    return out


def label_source(label: str) -> str:
    """Return which backend owns a label name."""
    return "worldcover" if label in WORLDCOVER_LABELS else "osm"


# Shown wherever a label appears, because the two backends measure different
# things under confusingly similar names. WorldCover ``built`` is every
# impervious surface -- roofs, roads, parking, driveways -- while OSM
# ``buildings`` is roof outlines alone, and on the same chip they came back 0.607
# and 0.171. A reader who mistakes one for the other draws the wrong conclusion.
SOURCE_TAGS = {"worldcover": "WorldCover", "osm": "OSM"}


def display_label(label: str) -> str:
    """Return a label name annotated with its source, e.g. ``WorldCover-built``.

    The stored key keeps its ``osm_`` prefix -- it distinguishes ``osm_canopy``
    from ``canopy`` in the data -- but the prefix is dropped for display, since
    the source tag already carries it and ``OSM-osm_canopy`` reads badly.
    """
    tag = SOURCE_TAGS[label_source(label)]
    shown = label[4:] if label.startswith("osm_") else label
    return f"{tag}-{shown}"


def _existing_labels(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load previously written label grids, so a partial run can be resumed.

    Overpass refuses connections outright once a burst trips its fair-use
    limits, which leaves some groups fetched and others not. Re-fetching the
    ones that already succeeded would spend exactly the requests that caused the
    block, so they are kept and only the gaps are retried.

    Args:
        path: An existing ``*_labels.npz``, or a path that does not exist.

    Returns:
        ``(grids, feature_counts)``, both empty if there is nothing to load.
    """
    if not path.exists():
        return {}, {}
    try:
        with np.load(path, allow_pickle=True) as data:
            names = [str(v) for v in data["label_names"]]
            grids = data["labels"]
            counts = data["feature_counts"]
            recorded = (
                [str(v) for v in data["label_sources"]]
                if "label_sources" in data
                else []
            )
    except (OSError, KeyError, ValueError):
        return {}, {}

    keep_grids: dict[str, np.ndarray] = {}
    keep_counts: dict[str, int] = {}
    for i, name in enumerate(names):
        # A label written from a different backend than currently owns that name
        # must not be reused. `canopy` meant OSM woodland polygons before and
        # means the WorldCover tree class now; keeping the old array would leave
        # one column of the correlation table holding two different measurements
        # depending on which location it came from.
        was = recorded[i] if i < len(recorded) else "osm"
        if was != label_source(name):
            logger.info(
                "discarding cached %r (was %s, now %s) -- it will be refetched",
                name,
                was,
                label_source(name),
            )
            continue
        keep_grids[name] = grids[i]
        keep_counts[name] = int(counts[i])
    return keep_grids, keep_counts


def _fetch_group_local(
    group: dict[str, Any],
    bounds: tuple[float, float, float, float],
    extract: Path,
) -> list[dict[str, Any]]:
    """Fetch one label group from a local extract.

    Returns Overpass-shaped elements, so the geometry handling below is shared
    with the API path and cannot drift between the two backends.
    """
    return osm_local.read_features(
        extract,
        bounds,
        group["specs"],
        want_points=group.get("point_radius_px") is not None,
    )


def _fetch_group(
    group: dict[str, Any],
    bounds: tuple[float, float, float, float],
    tiles: int,
    timeout: int,
    pause: float,
    url: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch one label group, tiling the bbox if asked, de-duplicating results.

    Args:
        group: A LABEL_GROUPS entry.
        bounds: The chip bbox.
        tiles: Divisions per axis.
        timeout: Per-query timeout.
        pause: Seconds between sub-queries.

    Returns:
        Unique Overpass elements.
    """
    elements: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    boxes = _split_bbox(bounds, tiles)
    for index, box in enumerate(boxes):
        query = osm.build_query(
            group["specs"],
            bbox=box,
            geometry=True,
            timeout=timeout,
            # Nodes are requested only for groups with point_radius_px, since
            # otherwise they contribute no area and only inflate the response.
            kinds=(
                ("node", "way", "relation")
                if group.get("point_radius_px") is not None
                else ("way", "relation")
            ),
        )
        for element in osm.run(query, timeout, url=url):
            key = (element.get("type"), element.get("id"))
            if key in seen:
                continue
            seen.add(key)
            elements.append(element)
        if index < len(boxes) - 1:
            time.sleep(pause)
    return elements


def _write_figure(
    out_path: Path,
    chip_npz: Path,
    grids: dict[str, np.ndarray],
    name: str,
) -> bool:
    """Draw the RGB chip beside each label mask, for eyeball validation.

    Printed coverage numbers cannot reveal a mask that is offset, flipped or
    transposed -- all three produce entirely plausible-looking percentages. Only
    seeing the mask land on the right ground rules them out, and a systematic
    one-patch offset would quietly weaken every correlation computed later.

    Args:
        out_path: PNG to write.
        chip_npz: The ``*_chip.npz`` saved during encoding.
        grids: Label name to ``(side, side)`` coverage.
        name: Location name, for the title.

    Returns:
        True if written, False if the chip file was missing.
    """
    if not chip_npz.exists():
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(chip_npz) as data:
        chip = data["chip"]

    # Band order is B02 B03 B04 B08 ... so RGB is indices 2, 1, 0. Percentile
    # stretch rather than max: a few bright roofs would otherwise darken
    # everything else into a uniform grey.
    rgb = chip[:, :, [2, 1, 0]].astype(np.float32)
    low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)

    order = sorted(grids)
    cols = len(order) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3.1 * cols, 3.6), squeeze=False)

    axes[0][0].imshow(rgb)
    axes[0][0].set_title("RGB chip", fontsize=10)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    for i, label in enumerate(order, start=1):
        ax = axes[0][i]
        # extent maps the coarse label grid onto the chip's pixel coordinates so
        # the panels are directly comparable despite different resolutions.
        ax.imshow(rgb, alpha=0.55)
        ax.imshow(
            grids[label],
            cmap="viridis",
            alpha=0.65,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=(0, chip.shape[1], chip.shape[0], 0),
        )
        ax.set_title(
            f"{display_label(label)}\nmean {float(grids[label].mean()):.3f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{name}: label coverage over imagery "
        f"(check the mask lands on the right ground)",
        fontsize=11,
    )
    # Reserve the top strip for the suptitle; with a single row of panels
    # tight_layout otherwise lets it collide with the per-panel titles.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


@app.command()
def main(
    location: Optional[str] = typer.Option(None, help="One location name."),
    all_locations: bool = typer.Option(
        False, "--all", help="Every location that already has an encoded artifact."
    ),
    labels: str = typer.Option(
        DEFAULT_LABELS,
        help=(
            "Comma-separated label groups. WorldCover raster: "
            f"{', '.join(WORLDCOVER_LABELS)}. OSM vector: "
            f"{', '.join(LABEL_GROUPS)}."
        ),
    ),
    artifacts: Path = typer.Option(
        Path("artifacts"), help="Where encoded results live."
    ),
    config: Optional[Path] = typer.Option(
        None, help="Run config, for chip_px and patch_px. Must match the encode run."
    ),
    registry: Optional[Path] = typer.Option(None, help="Alternative locations YAML."),
    pause: float = typer.Option(
        4.0,
        help=(
            "Seconds between Overpass queries. 4 s keeps six label groups "
            "inside the public instance's rate limit; 1 s reliably triggers "
            "HTTP 429."
        ),
    ),
    tiles: int = typer.Option(
        0,
        help=(
            "Override the per-group bbox split. 0 uses each group's own "
            "setting (4 for buildings, 2 for roads and parking). Raise it if "
            "queries still time out with 504."
        ),
    ),
    timeout: int = typer.Option(180, help="Per-query Overpass timeout, seconds."),
    cache_dir: Optional[Path] = typer.Option(
        None,
        help=(
            "Read WorldCover tiles from this local directory instead of over "
            "the network. Use it if GDAL here lacks curl support, or to avoid "
            "re-reading the same tile for every location."
        ),
    ),
    wc_version: str = typer.Option(
        worldcover.DEFAULT_VERSION, help="WorldCover product version."
    ),
    wc_year: str = typer.Option(
        worldcover.DEFAULT_YEAR, help="WorldCover product year."
    ),
    mirror: str = typer.Option(
        "main",
        help=(
            f"Overpass instance: {', '.join(osm.MIRRORS)}, or a full URL. "
            f"Switch to 'kumi' if the main one starts refusing connections."
        ),
    ),
    osm_source: str = typer.Option(
        "auto",
        help=(
            "Where OSM landmarks come from. 'local' reads a .osm.pbf from "
            "data/osm and never touches the network. 'api' always uses "
            "Overpass. 'auto' prefers a local extract covering the chip and "
            "falls back to Overpass."
        ),
    ),
    osm_pbf: Optional[Path] = typer.Option(
        None, help="Use this specific extract instead of auto-selecting one."
    ),
    check_mirrors: bool = typer.Option(
        False,
        "--check-mirrors",
        help=(
            "Probe every Overpass instance with a trivial query and report "
            "which respond, then exit. Fair-use blocks are per-instance and "
            "time-based, so when one refuses connections another usually works."
        ),
    ),
    keep_empty: bool = typer.Option(
        False,
        "--keep-empty",
        help=(
            "Trust cached groups that found zero features instead of retrying "
            "them. Only useful once you are confident the zero is real."
        ),
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help=(
            "Re-fetch groups that already succeeded. By default they are kept "
            "from the existing labels file and only missing groups are fetched, "
            "so a run interrupted by rate limiting resumes cheaply."
        ),
    ),
    figures: bool = typer.Option(
        True,
        help=(
            "Write a PNG of each label mask over the RGB chip. Worth keeping "
            "on: coverage percentages look plausible even when a mask is "
            "offset or flipped."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch OSM polygons and write per-patch label coverage for each chip."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    cfg = RunConfig.from_yaml(config) if config else RunConfig()
    chip_px, patch_px = cfg.tile.chip_px, cfg.tile.patch_px
    side = chip_px // patch_px

    wanted = [name.strip() for name in labels.split(",") if name.strip()]
    known = set(LABEL_GROUPS) | set(WORLDCOVER_LABELS)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise typer.BadParameter(
            f"unknown label group(s) {unknown}.\n"
            f"  WorldCover (raster): {', '.join(sorted(WORLDCOVER_LABELS))}\n"
            f"  OSM (vector):        {', '.join(sorted(LABEL_GROUPS))}"
        )
    wc_wanted = [n for n in wanted if n in WORLDCOVER_LABELS]
    osm_wanted = [n for n in wanted if n in LABEL_GROUPS]

    if check_mirrors:
        # One tiny query each, no retries: the point is a fast verdict on which
        # instance will accept work right now, not to get data.
        probe = osm.build_query(
            ["amenity=cafe"], bbox=(38.900, -77.040, 38.905, -77.035), timeout=10
        )
        typer.echo("probing Overpass instances...\n")
        for label_name, url in osm.MIRRORS.items():
            started = time.monotonic()
            try:
                found = osm.run(probe, 20, retries=0, url=url)
                typer.echo(
                    f"  {label_name:8s} OK       "
                    f"{time.monotonic() - started:5.1f}s  "
                    f"{len(found)} element(s)  {url}"
                )
            except osm.OverpassError as exc:
                reason = str(exc).split(" -- ")[0][:70]
                typer.echo(f"  {label_name:8s} FAILED   {reason}")
        typer.echo(
            "\nUse a working one with --mirror <name>. If all are refusing, the "
            "block is time-based:\nwait 15-60 minutes. The WorldCover labels "
            "need none of this and can be fetched meanwhile:\n"
            "  --labels canopy,grass,farmland,agriculture,water,built"
        )
        return

    if osm_source not in ("auto", "local", "api"):
        raise typer.BadParameter("--osm-source must be auto, local or api")

    endpoint = osm.MIRRORS.get(mirror, mirror)
    if not endpoint.startswith("http"):
        raise typer.BadParameter(
            f"--mirror must be one of {', '.join(osm.MIRRORS)} or a URL"
        )

    registry_sites = loc.load_locations(registry)
    if all_locations:
        names = [n for n in registry_sites if (artifacts / n / f"{n}.npz").exists()]
        if not names:
            raise typer.BadParameter(
                f"no encoded artifacts found under {artifacts}. Run "
                f"encode_location.py first."
            )
    elif location:
        names = [location]
    else:
        raise typer.BadParameter("pass --location or --all")

    typer.echo(
        f"labelling {len(names)} location(s) at {side}x{side} patches "
        f"({patch_px * cfg.tile.input_resolution_m} m each)\n"
        f"groups: {', '.join(wanted)}\n"
    )

    for index, name in enumerate(names, start=1):
        site = registry_sites.get(name) or loc.resolve(name, registry)
        typer.echo(f"[{index}/{len(names)}] {name}")
        crs, affine, row0, col0, scene_h, scene_w = _chip_geometry(
            loc.scene_dir_for(name), site.lat, site.lon, chip_px
        )
        bounds = _chip_bounds_lonlat(crs, affine, row0, col0, chip_px)
        logger.debug("chip bounds (latmin, lonmin, latmax, lonmax): %s", bounds)

        out_dir = artifacts / name
        label_path = out_dir / f"{name}_labels.npz"
        grids, counts = ({}, {}) if refresh else _existing_labels(label_path)

        # A cached group that found zero features is retried by default. Resume
        # otherwise makes a bug permanent: a broken reader writes zeros, the
        # next run sees them as "already fetched", and the fix never gets a
        # chance to run. Raster groups are exempt -- they record -1, not a count
        # -- and a genuinely empty group simply comes back empty again, cheaply,
        # now that the data is local.
        if not refresh and not keep_empty:
            # Not `name`: that is the location driving the outer loop, and
            # rebinding it here sent the figure looking for
            # `walkability_chip.npz` instead of `westminster_chip.npz`.
            empty = [
                group_key
                for group_key, count in counts.items()
                if count == 0 and group_key in wanted
            ]
            for group_key in empty:
                grids.pop(group_key, None)
                counts.pop(group_key, None)
            if empty:
                typer.echo(
                    f"    retrying {len(empty)} cached group(s) that found "
                    f"nothing: {', '.join(sorted(empty))}"
                )
        if grids:
            kept = [g for g in grids if g in wanted]
            if kept:
                typer.echo(
                    f"    keeping {len(kept)} group(s) already fetched: "
                    f"{', '.join(sorted(kept))}"
                )

        # WorldCover first: one range read covers every land-cover label at
        # once, so it costs the same whether one class is wanted or all seven.
        wc_todo = [n for n in wc_wanted if n not in grids]
        if wc_todo:
            try:
                # Named tile_names, not names: `names` is the outer list of
                # locations driving this loop, and shadowing it made the
                # progress counter report [2/1].
                tile_names = worldcover.tiles_for_bbox(bounds)
                if len(tile_names) > 1:
                    logger.warning(
                        "chip spans %d WorldCover tiles (%s); using the first, "
                        "so part of the chip may read as no-data",
                        len(tile_names),
                        ", ".join(tile_names),
                    )
                tile = tile_names[0]
                source = (
                    str(worldcover.local_path(cache_dir, tile, wc_version, wc_year))
                    if cache_dir is not None
                    else worldcover.tile_url(tile, wc_version, wc_year)
                )
                classes = worldcover.read_class_grid(
                    source, crs, affine, row0, col0, chip_px, scene_h, scene_w
                )
                nodata = float((classes == 0).mean())
                if nodata > 0.01:
                    typer.echo(
                        f"    {'WorldCover':24s} WARNING {nodata * 100:.1f}% "
                        f"no-data -- chip may cross a tile edge"
                    )
                composition = worldcover.describe(classes)
                typer.echo(
                    f"    {'WorldCover tile':24s} {tile}: "
                    + ", ".join(f"{n} {f * 100:.0f}%" for n, f in composition[:4])
                )
                for label, grid in worldcover.coverage_grids(
                    classes, wc_todo, patch_px
                ).items():
                    grids[label] = grid
                    counts[label] = -1  # raster: no discrete feature count
                    typer.echo(
                        f"    {display_label(label):24s} {'raster':>6s}  "
                        f"{float((grid > 0).mean()) * 100:5.1f}% of patches "
                        f"touched  mean coverage {float(grid.mean()):.4f}"
                    )
            except (RuntimeError, OSError) as exc:
                typer.echo(f"    {'WorldCover':24s} FAILED: {exc}")

        # Resolve the OSM backend per location: extracts are regional, so a
        # location outside every downloaded extract still needs the API.
        extract: Path | None = None
        if osm_source in ("auto", "local") and osm_wanted:
            extract = osm_pbf or osm_local.find_extract(bounds)
            if extract is None and osm_source == "local":
                typer.echo(
                    f"    {'OSM':24s} SKIPPED: no extract in "
                    f"{osm_local.extracts_dir()} covers this chip. Run "
                    f"scripts/fetch_reference_data.py --osm"
                )
                osm_wanted_here: list[str] = []
            else:
                osm_wanted_here = list(osm_wanted)
                if extract is not None:
                    typer.echo(f"    {'OSM extract':24s} {extract.name}")
        else:
            osm_wanted_here = list(osm_wanted)

        for group_name in osm_wanted_here:
            if group_name in grids:
                continue
            group = LABEL_GROUPS[group_name]
            group_tiles = tiles if tiles > 0 else int(group.get("tiles", 1))
            try:
                if extract is not None:
                    elements = _fetch_group_local(group, bounds, extract)
                else:
                    elements = _fetch_group(
                        group, bounds, group_tiles, timeout, pause, endpoint
                    )
            except (osm.OverpassError, RuntimeError) as exc:
                typer.echo(f"    {display_label(group_name):24s} FAILED: {exc}")
                continue
            shapes, used = _shapes_for(elements, group, crs, affine, row0, col0)
            grid = _rasterize(
                shapes, chip_px, patch_px, group.get("line_width_px", 0.0)
            )
            grids[group_name] = grid
            counts[group_name] = used
            covered = float((grid > 0).mean())
            typer.echo(
                f"    {display_label(group_name):24s} {used:5d} feat  "
                f"{covered * 100:5.1f}% of patches touched  "
                f"mean coverage {float(grid.mean()):.4f}"
            )
            time.sleep(pause)

        if not grids:
            typer.echo("    nothing written")
            continue

        # How much ground carries any label at all. The groups overlap, so this
        # is the per-patch maximum rather than a sum -- and it is the honest
        # check on whether the label set describes this chip or leaves most of
        # it blank. A low number means correlations are being computed against
        # a mostly-empty target.
        stacked = np.stack([grids[k] for k in sorted(grids)], axis=0)
        any_label = stacked.max(axis=0)
        typer.echo(
            f"    {'ANY label (all sources)':24s}        "
            f"{float((any_label > 0).mean()) * 100:5.1f}% of patches touched  "
            f"mean coverage {float(any_label.mean()):.4f}"
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        path = label_path
        np.savez_compressed(
            path,
            label_names=np.array(sorted(grids)),
            label_sources=np.array([label_source(k) for k in sorted(grids)]),
            labels=np.stack([grids[k] for k in sorted(grids)], axis=0),
            feature_counts=np.array([counts[k] for k in sorted(grids)]),
            grid_shape=np.array([side, side]),
            chip_px=chip_px,
            patch_px=patch_px,
            lat=site.lat,
            lon=site.lon,
            chip_bounds=np.array(bounds),
            config_fingerprint=cfg.fingerprint(),
        )
        (out_dir / f"{name}_labels.json").write_text(
            json.dumps(
                {
                    "location": name,
                    "sources": {k: label_source(k) for k in sorted(grids)},
                    "osm_groups": {
                        k: LABEL_GROUPS[k]["specs"]
                        for k in sorted(grids)
                        if k in LABEL_GROUPS
                    },
                    "worldcover_classes": {
                        k: list(worldcover.CLASSES[k])
                        for k in sorted(grids)
                        if k in worldcover.CLASSES
                    },
                    "feature_counts": counts,
                    "chip_bounds_latmin_lonmin_latmax_lonmax": list(bounds),
                    "config_fingerprint": cfg.fingerprint(),
                },
                indent=2,
            )
        )
        typer.echo(f"    wrote {path}")

        if figures:
            figure_path = out_dir / "figures" / f"{name}_labels.png"
            if _write_figure(figure_path, out_dir / f"{name}_chip.npz", grids, name):
                typer.echo(f"    wrote {figure_path}")
            else:
                typer.echo(
                    f"    no {name}_chip.npz, skipping figure -- re-run "
                    f"encode_location.py to get one"
                )
        typer.echo("")

    typer.echo(
        "Next:\n  python scripts/correlate_labels.py --all"
    )


if __name__ == "__main__":
    app()
