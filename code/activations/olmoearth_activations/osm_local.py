"""Read OSM features from a local ``.osm.pbf`` extract instead of Overpass.

Why
---
Overpass is a free shared service with a concurrency limit, and a run that asks
for twelve labels across seven chips is bursty enough to get an IP refused
outright. A regional extract removes the dependency entirely: download once,
query locally as often as you like.

It is also better science. Overpass answers from a live database that changes
under you, so a label built today is not reproducible tomorrow. An extract is a
dated snapshot, and the date is in the filename.

Integration
-----------
:func:`read_features` returns dicts shaped exactly like Overpass elements --
``{"type", "id", "geometry"|"lat"/"lon", "tags"}`` -- so everything downstream
(ring extraction, line widening, point discs, rasterisation) works unchanged.
The backend is swappable precisely because the boundary is this one shape.

Layers in GDAL's OSM driver, and what each is used for here:

``points``
    Nodes with tags. The only source for a pharmacy or shop mapped as a bare
    point, which is most of them.
``lines``
    Ways. Roads and footpaths.
``multipolygons``
    Closed ways and multipolygon relations, already assembled into areas -- which
    is the work this saves over parsing raw OSM.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# GDAL's default osmconf.ini promotes these keys to real columns; anything else
# lands in the `other_tags` hstore string and has to be matched by text.
PROMOTED_KEYS = frozenset(
    {
        "aeroway",
        "amenity",
        "barrier",
        "boundary",
        "building",
        "craft",
        "geological",
        "highway",
        "historic",
        "land_area",
        "landuse",
        "leisure",
        "man_made",
        "military",
        "natural",
        "office",
        "place",
        "shop",
        "sport",
        "tourism",
    }
)

LAYERS = ("points", "lines", "multipolygons")


def extracts_dir() -> Path:
    """Return the directory holding downloaded ``.osm.pbf`` extracts."""
    from olmoearth_activations.locations import repo_root

    root = repo_root()
    return (root / "data" / "osm") if root else Path("data/osm")


def available_extracts() -> list[Path]:
    """Return every extract on disk, newest filename last."""
    directory = extracts_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.osm.pbf"))


def extract_bounds(path: Path) -> tuple[float, float, float, float] | None:
    """Return an extract's ``(lonmin, latmin, lonmax, latmax)``, or None.

    Read from the file's own metadata so extracts can be matched to locations
    without hardcoding which state covers what.
    """
    try:
        import pyogrio
    except ImportError:
        return None
    try:
        info = pyogrio.read_info(str(path), layer="multipolygons")
    except Exception as exc:  # noqa: BLE001 - depends on GDAL build
        logger.debug("could not read info from %s: %s", path, exc)
        return None
    bounds = info.get("total_bounds")
    if bounds is None or len(bounds) != 4:
        return None
    return tuple(float(v) for v in bounds)  # type: ignore[return-value]


def find_extract(bbox: tuple[float, float, float, float]) -> Path | None:
    """Return the extract covering a chip, or None.

    Args:
        bbox: ``(latmin, lonmin, latmax, lonmax)`` -- the chip bbox convention
            used elsewhere in this package, not GDAL's lon-first order.

    Returns:
        The smallest covering extract, so a city extract wins over a state one
        and reads stay fast.
    """
    latmin, lonmin, latmax, lonmax = bbox
    covering: list[tuple[float, Path]] = []
    for path in available_extracts():
        bounds = extract_bounds(path)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        if x0 <= lonmin and x1 >= lonmax and y0 <= latmin and y1 >= latmax:
            covering.append(((x1 - x0) * (y1 - y0), path))
    if not covering:
        return None
    return min(covering)[1]


def _text(value: Any) -> str:
    """Coerce a field value to a string, treating null-ish values as empty.

    Values arrive from numpy arrays, where a missing string is often ``nan`` --
    which is *truthy*, so a naive check reports every feature as tagged.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        # nan != nan; any float here means "no string value".
        return ""
    text = str(value)
    return "" if text in ("nan", "None", "<NA>") else text


def _matches(row: Any, specs: Iterable[str]) -> bool:
    """True if a feature row satisfies any of the tag specs.

    Checks promoted columns first, then the ``other_tags`` hstore text, because
    GDAL promotes only a fixed set of keys to real columns -- ``healthcare`` and
    ``public_transport`` are not among them, so those specs are only findable in
    the hstore. The hstore is matched loosely on spacing, since its exact
    formatting varies with GDAL version.
    """
    other = _text(row.get("other_tags"))
    for spec in specs:
        if "=" in spec:
            key, value = spec.split("=", 1)
            if _text(row.get(key)) == value:
                return True
            if re.search(
                rf'"{re.escape(key)}"\s*=>\s*"{re.escape(value)}"', other
            ):
                return True
        else:
            if _text(row.get(spec)):
                return True
            if re.search(rf'"{re.escape(spec)}"\s*=>', other):
                return True
    return False


def _element(row: Any, geometry: Any, layer: str, index: int) -> dict[str, Any]:
    """Convert one row into an Overpass-shaped element."""
    osm_id = (
        _text(row.get("osm_id")) or _text(row.get("osm_way_id")) or f"{layer}{index}"
    )
    tags = {
        key: _text(value)
        for key, value in row.items()
        if key in PROMOTED_KEYS and _text(value)
    }

    if layer == "points":
        return {
            "type": "node",
            "id": osm_id,
            "lat": float(geometry.y),
            "lon": float(geometry.x),
            "tags": tags,
        }

    # Overpass reports geometry as a list of {lat, lon}; mirror that exactly so
    # osm.rings / osm.lines need no special case for this backend.
    def ring(coords: Iterable[Any]) -> list[dict[str, float]]:
        return [{"lat": float(y), "lon": float(x)} for x, y in coords]

    kind = "way" if layer == "lines" else "relation"
    if geometry.geom_type in ("Polygon",):
        return {
            "type": kind,
            "id": osm_id,
            "geometry": ring(geometry.exterior.coords),
            "tags": tags,
        }
    if geometry.geom_type in ("MultiPolygon",):
        return {
            "type": "relation",
            "id": osm_id,
            "members": [
                {"geometry": ring(part.exterior.coords)} for part in geometry.geoms
            ],
            "tags": tags,
        }
    if geometry.geom_type in ("LineString",):
        return {
            "type": "way",
            "id": osm_id,
            "geometry": ring(geometry.coords),
            "tags": tags,
        }
    if geometry.geom_type in ("MultiLineString",):
        return {
            "type": "relation",
            "id": osm_id,
            "members": [{"geometry": ring(part.coords)} for part in geometry.geoms],
            "tags": tags,
        }
    return {}


def read_features(
    path: Path,
    bbox: tuple[float, float, float, float],
    specs: list[str],
    want_points: bool = False,
) -> list[dict[str, Any]]:
    """Read matching features from an extract, as Overpass-shaped elements.

    Args:
        path: An ``.osm.pbf`` extract.
        bbox: ``(latmin, lonmin, latmax, lonmax)``.
        specs: Tag specs, ``key=value`` or bare ``key``.
        want_points: Include node-mapped features. Off by default because a
            point has no area and only matters for groups that buffer it.

    Returns:
        Elements ready for the same geometry handling as Overpass results.

    Raises:
        RuntimeError: If pyogrio is missing or the extract cannot be read.
    """
    # pyogrio's raw reader returns numpy arrays plus WKB geometry, which shapely
    # parses directly. read_dataframe() would be tidier but requires geopandas,
    # a large dependency chain for something used only to hand coordinates
    # straight back out again.
    try:
        try:
            from pyogrio.raw import read as ogr_read
        except ImportError:  # pragma: no cover - layout differs across versions
            from pyogrio import read as ogr_read
    except ImportError as exc:
        raise RuntimeError(
            "reading a local extract needs pyogrio:\n  uv pip install pyogrio"
        ) from exc
    try:
        import shapely
    except ImportError as exc:
        raise RuntimeError(
            "reading a local extract needs shapely:\n  uv pip install shapely"
        ) from exc

    latmin, lonmin, latmax, lonmax = bbox
    # GDAL spatial filters are lon-first.
    window = (lonmin, latmin, lonmax, latmax)

    layers = LAYERS if want_points else tuple(x for x in LAYERS if x != "points")
    out: list[dict[str, Any]] = []
    failures: list[str] = []
    read_total = 0
    for layer in layers:
        try:
            # GDAL warns about unclosed rings and polygons it cannot organise on
            # essentially every OSM extract -- crowd-sourced geometry is untidy
            # and GDAL repairs it anyway. Left unsuppressed, the warnings repeat
            # per layer per group and bury the actual feature counts.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                result = ogr_read(str(path), layer=layer, bbox=window)
        except Exception as exc:  # noqa: BLE001 - GDAL raises various types
            # Warning, not debug. Swallowing this quietly makes a broken read
            # indistinguishable from an area that genuinely has no features --
            # every group reports 0 and the run looks successful.
            logger.warning("cannot read layer %r from %s: %s", layer, path.name, exc)
            failures.append(layer)
            continue

        meta, _fids, wkb, field_data = result
        fields = [str(name) for name in meta["fields"]]
        if wkb is None or len(wkb) == 0:
            logger.debug("layer %r: no features in bbox", layer)
            continue
        geometries = shapely.from_wkb(wkb)
        read_total += len(geometries)
        for index, geometry in enumerate(geometries):
            if geometry is None or geometry.is_empty:
                continue
            row = {
                name: (values[index] if index < len(values) else None)
                for name, values in zip(fields, field_data, strict=False)
            }
            if not _matches(row, specs):
                continue
            element = _element(row, geometry, layer, index)
            if element:
                out.append(element)

    if failures and not out:
        raise RuntimeError(
            f"every layer failed to read from {path.name} ({', '.join(failures)}). "
            f"GDAL may lack the OSM driver, or the file may be truncated. Check "
            f"with:\n  python -c \"import pyogrio; "
            f"print(pyogrio.list_layers('{path}'))\""
        )
    if read_total and not out:
        # Read fine, matched nothing: the specs do not line up with how GDAL
        # exposes these tags. That is a different bug from a failed read, and
        # saying so saves debugging the wrong half.
        logger.warning(
            "%s: read %d feature(s) in bbox but none matched %s -- the tag specs "
            "are not matching GDAL's columns or other_tags",
            path.name,
            read_total,
            specs,
        )
    logger.debug(
        "%s: %d of %d feature(s) matched %s", path.name, len(out), read_total, specs
    )
    return out
