"""Named locations, so runs are labelled rather than typed as coordinates.

Two problems this solves.

**Coordinates are not memorable.** ``--lat 38.90324 --lon -77.036964`` says
nothing about where it is, is easy to fat-finger, and a transposed digit lands
you somewhere else entirely with no error -- the encoder happily returns
embeddings for the wrong place. ``--location dc`` cannot silently drift.

**One scene directory cannot hold two scenes.** ``SafeSceneSource`` globs each
band across every ``*.SAFE`` under its directory and takes the first sorted
match, so a second scene does not extend coverage -- it shadows the first.
Alphabetical order decides which, which is how a ``T18SUH`` download beat the
``T18SUJ`` one that actually covered the target. Giving every location its own
subdirectory makes that collision structurally impossible instead of a warning
nobody reads.

This module deliberately depends on nothing heavier than PyYAML, so resolving a
name or a path does not require torch, rasterio, or a checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_REGISTRY = "configs/locations.yaml"

# How far inside a scene's footprint a point must sit before that scene is
# considered usable. A chip is read centred on the point, so a point near the
# edge cannot supply a full one. 0.03 degrees is roughly 3 km, enough for chips
# up to 512 px at 10 m.
DEFAULT_EDGE_MARGIN_DEG = 0.03


@dataclass(frozen=True)
class Location:
    """One named place to encode.

    Attributes:
        name: Short identifier used on the command line and as the scene
            subdirectory name.
        lat: Latitude in degrees (EPSG:4326).
        lon: Longitude in degrees (EPSG:4326).
        stratum: Optional grouping label, e.g. ``urban`` or ``rural``. Carried
            through so comparison runs can group sites without a second file.
        note: Optional free text, for why this point was chosen.
    """

    name: str
    lat: float
    lon: float
    stratum: str | None = None
    note: str | None = None


def repo_root() -> Path | None:
    """Return the repository root, or None if this file is not inside one.

    Found by walking up for a ``.git`` entry rather than counting parent levels,
    so moving this module between package subdirectories does not silently
    change where data is looked for.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def scenes_root() -> Path:
    """Return the directory that holds every location's scene subdirectory."""
    root = repo_root()
    return root / "data" / "scenes" if root else Path("scenes")


def scene_dir_for(name: str) -> Path:
    """Return the scene directory belonging to one location.

    Args:
        name: A location name.

    Returns:
        ``<repo>/data/scenes/<name>``. One scene per location, which is what
        keeps the band glob unambiguous.
    """
    return scenes_root() / name


def scene_footprint(safe: Path) -> tuple[float, float, float, float] | None:
    """Return (latmin, latmax, lonmin, lonmax) for a ``.SAFE``, or None.

    Read from the product's own metadata rather than by opening a band, so
    checking coverage stays a pure-stdlib operation -- no rasterio, no GDAL, and
    usable before any heavy dependency is installed.

    The returned box bounds a footprint that is not axis-aligned in lat/lon,
    since the tiles sit on a UTM grid. It therefore slightly over-claims near
    the corners. The consequence is only an occasional wasted reuse attempt; the
    encode step reports a window falling outside a scene clearly.

    Args:
        safe: Path to a ``*.SAFE`` directory.

    Returns:
        The bounding box, or None if the metadata is missing or unparseable.
    """
    meta = safe / "MTD_MSIL2A.xml"
    if not meta.exists():
        return None
    try:
        text = meta.read_text(errors="replace")
    except OSError:
        return None
    match = re.search(r"<EXT_POS_LIST>(.*?)</EXT_POS_LIST>", text, re.S)
    if not match:
        return None
    try:
        values = [float(v) for v in match.group(1).split()]
    except ValueError:
        return None
    if len(values) < 6:
        return None
    lats, lons = values[0::2], values[1::2]
    return min(lats), max(lats), min(lons), max(lons)


def find_covering_scene(
    lat: float,
    lon: float,
    margin_deg: float = DEFAULT_EDGE_MARGIN_DEG,
    skip: Path | None = None,
) -> Path | None:
    """Find a scene already on disk that covers a point.

    A Sentinel-2 tile is 110 km across, so one download usually covers a whole
    metropolitan area. Fetching a fresh gigabyte per location would multiply
    that by the number of sites without adding any data.

    Args:
        lat: Latitude of interest.
        lon: Longitude of interest.
        margin_deg: Required clearance from the footprint edge.
        skip: A directory to ignore, normally the intended destination.

    Returns:
        The resolved path to a real ``*.SAFE``, or None. Always the real
        directory rather than a symlink pointing at it -- linking to a link
        chains, and the chain breaks if the middle location is removed.
    """
    # Real directories sort before symlinks, so the result is the actual
    # download rather than one of the links to it.
    candidates = sorted(
        scenes_root().glob("*/*.SAFE"), key=lambda p: (p.is_symlink(), str(p))
    )
    for candidate in candidates:
        if skip is not None and candidate.parent.resolve() == skip.resolve():
            continue
        box = scene_footprint(candidate.resolve())
        if box is None:
            continue
        latmin, latmax, lonmin, lonmax = box
        if (
            latmin + margin_deg <= lat <= latmax - margin_deg
            and lonmin + margin_deg <= lon <= lonmax - margin_deg
        ):
            return candidate.resolve()
    return None


def coverage(
    registry: str | Path | None = None,
    margin_deg: float = DEFAULT_EDGE_MARGIN_DEG,
) -> list[tuple[Location, str, Path | None]]:
    """Classify every registry location by whether imagery is available.

    Answers "how many scenes do I still need to download", which is otherwise
    invisible until a run fails on a missing band.

    Args:
        registry: Registry path, or None for the repo default.
        margin_deg: Required clearance from a footprint edge.

    Returns:
        One ``(location, status, scene)`` per entry, where status is:

        ``own``
            The location's own directory already holds a scene.
        ``reuse``
            Another location's scene covers this point, so no download is
            needed -- ``fetch_scene.py`` will symlink it.
        ``missing``
            Nothing on disk covers it.

        ``scene`` is the usable ``*.SAFE`` for the first two, else None.
    """
    out: list[tuple[Location, str, Path | None]] = []
    for site in load_locations(registry).values():
        own = sorted(scene_dir_for(site.name).glob("*.SAFE"))
        if own:
            out.append((site, "own", own[0]))
            continue
        shared = find_covering_scene(site.lat, site.lon, margin_deg)
        out.append((site, "reuse", shared) if shared else (site, "missing", None))
    return out


def registry_path(path: str | Path | None = None) -> Path:
    """Resolve the location registry path.

    Args:
        path: An explicit path, or None to use the repo's default registry.

    Returns:
        The path to a YAML registry file.
    """
    if path is not None:
        return Path(path)
    root = repo_root()
    if root is None:
        return Path(DEFAULT_REGISTRY)
    return root / "code" / "activations" / DEFAULT_REGISTRY


def load_locations(path: str | Path | None = None) -> dict[str, Location]:
    """Load the location registry, keyed by name.

    The file is a YAML list of mappings, the same shape
    ``scripts/compare_locations.py`` already reads, so one file serves both
    single-location lookup and multi-location comparison runs.

    Args:
        path: Registry path, or None for the repo default.

    Returns:
        Mapping of name to :class:`Location`.

    Raises:
        FileNotFoundError: If the registry does not exist.
        ValueError: If the file is not a list of mappings, if an entry is
            missing a required key, or if a name is used twice.
    """
    resolved = registry_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"no location registry at {resolved}. Create it, or pass explicit "
            f"--lat/--lon."
        )
    with resolved.open() as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(f"{resolved}: expected a YAML list of locations")

    out: dict[str, Location] = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{resolved}: entry {i} is not a mapping")
        missing = {"name", "lat", "lon"} - set(entry)
        if missing:
            raise ValueError(
                f"{resolved}: entry {i} is missing {sorted(missing)}"
            )
        name = str(entry["name"])
        if name in out:
            # Silently keeping one of two same-named entries would make runs
            # depend on file order, which is exactly the class of bug this
            # module exists to remove.
            raise ValueError(f"{resolved}: duplicate location name {name!r}")
        out[name] = Location(
            name=name,
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            stratum=entry.get("stratum"),
            note=entry.get("note"),
        )
    return out


def resolve(name: str, path: str | Path | None = None) -> Location:
    """Look up one location by name.

    Args:
        name: The location name.
        path: Registry path, or None for the repo default.

    Returns:
        The matching :class:`Location`.

    Raises:
        KeyError: If the name is not in the registry. The message lists what is
            available, because a typo and a missing entry look identical from
            the command line otherwise.
    """
    known = load_locations(path)
    if name not in known:
        raise KeyError(
            f"unknown location {name!r}. Registry has: "
            f"{', '.join(sorted(known)) or '(empty)'}"
        )
    return known[name]
