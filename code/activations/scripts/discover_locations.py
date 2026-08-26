#!/usr/bin/env python
"""Find real-world features in OpenStreetMap and add them to the registry.

Turns a tag into a list of named locations. Instead of hand-placing points on a
map and guessing what is there, ask OSM for every hospital, supermarket or field
in a region and get names plus coordinates back.

The same query serves two purposes, which is the point. The coordinates say
where to point the encoder; the records say what is actually on the ground
there. So the label set for "does any dimension respond to hospitals?" comes
from the same place as the site list, rather than being assembled separately and
hoped to line up.

Tags worth knowing
------------------
``amenity=hospital``      acute-care hospitals
``healthcare=clinic``     smaller health centres
``shop=supermarket``      grocery stores
``landuse=farmland``      cultivated fields
``leisure=park``          parks

Detectability varies enormously and the tag does not warn you. Farmland and
parks are land *cover* and are directly measurable from reflectance. Hospitals
are land *use* -- reachable only through building morphology, since OlmoEarth's
pretraining included OSM ``building``/``parking``/``helipad`` rasters but never
a functional label. Supermarkets are both functionally invisible and, at roughly
3,000-6,000 m², only 2-4 patches wide at 40 m.

Uses the standard library only -- no new dependency, no API key.

Examples:
    # Every hospital in Maryland, listed but not written
    python scripts/discover_locations.py --tag amenity=hospital \\
        --area Maryland --dry-run

    # Write the 15 nearest to a point into the registry
    python scripts/discover_locations.py --tag amenity=hospital \\
        --area Maryland --near 39.0,-77.1 --limit 15
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import typer

from olmoearth_activations import locations as loc
from olmoearth_activations import osm

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _slug(text: str, fallback: str) -> str:
    """Turn a place name into a registry-safe identifier.

    Args:
        text: The raw name.
        fallback: Used when the name has no usable characters.

    Returns:
        A lowercase underscore-separated slug, trimmed to a sane length.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:48] or fallback


def _coordinate(element: dict[str, Any]) -> tuple[float, float] | None:
    """Return (lat, lon) for a node or a way/relation centre, or None."""
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def _append_entries(path: Path, entries: list[dict[str, Any]], tag: str) -> None:
    """Append registry entries as text, preserving the file's comments.

    Rewriting the file through a YAML dumper would silently delete every
    explanatory comment in it, and those comments are the only record of why
    each hand-picked location was chosen. Appending text keeps them.

    Args:
        path: Registry file, created if absent.
        entries: Dicts with name/lat/lon/stratum/note.
        tag: The tag these came from, recorded in the section header.
    """
    lines = [
        "",
        f"# {'-' * 62}",
        f"# Discovered from OpenStreetMap, tag {tag}.",
        "# Coordinates are OSM's, not hand-placed. A way or relation is reduced",
        "# to its centre, so a large campus is represented by one point.",
        f"# {'-' * 62}",
        "",
    ]
    for entry in entries:
        lines.append(f"- name: {entry['name']}")
        lines.append(f"  lat: {entry['lat']:.6f}")
        lines.append(f"  lon: {entry['lon']:.6f}")
        lines.append(f"  stratum: {entry['stratum']}")
        lines.append(f"  note: {entry['note']}")
        lines.append("")
    with path.open("a") as fh:
        fh.write("\n".join(lines))


@app.command()
def main(
    tag: str = typer.Option(
        ..., help="OSM tag as key=value, e.g. amenity=hospital."
    ),
    area: Optional[str] = typer.Option(
        None, help="Administrative area name, e.g. Maryland."
    ),
    bbox: Optional[str] = typer.Option(
        None, help="Bounding box: minlat,minlon,maxlat,maxlon."
    ),
    near: Optional[str] = typer.Option(
        None,
        help=(
            "'lat,lon'. Sort results by distance from here and keep the "
            "closest --limit. Use it to stay inside a tile you already have."
        ),
    ),
    limit: int = typer.Option(20, help="Maximum entries to keep."),
    prefix: Optional[str] = typer.Option(
        None, help="Name prefix. Defaults to the tag's value, e.g. 'hospital'."
    ),
    stratum: Optional[str] = typer.Option(
        None, help="Stratum label for the entries. Defaults to the tag value."
    ),
    registry: Optional[Path] = typer.Option(
        None, help="Registry to append to. Defaults to configs/locations.yaml."
    ),
    timeout: int = typer.Option(180, help="Query timeout in seconds."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what was found without writing."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Query OSM for a tag and append the results as named locations."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    tag_value = tag.split("=", 1)[1] if "=" in tag else tag
    name_prefix = prefix if prefix is not None else _slug(tag_value, "osm")
    label = stratum if stratum is not None else tag_value
    target = loc.registry_path(registry)

    bbox_tuple = None
    if bbox:
        parts = [float(p.strip()) for p in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be minlat,minlon,maxlat,maxlon")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
    if "=" not in tag and not tag.strip():
        raise typer.BadParameter("--tag must not be empty")
    try:
        query = osm.build_query(
            [tag], bbox=bbox_tuple, area=area, geometry=False, timeout=timeout
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    logger.debug("query:\n%s", query)
    typer.echo(f"querying OSM for {tag} in {area or bbox}...")
    started = time.monotonic()
    try:
        elements = osm.run(query, timeout)
    except osm.OverpassError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"{len(elements)} raw element(s) in {time.monotonic() - started:.1f}s"
    )
    if not elements:
        typer.echo(
            "Nothing found. Check the tag spelling, and that --area names an "
            "area OSM knows (try --bbox instead)."
        )
        raise typer.Exit(code=1)

    anchor: tuple[float, float] | None = None
    if near:
        parts = [p.strip() for p in near.split(",")]
        if len(parts) != 2:
            raise typer.BadParameter("--near must be 'lat,lon'")
        anchor = (float(parts[0]), float(parts[1]))

    # Existing names are skipped rather than renamed: a registry where the same
    # place appears twice under different names would quietly double-count it in
    # any later correlation.
    try:
        existing = set(loc.load_locations(target))
    except FileNotFoundError:
        existing = set()

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in elements:
        point = _coordinate(element)
        if point is None:
            continue
        tags = element.get("tags", {}) or {}
        raw_name = tags.get("name") or tags.get("operator") or ""
        base = _slug(raw_name, f"{element.get('type', 'x')}{element.get('id', '')}")
        name = f"{name_prefix}_{base}"
        if name in existing or name in seen:
            continue
        seen.add(name)
        note_bits = [f"OSM {element.get('type')}/{element.get('id')}", tag]
        if raw_name:
            note_bits.insert(0, raw_name)
        candidates.append(
            {
                "name": name,
                "lat": point[0],
                "lon": point[1],
                "stratum": label,
                "note": " | ".join(note_bits),
                "_km": _km_between(anchor, point) if anchor else 0.0,
            }
        )

    if anchor:
        candidates.sort(key=lambda c: c["_km"])
    kept = candidates[:limit]

    typer.echo(f"\n{len(candidates)} usable, keeping {len(kept)}:")
    for entry in kept:
        distance = f"  {entry['_km']:6.1f} km" if anchor else ""
        typer.echo(
            f"  {entry['name']:44s} ({entry['lat']:9.5f}, "
            f"{entry['lon']:10.5f}){distance}"
        )
    if len(candidates) > len(kept):
        # Silent truncation would read as "this is everything there is".
        typer.echo(
            f"\n  ...{len(candidates) - len(kept)} more not kept (--limit "
            f"{limit})."
        )

    # The cost of this batch, stated before anything is written. Otherwise the
    # first hint that 15 locations means 15 GB arrives one gigabyte at a time.
    reusable = sum(
        1
        for entry in kept
        if loc.find_covering_scene(entry["lat"], entry["lon"]) is not None
    )
    needed = len(kept) - reusable
    typer.echo(
        f"\nimagery: {reusable} of {len(kept)} covered by scenes already on "
        f"disk (symlinked, free)"
    )
    if needed:
        typer.echo(
            f"         {needed} would need a download, roughly 1 GB each.\n"
            f"         Several may share a tile -- a tile is 110 km wide -- and "
            f"fetch_scene.py\n"
            f"         re-checks the disk each time, so fetching them in "
            f"sequence downloads\n"
            f"         each tile once. The real total is usually well under "
            f"{needed}."
        )

    if dry_run:
        typer.echo("\n--dry-run, registry unchanged.")
        return
    if not kept:
        typer.echo("\nNothing new to add.")
        return

    for entry in kept:
        entry.pop("_km", None)
    _append_entries(target, kept, tag)
    typer.echo(f"\nappended {len(kept)} location(s) to {target}")
    typer.echo(
        "\nNext, per location:\n"
        "  python scripts/fetch_scene.py --location <name>   # reuses a covering scene if there is one\n"
        "  python scripts/encode_location.py --location <name>"
    )


if __name__ == "__main__":
    app()
