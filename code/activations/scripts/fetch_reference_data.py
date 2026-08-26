#!/usr/bin/env python
"""Download every reference dataset the labelling step needs, once.

After this runs, ``fetch_labels.py --osm-source local`` never touches a web API:
land cover comes from WorldCover tiles on disk and landmarks from OpenStreetMap
regional extracts on disk. That removes the fair-use blocks entirely, and it
pins both sources to a dated snapshot -- Overpass answers from a live database
that changes under you, so a label built today is otherwise not reproducible
tomorrow.

Where things land, and why the names look like they do
-----------------------------------------------------
``data/osm/us-maryland-20260825.osm.pbf``
    Region, then the date it was downloaded. The region says what ground it
    covers; the date says which snapshot, which is what makes a result citable.

``data/worldcover/ESA_WorldCover_10m_2021_v200_N39W078_Map.tif``
    ESA's own filename, kept verbatim because it already states resolution
    (10 m), year, product version and tile. ``N39W078`` is the tile's south-west
    corner on a 3-degree grid, so N39W078 covers 39-42 N, 78-75 W.

A ``README.md`` is written into each directory recording the exact bounds of
every file and which registry locations fall inside it, so a year from now it is
obvious what is on disk and what it covers.

Examples:
    # Everything needed for the current location registry
    python scripts/fetch_reference_data.py --all

    # Just the OSM extracts, specific regions
    python scripts/fetch_reference_data.py \\
        --region north-america/us/maryland \\
        --region north-america/us/district-of-columbia
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from olmoearth_activations import locations as loc
from olmoearth_activations import osm_local, worldcover

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

GEOFABRIK = "https://download.geofabrik.de"
# Regions covering the current registry. Geofabrik paths, without the .osm.pbf
# suffix. State-level extracts are a good size trade-off: a US state is tens of
# megabytes, small enough to download in a minute and large enough that one file
# usually covers every location in a metro area.
DEFAULT_REGIONS = (
    "north-america/us/maryland",
    "north-america/us/district-of-columbia",
    "north-america/us/virginia",
)


def _download(url: str, dest: Path, label: str) -> bool:
    """Stream a URL to a file, reporting progress.

    Returns:
        True if downloaded, False if it already existed or failed.
    """
    if dest.exists():
        typer.echo(f"  {label}: already have {dest.name} "
                   f"({dest.stat().st_size / 1e6:.0f} MB)")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    typer.echo(f"  {label}")
    typer.echo(f"    from {url}")
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "olmoearth-activations/0.1 (research)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            if total:
                typer.echo(f"    size {total / 1e6:.0f} MB")
            done = 0
            last_draw = 0.0
            with partial.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    # Redraw at most twice a second: a progress bar that writes
                    # per megabyte floods the terminal on a 400 MB file.
                    if now - last_draw < 0.5 and done < (total or 1):
                        continue
                    last_draw = now
                    typer.echo("\r" + _progress_line(done, total, now - started),
                               nl=False)
        typer.echo("")
        # Rename only on success, so an interrupted download cannot be mistaken
        # for a complete one on the next run.
        partial.replace(dest)
        elapsed = time.monotonic() - started
        typer.echo(
            f"    done: {dest.name} "
            f"({dest.stat().st_size / 1e6:.0f} MB in {elapsed:.0f}s)"
        )
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        typer.echo("")
        typer.echo(f"    FAILED {exc}")
        return False


def _progress_line(done: int, total: int, elapsed: float) -> str:
    """Render a one-line progress bar with rate and ETA.

    Written by hand rather than with tqdm: this is the only place in the package
    that would need it, and a dependency for one progress bar is a poor trade.
    """
    rate = done / elapsed if elapsed > 0 else 0.0
    speed = f"{rate / 1e6:5.1f} MB/s"
    if not total:
        return f"    {done / 1e6:6.0f} MB  {speed}"
    fraction = min(done / total, 1.0)
    width = 28
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    remaining = (total - done) / rate if rate > 0 else 0
    eta = f"ETA {remaining / 60:4.1f} min" if remaining > 90 else f"ETA {remaining:3.0f}s"
    return (
        f"    [{bar}] {fraction * 100:3.0f}%  "
        f"{done / 1e6:5.0f}/{total / 1e6:.0f} MB  {speed}  {eta}"
    )


def _write_osm_readme(directory: Path) -> None:
    """Record what each extract covers, and which locations fall inside."""
    sites = loc.load_locations()
    lines = [
        "# OpenStreetMap regional extracts",
        "",
        "Downloaded by `scripts/fetch_reference_data.py`. Used by",
        "`fetch_labels.py --osm-source local`, which picks the smallest extract",
        "whose bounds contain the chip -- so a city extract wins over a state one.",
        "",
        "Filenames are `<region>-<download date>.osm.pbf`. The date matters: it is",
        "the snapshot the labels were built from.",
        "",
    ]
    for path in sorted(directory.glob("*.osm.pbf")):
        bounds = osm_local.extract_bounds(path)
        lines.append(f"## `{path.name}`")
        lines.append("")
        lines.append(f"- size: {path.stat().st_size / 1e6:.0f} MB")
        if bounds is None:
            lines.append("- bounds: unknown (install pyogrio to read them)")
        else:
            x0, y0, x1, y1 = bounds
            lines.append(
                f"- bounds: lon {x0:.3f} to {x1:.3f}, lat {y0:.3f} to {y1:.3f}"
            )
            inside = [
                n for n, s in sites.items() if x0 <= s.lon <= x1 and y0 <= s.lat <= y1
            ]
            lines.append(
                f"- covers {len(inside)} registry location(s): "
                f"{', '.join(inside) or 'none'}"
            )
        lines.append("")
    (directory / "README.md").write_text("\n".join(lines))


def _write_worldcover_readme(directory: Path) -> None:
    """Record which tile covers which locations."""
    sites = loc.load_locations()
    lines = [
        "# ESA WorldCover 10 m land cover",
        "",
        "Downloaded by `scripts/fetch_reference_data.py`. Used by",
        "`fetch_labels.py --cache-dir data/worldcover`.",
        "",
        "Tiles are 3x3 degrees, named by south-west corner: `N39W078` covers",
        "39-42 N and 78-75 W. Class codes: 10 tree, 20 shrub, 30 grass,",
        "40 cropland, 50 built-up, 60 bare, 80 water, 90 wetland.",
        "",
    ]
    for path in sorted(directory.glob("*.tif")):
        tile = path.stem.split("_")[-2] if "_" in path.stem else "?"
        inside = [n for n, s in sites.items() if worldcover.tile_name(s.lat, s.lon) == tile]
        lines.append(f"## `{path.name}`")
        lines.append("")
        lines.append(f"- tile: {tile}")
        lines.append(f"- size: {path.stat().st_size / 1e6:.0f} MB")
        lines.append(
            f"- covers {len(inside)} registry location(s): "
            f"{', '.join(inside) or 'none'}"
        )
        lines.append("")
    (directory / "README.md").write_text("\n".join(lines))


@app.command()
def main(
    all_data: bool = typer.Option(
        False, "--all", help="Fetch both OSM extracts and WorldCover tiles."
    ),
    osm: bool = typer.Option(False, "--osm", help="Fetch OSM extracts only."),
    landcover: bool = typer.Option(
        False, "--landcover", help="Fetch WorldCover tiles only."
    ),
    region: Optional[list[str]] = typer.Option(
        None,
        help=(
            "Geofabrik region path, repeatable, e.g. north-america/us/maryland. "
            "Defaults to the regions covering the current registry."
        ),
    ),
    wc_version: str = typer.Option(worldcover.DEFAULT_VERSION),
    wc_year: str = typer.Option(worldcover.DEFAULT_YEAR),
    registry: Optional[Path] = typer.Option(None, help="Alternative locations YAML."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download OSM extracts and WorldCover tiles into the data directory."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    if not (all_data or osm or landcover):
        raise typer.BadParameter("pass --all, or --osm and/or --landcover")

    root = loc.repo_root() or Path.cwd()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    sites = loc.load_locations(registry)

    if all_data or landcover:
        directory = root / "data" / "worldcover"
        # One tile per location, de-duplicated -- seven locations across two
        # tiles means two downloads, not seven.
        tiles = sorted({worldcover.tile_name(s.lat, s.lon) for s in sites.values()})
        typer.echo(
            f"\nWorldCover: {len(tiles)} tile(s) for {len(sites)} location(s) "
            f"-> {directory}"
        )
        for tile in tiles:
            covers = [
                n for n, s in sites.items()
                if worldcover.tile_name(s.lat, s.lon) == tile
            ]
            dest = worldcover.local_path(directory, tile, wc_version, wc_year)
            _download(
                worldcover.tile_url(tile, wc_version, wc_year),
                dest,
                f"{tile} ({', '.join(covers)})",
            )
        _write_worldcover_readme(directory)
        typer.echo(f"  wrote {directory / 'README.md'}")

    if all_data or osm:
        directory = root / "data" / "osm"
        regions = list(region) if region else list(DEFAULT_REGIONS)
        typer.echo(f"\nOSM extracts: {len(regions)} region(s) -> {directory}")
        # State extracts run 20-400 MB each. Saying so before starting means a
        # long quiet stretch reads as expected rather than as a hang.
        typer.echo(
            "  State extracts are large (20-400 MB each). Expect several "
            "minutes on a normal connection;\n  a progress bar with ETA is "
            "shown per file. Each is downloaded once and reused."
        )
        for path in regions:
            slug = path.replace("north-america/us/", "us-").replace("/", "-")
            dest = directory / f"{slug}-{stamp}.osm.pbf"
            # An older snapshot of the same region is still usable; skip rather
            # than accumulate near-duplicates of a 90 MB file.
            existing = sorted(directory.glob(f"{slug}-*.osm.pbf"))
            if existing and not dest.exists():
                typer.echo(
                    f"  {slug}: already have {existing[-1].name}; delete it to "
                    f"refresh"
                )
                continue
            _download(f"{GEOFABRIK}/{path}-latest.osm.pbf", dest, slug)
        _write_osm_readme(directory)
        typer.echo(f"  wrote {directory / 'README.md'}")

    typer.echo(
        "\nDone. Label everything without touching a web API:\n"
        "  python scripts/fetch_labels.py --all --config configs/default.yaml \\\n"
        "      --osm-source local --cache-dir ../../data/worldcover"
    )
    if shutil.which("python") and not _pyogrio_present():
        typer.echo(
            "\nNOTE: reading .osm.pbf needs pyogrio, which is not installed:\n"
            "  uv pip install pyogrio"
        )


def _pyogrio_present() -> bool:
    """True if pyogrio can be imported."""
    try:
        import pyogrio  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    app()
