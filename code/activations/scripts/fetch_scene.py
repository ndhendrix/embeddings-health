#!/usr/bin/env python
"""Download the Sentinel-2 L2A scene for a named location, via EODAG.

Replaces the browser round-trip: search Copernicus for a product covering one
location, pick one, and unpack it into that location's scene directory so
encode_location.py can find it with no further argument.

What it will and will not do
----------------------------
It downloads **L2A** only. SafeSceneSource globs
``*.SAFE/GRANULE/*/IMG_DATA/*/*_B02_*.jp2``, and that resolution subdirectory
exists only in L2A -- L1C stores bands directly under IMG_DATA and silently
matches nothing.

It refuses to add a second scene to a directory that already has one. Two
scenes in one directory do not extend coverage; the alphabetically first one
shadows the other, which is how a T18SUH product beat the T18SUJ one that
actually covered downtown DC.

Credentials
-----------
EODAG reads Copernicus credentials from the environment. Set these once, from
the same free account used for the browser -- this script never prompts for or
stores them::

    export EODAG__COP_DATASPACE__AUTH__CREDENTIALS__USERNAME='you@example.org'
    export EODAG__COP_DATASPACE__AUTH__CREDENTIALS__PASSWORD='...'

Install
-------
EODAG is an optional dependency::

    uv pip install 'eodag>=3.0'

Example:
    python scripts/fetch_scene.py --location dc --start 2022-06-01 --end 2022-06-30
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import typer

from olmoearth_activations import locations as loc

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

# EODAG 4.x calls this "collection"; 2.x/3.x called it "productType". Passing the
# old name does not warn -- eodag reports "Field required: collection" and then
# tries every provider in turn, each failing identically, which reads like a
# credential or coverage problem rather than a wrong kwarg. Verified against
# eodag/types/search_args.py and eodag/resources/providers/cop_dataspace.yml,
# where S2_MSI_L2A maps to product:type S2MSI2A -- the .SAFE product with the
# .jp2 bands SafeSceneSource needs. Do not switch to S2_MSI_L2A_COG or _JP2:
# those are other providers' repackagings with a different layout on disk.
COLLECTION = "S2_MSI_L2A"
PROVIDER = "cop_dataspace"
CRED_ENV = (
    "EODAG__COP_DATASPACE__AUTH__CREDENTIALS__USERNAME",
    "EODAG__COP_DATASPACE__AUTH__CREDENTIALS__PASSWORD",
)


def _require_credentials() -> None:
    """Fail before a long search if credentials are absent.

    Raises:
        typer.BadParameter: If either credential variable is unset.
    """
    missing = [name for name in CRED_ENV if not os.environ.get(name)]
    if missing:
        raise typer.BadParameter(
            "Copernicus credentials not found in the environment. Set:\n  "
            + "\n  ".join(f"export {name}='...'" for name in missing)
            + "\nUse the same free account as the Copernicus browser. If you "
            "have configured EODAG through ~/.config/eodag/eodag.yml instead, "
            "pass --skip-credential-check."
        )


def _bbox(lat: float, lon: float, pad_deg: float) -> dict[str, float]:
    """Return a small search box around a point.

    A point geometry can sit exactly on a tile boundary and match neighbours
    ambiguously; a small box makes the intersection explicit. The box only
    selects candidate products -- the chip is still read at the exact
    coordinate.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        pad_deg: Half-width of the box, in degrees.

    Returns:
        An EODAG ``geom`` mapping.
    """
    return {
        "lonmin": lon - pad_deg,
        "latmin": lat - pad_deg,
        "lonmax": lon + pad_deg,
        "latmax": lat + pad_deg,
    }


def _search_results(dag: Any, **kwargs: Any) -> list[Any]:
    """Run a search and normalise the return shape across EODAG versions.

    EODAG 2.x returned ``(SearchResult, total_count)``; 3.x returns the
    ``SearchResult`` alone. Unpacking one shape as the other fails confusingly,
    so detect it rather than pinning a version.

    Args:
        dag: An ``EODataAccessGateway``.
        **kwargs: Passed through to ``dag.search``.

    Returns:
        A list of products.
    """
    found = dag.search(**kwargs)
    if isinstance(found, tuple):
        found = found[0]
    return list(found)


def _cloud_cover(product: Any) -> float | None:
    """Return a product's cloud cover percentage, or None if not reported."""
    props = getattr(product, "properties", {}) or {}
    for key in ("cloudCover", "cloud_cover", "eo:cloud_cover"):
        value = props.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _sensing_date(product: Any) -> str:
    """Return a product's sensing date as ``YYYY-MM-DD``, or '?' if unknown.

    Several key names are tried because eodag moved to STAC property names in
    4.x while keeping the older ones for some providers. This is not cosmetic:
    ``--target-date`` sorts on this value, so a '?' here silently disables that
    option rather than failing.

    The final fallback parses the product title, which for Sentinel-2 always
    embeds the acquisition stamp (``S2A_MSIL2A_20220619T154821_...``). That is
    the most reliable source of all -- it is in the product name by
    specification.
    """
    props = getattr(product, "properties", {}) or {}
    for key in (
        "startTimeFromAscendingNode",
        "start_datetime",
        "datetime",
        "completionTimeFromAscendingNode",
    ):
        value = props.get(key)
        if value and str(value)[:4].isdigit():
            return str(value)[:10]

    title = str(props.get("title") or props.get("id") or product)
    match = re.search(r"_(\d{4})(\d{2})(\d{2})T\d{6}", title)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "?"


# EODAG takes the destination as ``output_dir``, verified against 4.7.1
# (eodag/types/download_args.py). Do not try to discover this by inspecting
# ``download``'s signature -- it accepts everything through **kwargs, so
# introspection reports the parameter as absent and silently picks the wrong
# name. Pre-3.0 called it ``outputs_prefix``; the pyproject floor rules that out.
DOWNLOAD_DIR_KWARG = "output_dir"


@app.command()
def main(
    location: Optional[str] = typer.Option(
        None, help="Name from configs/locations.yaml. Sets both point and destination."
    ),
    lat: Optional[float] = typer.Option(None, help="Latitude, if not using --location."),
    lon: Optional[float] = typer.Option(None, help="Longitude, if not using --location."),
    dest: Optional[Path] = typer.Option(
        None, help="Override the destination directory."
    ),
    start: str = typer.Option("2022-06-01", help="Earliest sensing date, YYYY-MM-DD."),
    end: str = typer.Option("2022-06-30", help="Latest sensing date, YYYY-MM-DD."),
    max_cloud: float = typer.Option(10.0, help="Maximum cloud cover percent."),
    target_date: Optional[str] = typer.Option(
        None,
        help=(
            "Prefer the scene nearest this date rather than the least cloudy. "
            "Match your config's tile.date when comparing runs."
        ),
    ),
    pad_deg: float = typer.Option(0.02, help="Half-width of the search box, degrees."),
    no_reuse: bool = typer.Option(
        False,
        "--no-reuse",
        help=(
            "Download even if a scene already on disk covers this point. "
            "Reuse is the default because a tile is 110 km wide and usually "
            "covers many locations."
        ),
    ),
    edge_margin_deg: float = typer.Option(
        0.03,
        help=(
            "How far inside a scene's footprint a point must sit to reuse it. "
            "0.03 deg is about 3 km, enough for chips up to 512 px."
        ),
    ),
    registry: Optional[Path] = typer.Option(None, help="Alternative locations YAML."),
    status: bool = typer.Option(
        False,
        "--status",
        help="Print imagery coverage for every registry location and exit.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Search and list candidates without downloading."
    ),
    skip_credential_check: bool = typer.Option(
        False, help="Skip the env-var check, e.g. when configured via eodag.yml."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search for and download one L2A scene covering a location."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if status:
        rows = loc.coverage(registry, edge_margin_deg)
        counts = {"own": 0, "reuse": 0, "missing": 0}
        typer.echo(f"{'location':26s} {'imagery':9s} scene")
        typer.echo("-" * 72)
        for site, state, scene in rows:
            counts[state] += 1
            where = ""
            if state == "own":
                where = "in its own directory"
            elif state == "reuse" and scene is not None:
                where = f"covered by {scene.parent.name}"
            else:
                where = "needs download"
            typer.echo(f"{site.name:26s} {state:9s} {where}")
        typer.echo("-" * 72)
        typer.echo(
            f"{counts['own']} downloaded, {counts['reuse']} reusable for free, "
            f"{counts['missing']} still to fetch"
        )
        if counts["missing"]:
            typer.echo(
                f"\n{counts['missing']} location(s) need imagery, about 1 GB "
                f"each before tile sharing:"
            )
            for site, state, _ in rows:
                if state == "missing":
                    typer.echo(
                        f"  python scripts/fetch_scene.py --location {site.name}"
                    )
        return

    if location is not None:
        site = loc.resolve(location, registry)
        # Explicit coordinates win over the registry, so a one-off probe near a
        # named site does not need a registry edit.
        point_lat = lat if lat is not None else site.lat
        point_lon = lon if lon is not None else site.lon
        destination = dest or loc.scene_dir_for(site.name)
        label = site.name
    else:
        if lat is None or lon is None:
            raise typer.BadParameter("pass --location, or both --lat and --lon")
        if dest is None:
            raise typer.BadParameter("--dest is required when not using --location")
        point_lat, point_lon, destination, label = lat, lon, dest, "ad-hoc"

    existing = sorted(destination.glob("*.SAFE")) if destination.exists() else []
    if existing and not dry_run:
        raise typer.BadParameter(
            f"{destination} already holds {len(existing)} scene(s): "
            f"{', '.join(p.name for p in existing)}.\nA second scene would not "
            f"add coverage -- the alphabetically first one wins the band glob. "
            f"Remove or move the existing scene first."
        )

    # Reuse before download: a scene already on disk that covers this point is
    # the same gigabyte, so link it instead of fetching it again.
    if not no_reuse:
        shared = loc.find_covering_scene(
            point_lat, point_lon, edge_margin_deg, destination
        )
        if shared is not None:
            if dry_run:
                typer.echo(
                    f"would reuse {shared.parent.name}/{shared.name} "
                    f"(covers this point); --dry-run, nothing linked"
                )
                return
            destination.mkdir(parents=True, exist_ok=True)
            link = destination / shared.name
            # Relative target, so moving or copying data/scenes keeps it valid.
            link.symlink_to(Path("..") / shared.parent.name / shared.name)
            typer.echo(
                f"reusing {shared.parent.name}/{shared.name}\n"
                f"  symlinked into {destination} -- no download needed\n"
                f"  (pass --no-reuse to force a separate copy)"
            )
            typer.echo(
                f"\nReady:\n  python scripts/encode_location.py "
                f"--location {label}"
            )
            return

    # Validate dates before searching. An impossible date like 2022-06-31 is
    # accepted by the provider and simply matches nothing, which is
    # indistinguishable from genuinely cloudy months.
    for flag, value in (("--start", start), ("--end", end)):
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise typer.BadParameter(f"{flag}={value!r} is not a real date: {exc}")
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise typer.BadParameter(f"--start {start} is after --end {end}")

    if not skip_credential_check:
        _require_credentials()

    try:
        from eodag import EODataAccessGateway
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise typer.BadParameter(
            "eodag is not installed. Install it with:\n"
            "  uv pip install 'eodag>=3.0'"
        ) from exc

    dag = EODataAccessGateway()
    try:
        dag.set_preferred_provider(PROVIDER)
    except Exception as exc:  # noqa: BLE001 - provider naming varies by version
        logger.warning(
            "could not set preferred provider %r (%s); using EODAG's default",
            PROVIDER,
            exc,
        )

    typer.echo(
        f"searching {COLLECTION} for {label} at ({point_lat:.5f}, "
        f"{point_lon:.5f}), {start}..{end}, cloud <= {max_cloud}%"
    )
    # The cloud filter is keyed "eo:cloud_cover", a STAC extension name, not
    # "cloudCover". cop_dataspace.yml maps it to a DoubleAttribute `le`
    # comparison -- a real maximum. Passing the wrong name is not rejected: the
    # unrecognised kwarg falls through to a generic StringAttribute *equality*,
    # so the query asks for scenes whose cloud cover is exactly "10.0" and
    # matches essentially nothing, reported as "no products matched".
    search_kwargs: dict[str, Any] = {
        "collection": COLLECTION,
        "geom": _bbox(point_lat, point_lon, pad_deg),
        "start": start,
        "end": end,
        "provider": PROVIDER,
        "eo:cloud_cover": max_cloud,
        # Three months of Sentinel-2 over one tile is roughly 35 products; the
        # default page of 20 would silently truncate the candidate list.
        "items_per_page": 100,
    }
    products = _search_results(dag, **search_kwargs)

    # Filter again locally. The server-side comparison is provider-specific and
    # has already been wrong once here; re-checking against the product's own
    # reported value costs nothing and cannot be fooled by a mapping change.
    before = len(products)
    products = [
        product
        for product in products
        if (_cloud_cover(product) or 0.0) <= max_cloud
    ]
    if before != len(products):
        logger.info(
            "server returned %d product(s); %d within %.0f%% cloud",
            before,
            len(products),
            max_cloud,
        )
    if not products:
        raise typer.BadParameter(
            "no products matched. Widen --start/--end, raise --max-cloud, or "
            "check that the coordinate is right."
        )

    # Least cloud by default; nearest-to-date when asked. Cloud is the usual
    # thing that ruins a chip, but matching the config's date matters more when
    # a run is meant to be comparable to an earlier one.
    if target_date:
        want = datetime.strptime(target_date, "%Y-%m-%d").date()

        def sort_key(product: Any) -> tuple[float, float]:
            stamp = _sensing_date(product)
            try:
                delta = abs((date.fromisoformat(stamp) - want).days)
            except ValueError:
                delta = float("inf")
            return (float(delta), _cloud_cover(product) or 100.0)

    else:

        def sort_key(product: Any) -> tuple[float, float]:
            return (_cloud_cover(product) or 100.0, 0.0)

    products.sort(key=sort_key)

    typer.echo(f"\n{len(products)} candidate(s), best first:")
    for product in products[:10]:
        cloud = _cloud_cover(product)
        cloud_text = f"{cloud:5.1f}%" if cloud is not None else "    ?"
        title = getattr(product, "properties", {}).get("title", str(product))
        typer.echo(f"  {_sensing_date(product)}  cloud={cloud_text}  {title}")

    if dry_run:
        typer.echo("\n--dry-run, nothing downloaded.")
        return

    chosen = products[0]
    destination.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"\ndownloading into {destination} (roughly 1 GB, this takes a while)"
    )
    path = dag.download(
        chosen, extract=True, **{DOWNLOAD_DIR_KWARG: str(destination)}
    )
    typer.echo(f"download reported: {path}")

    unpacked = sorted(destination.rglob("*.SAFE"))
    if not unpacked:
        typer.echo(
            "\nWARNING: no *.SAFE directory found under the destination. If a "
            ".zip was left instead, unzip it in place -- the band glob needs the "
            "expanded directory, not the archive."
        )
        return

    typer.echo(f"\nscene: {unpacked[0].name}")
    typer.echo("verifying the 12 bands the loader needs...")
    bands = ["B01", "B02", "B03", "B04", "B05", "B06",
             "B07", "B08", "B8A", "B09", "B11", "B12"]
    missing = [
        band
        for band in bands
        if not list(destination.glob(f"*.SAFE/GRANULE/*/IMG_DATA/*/*_{band}_*.jp2"))
    ]
    if missing:
        typer.echo(
            f"  MISSING: {', '.join(missing)}\n"
            f"  This is what an L1C product looks like -- its bands are not in "
            f"resolution subdirectories. Re-download as L2A."
        )
        raise typer.Exit(code=1)

    typer.echo("  all 12 bands present")
    hint = f"--location {label}" if location else f"--lat {point_lat} --lon {point_lon}"
    typer.echo(f"\nReady:\n  python scripts/encode_location.py {hint}")


if __name__ == "__main__":
    app()
