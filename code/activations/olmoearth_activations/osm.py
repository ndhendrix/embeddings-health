"""Querying OpenStreetMap through Overpass.

Two consumers, two needs. Choosing *where* to encode wants one coordinate per
feature (``out center``). Labelling *which patches* contain a feature wants its
outline (``out geom``) -- a hospital campus covers many 40 m patches and a
centre point marks one of them, so centre points would label the other 95% of
real hospital patches as not-hospital and bury the very correlation being looked
for.

Standard library only: no API key, no new dependency.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Higher-capacity community mirrors. The main instance enforces fair use by
# refusing connections outright (Errno 61) once a burst of requests trips it,
# not just by returning 429 -- so having somewhere else to go matters for any
# run that issues dozens of queries.
MIRRORS = {
    "main": OVERPASS_URL,
    "kumi": "https://overpass.kumi.systems/api/interpreter",
    "coffee": "https://overpass.private.coffee/api/interpreter",
}

# Status codes worth retrying. 500 is included because Overpass uses it for
# transient internal failures as well as for genuine query problems, and the two
# are only distinguishable from the response body -- so retry, then report what
# the body actually said.
RETRY_CODES = frozenset({429, 500, 502, 503, 504})
# Overpass operators ask for a descriptive agent so heavy users are identifiable.
USER_AGENT = "olmoearth-activations/0.1 (research; contact via repo)"


class OverpassError(RuntimeError):
    """Overpass could not be reached, or refused the query."""


def selector(spec: str) -> str:
    """Turn a tag spec into an Overpass selector.

    Args:
        spec: Either ``key=value`` for an exact match, or a bare ``key`` to
            match any feature carrying that key -- ``highway`` matches every
            road class, which is what a walkability label wants.

    Returns:
        A bracketed Overpass selector.
    """
    if "=" in spec:
        key, value = spec.split("=", 1)
        return f'["{key}"="{value}"]'
    return f'["{spec}"]'


def build_query(
    specs: list[str],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    area: str | None = None,
    geometry: bool = False,
    timeout: int = 180,
    kinds: tuple[str, ...] = ("node", "way", "relation"),
) -> str:
    """Build an Overpass QL query.

    Nodes, ways and relations are all requested because the same feature type is
    mapped inconsistently -- a park may be a single way or a multi-polygon
    relation, and skipping either loses real features silently.

    Args:
        specs: Tag specs, combined as a union.
        bbox: ``(latmin, lonmin, latmax, lonmax)``.
        area: Administrative area name, used instead of bbox.
        geometry: Request full outlines rather than centre points.
        timeout: Server-side timeout in seconds.
        kinds: Which element kinds to request.

    Returns:
        The query string.

    Raises:
        ValueError: If neither bbox nor area is given.
    """
    if bbox is not None:
        scope = ""
        where = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
    elif area is not None:
        # admin_level is left unconstrained: US states are 4 and counties 6, so
        # pinning it would break one of the two.
        scope = f'area["name"="{area}"]->.searchArea;'
        where = "(area.searchArea)"
    else:
        raise ValueError("pass bbox or area")

    body = "\n".join(
        f"  {kind}{selector(spec)}{where};" for spec in specs for kind in kinds
    )
    out = "out geom tags;" if geometry else "out center tags;"
    return f"[out:json][timeout:{timeout}];\n{scope}\n(\n{body}\n);\n{out}"


def run(
    query: str,
    timeout: int = 180,
    *,
    retries: int = 4,
    backoff: float = 5.0,
    url: str | None = None,
) -> list[dict[str, Any]]:
    """Post a query to Overpass and return its elements, retrying when throttled.

    429 (too many requests) and 504 (gateway timeout) are how the public
    instance says "not right now" -- they carry no information about the query
    and are the expected response to several queries in quick succession. They
    are retried with a growing wait rather than surfaced, because failing a
    label group mid-run leaves a partial label set that looks like real absence:
    a missing ``water`` group and a genuinely water-free chip are
    indistinguishable downstream.

    Args:
        query: Overpass QL.
        timeout: Client-side timeout in seconds.
        retries: Attempts after the first before giving up.
        backoff: Base seconds to wait, doubled each attempt.

    Returns:
        The ``elements`` list.

    Raises:
        OverpassError: On non-retryable failure, or once retries run out.
    """
    data = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(
        url or OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT}
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            return list(payload.get("elements", []))
        except urllib.error.HTTPError as exc:
            last = exc
            # Overpass explains itself in the response body -- "runtime error:
            # Query run out of memory", "parse error", a rate-limit notice. It
            # is the only thing that distinguishes a query bug from a busy
            # server, so surface it instead of guessing from the status code.
            detail = ""
            try:
                body = exc.read().decode("utf-8", "replace").strip()
                if body:
                    collapsed = " ".join(body.split())
                    detail = f" -- {collapsed[:400]}"
            except Exception:  # noqa: BLE001 - diagnostics must not mask the error
                pass
            if exc.code not in RETRY_CODES or attempt == retries:
                raise OverpassError(
                    f"Overpass returned HTTP {exc.code} after "
                    f"{attempt + 1} attempt(s){detail}"
                ) from exc
            wait = backoff * (2**attempt)
            logger.warning(
                "Overpass HTTP %d%s, retrying in %.0fs (attempt %d of %d)",
                exc.code,
                detail,
                wait,
                attempt + 1,
                retries,
            )
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt == retries:
                raise OverpassError(f"could not reach Overpass: {exc}") from exc
            wait = backoff * (2**attempt)
            logger.warning("Overpass unreachable (%s), retrying in %.0fs", exc, wait)
            time.sleep(wait)
        except json.JSONDecodeError as exc:
            raise OverpassError(
                "Overpass returned something that is not JSON"
            ) from exc
    raise OverpassError(f"Overpass failed after {retries + 1} attempts: {last}")


def rings(element: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Extract closed lat/lon rings from an ``out geom`` element.

    Ways carry their own ``geometry``; relations carry one per member. Inner
    rings are *not* distinguished from outer ones, so a park with a hole in it is
    treated as solid. That over-claims coverage slightly and is the right
    trade-off here -- holes are rare at 40 m, and the alternative is a full
    multipolygon assembler.

    Args:
        element: One Overpass element.

    Returns:
        A list of rings, each a list of ``(lat, lon)``. Empty for nodes, which
        have no area.
    """
    out: list[list[tuple[float, float]]] = []
    geometry = element.get("geometry")
    if isinstance(geometry, list) and len(geometry) >= 3:
        out.append([(float(p["lat"]), float(p["lon"])) for p in geometry if "lat" in p])
    for member in element.get("members", []) or []:
        member_geom = member.get("geometry")
        if isinstance(member_geom, list) and len(member_geom) >= 3:
            out.append(
                [(float(p["lat"]), float(p["lon"])) for p in member_geom if "lat" in p]
            )
    return [ring for ring in out if len(ring) >= 3]


def lines(element: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Extract open lat/lon paths from an ``out geom`` element.

    Roads and paths are lines, not areas. Returned separately from :func:`rings`
    so a caller can give them a width instead of trying to fill them.

    Args:
        element: One Overpass element.

    Returns:
        A list of paths, each a list of ``(lat, lon)``.
    """
    geometry = element.get("geometry")
    if isinstance(geometry, list) and len(geometry) >= 2:
        return [[(float(p["lat"]), float(p["lon"])) for p in geometry if "lat" in p]]
    return []
