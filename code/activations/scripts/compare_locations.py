#!/usr/bin/env python
"""Encode several labelled locations and compare them.

This is the intended first look: pick a handful of urban and rural points, encode
each, and read off whether the model separates them at all before investing in
anything more elaborate. If urban and rural chips are not distinguishable in the
pooled vectors, no amount of direction-finding downstream will help.

Locations come from a YAML file::

    - name: dc_downtown
      lat: 38.90324
      lon: -77.036964
      stratum: urban
    - name: shenandoah
      lat: 38.5100
      lon: -78.4400
      stratum: rural

Every point must fall inside a scene present in --scene-dir, which defaults to
the repo's ``data/scenes``. Points that fall outside are reported and skipped
rather than aborting the run, so one bad coordinate does not waste the whole
batch.

Example:
    python scripts/compare_locations.py --locations configs/locations.example.yaml \
        --config configs/default.yaml --out artifacts/compare
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml

from olmoearth_activations import analysis, viz
from olmoearth_activations.cli import _load_config, _setup_logging, write_run_manifest
from olmoearth_activations.encode import Encoder
from olmoearth_activations.loader import OlmoEarthModel
from olmoearth_activations.tiles import (
    SafeSceneSource,
    default_scene_dir,
    normalize_chip,
)

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@app.command()
def main(
    locations: Path = typer.Option(..., help="YAML list of name/lat/lon/stratum."),
    scene_dir: Path = typer.Option(
        default_scene_dir(), help="Directory with *.SAFE scenes."
    ),
    config: Optional[Path] = typer.Option(None, help="YAML run config."),
    tap: str = typer.Option("-1", help="Tap label or index for the comparison."),
    out: Path = typer.Option(Path("artifacts/compare"), help="Output directory."),
    allow_path_fallback: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode every location and write a similarity matrix plus per-site outputs."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    with locations.open() as fh:
        sites = yaml.safe_load(fh) or []
    if not isinstance(sites, list) or not sites:
        raise typer.BadParameter(f"{locations}: expected a non-empty YAML list")

    model = OlmoEarthModel.load(cfg.model, allow_path_fallback=allow_path_fallback)
    source = SafeSceneSource(scene_dir=scene_dir)
    encoder = Encoder(model, cfg.tile, cfg.tap)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    labels: list[str] = []
    strata: list[str] = []
    skipped: list[dict[str, object]] = []

    for site in sites:
        name = str(site.get("name") or f"{site['lat']:.4f},{site['lon']:.4f}")
        try:
            raw = source.read(site["lat"], site["lon"], cfg.tile.chip_px)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("skipping %s: %s", name, exc)
            skipped.append({"name": name, "reason": str(exc)})
            continue

        chip = normalize_chip(raw, cfg.tile.normalize)
        result = encoder.encode(
            chip,
            latlon=(site["lat"], site["lon"]),
            extra_meta={
                "name": name,
                "stratum": site.get("stratum"),
                "config_fingerprint": cfg.fingerprint(),
            },
        )
        result.save(out / f"{name}.npz", dtype=cfg.save_dtype)
        results.append(result)
        labels.append(name)
        strata.append(str(site.get("stratum") or "unknown"))
        logger.info("encoded %s (%s)", name, site.get("stratum"))

    if len(results) < 2:
        typer.echo(
            f"only {len(results)} location(s) encoded successfully; need at "
            f"least two to compare.",
            err=True,
        )
        for entry in skipped:
            typer.echo(f"  skipped {entry['name']}: {entry['reason']}", err=True)
        raise typer.Exit(code=1)

    tap_sel: str | int = tap if not _is_int(tap) else int(tap)
    similarity = analysis.compare_locations(results, tap_sel, labels=labels)
    similarity.to_csv(out / "similarity.csv")
    viz.save(viz.plot_location_similarity(similarity), str(out / "similarity.png"))

    # Per-site pooled vectors, tagged with stratum, so downstream code can fit
    # anything it likes without re-running the model.
    pooled = pd.concat(
        [analysis.pool_all(r, tap_sel) for r in results], ignore_index=True
    )
    pooled.insert(0, "stratum", strata)
    pooled.insert(0, "name", labels)
    pooled.to_csv(out / "pooled_vectors.csv", index=False)

    # Within- versus between-stratum similarity is the actual question here.
    summary = _stratum_summary(similarity, strata)
    summary.to_csv(out / "stratum_summary.csv", index=False)

    write_run_manifest(
        out, cfg, model, {"locations": sites, "skipped": skipped, "tap": tap}
    )
    typer.echo(summary.to_string(index=False))
    typer.echo(f"\nwrote {out}/similarity.csv, pooled_vectors.csv, stratum_summary.csv")


def _stratum_summary(similarity: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """Mean cosine similarity within and between strata.

    Args:
        similarity: Square similarity matrix.
        strata: Stratum label per row, in matrix order.

    Returns:
        DataFrame with columns ``group_a``, ``group_b``, ``n_pairs``,
        ``mean_cosine``. Self-pairs on the diagonal are excluded, since a chip's
        similarity to itself is 1 by construction and would inflate the
        within-stratum figure.
    """
    values = similarity.to_numpy()
    rows: list[dict[str, object]] = []
    unique = sorted(set(strata))
    for i, group_a in enumerate(unique):
        for group_b in unique[i:]:
            pairs = [
                values[r, c]
                for r in range(len(strata))
                for c in range(len(strata))
                if r < c and {strata[r], strata[c]} == {group_a, group_b}
            ]
            if not pairs:
                continue
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_pairs": len(pairs),
                    "mean_cosine": float(sum(pairs) / len(pairs)),
                }
            )
    return pd.DataFrame(rows)


def _is_int(value: str) -> bool:
    """True if the string parses as an integer."""
    try:
        int(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    app()
