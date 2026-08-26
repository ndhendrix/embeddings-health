"""Command-line entry points.

These are deliberately thin: they parse arguments, resolve a config, call into
the library, and write outputs plus a run manifest. Any logic worth testing
belongs in the modules, not here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from olmoearth_activations import __version__, analysis, locations, viz
from olmoearth_activations.config import RunConfig
from olmoearth_activations.encode import Encoder
from olmoearth_activations.loader import OlmoEarthModel
from olmoearth_activations.tiles import (
    RegionReader,
    SafeSceneSource,
    default_scene_dir,
    normalize_chip,
)

app = typer.Typer(add_completion=False, help=__doc__)
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure root logging once, at the entry point."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _load_config(path: Optional[Path]) -> RunConfig:
    """Load a config from YAML, or return defaults."""
    if path is None:
        logger.info("no --config given, using defaults")
        return RunConfig()
    cfg = RunConfig.from_yaml(path)
    logger.info("loaded config from %s (fingerprint %s)", path, cfg.fingerprint())
    return cfg


def write_run_manifest(
    out_dir: Path, cfg: RunConfig, model: OlmoEarthModel, extra: dict | None = None
) -> Path:
    """Write a run-level manifest next to the outputs.

    Args:
        out_dir: Directory to write into.
        cfg: The resolved config.
        model: The loaded model, for architecture and version facts.
        extra: Anything else worth recording.

    Returns:
        The manifest path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "olmoearth_activations_version": __version__,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "model": model.manifest(),
        **(extra or {}),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", path)
    return path


@app.command("encode-location")
def encode_location_cmd(
    location: Optional[str] = typer.Option(
        None,
        help=(
            "Name from configs/locations.yaml. Supplies the coordinate, the "
            "scene directory (data/scenes/<name>) and the output label, so a "
            "run is named rather than typed as digits."
        ),
    ),
    lat: Optional[float] = typer.Option(
        None, help="Latitude in degrees (EPSG:4326). Overrides --location."
    ),
    lon: Optional[float] = typer.Option(
        None, help="Longitude in degrees (EPSG:4326). Overrides --location."
    ),
    scene_dir: Optional[Path] = typer.Option(
        None,
        help=(
            "Directory holding the *.SAFE scene. Defaults to "
            "data/scenes/<location>, or data/scenes without --location."
        ),
    ),
    registry: Optional[Path] = typer.Option(
        None, help="Alternative locations YAML."
    ),
    config: Optional[Path] = typer.Option(None, help="YAML run config."),
    out: Path = typer.Option(
        Path("artifacts/encode"), help="Output directory."
    ),
    name: Optional[str] = typer.Option(
        None, help="Label recorded in the result metadata."
    ),
    allow_path_fallback: bool = typer.Option(
        False,
        help=(
            "Download config+weights directly if the installed ModelID enum "
            "lacks the requested model. See loader.OlmoEarthModel.load."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode one location and write embeddings, activations, and a manifest."""
    _setup_logging(verbose)

    # A named location fills in three things at once; each stays overridable so
    # a one-off probe near a known site needs no registry edit.
    if location is not None:
        site = locations.resolve(location, registry)
        lat = lat if lat is not None else site.lat
        lon = lon if lon is not None else site.lon
        scene_dir = scene_dir or locations.scene_dir_for(site.name)
        name = name or site.name
        logger.info(
            "location %r -> (%.5f, %.5f), scenes in %s",
            site.name,
            lat,
            lon,
            scene_dir,
        )
    if lat is None or lon is None:
        raise typer.BadParameter(
            "pass --location, or both --lat and --lon"
        )
    if scene_dir is None:
        scene_dir = default_scene_dir()

    cfg = _load_config(config)
    model = OlmoEarthModel.load(cfg.model, allow_path_fallback=allow_path_fallback)
    source = SafeSceneSource(scene_dir=scene_dir)

    raw = source.read(lat, lon, cfg.tile.chip_px)
    chip = normalize_chip(raw, cfg.tile.normalize)
    encoder = Encoder(model, cfg.tile, cfg.tap)
    result = encoder.encode(
        chip,
        latlon=(lat, lon),
        extra_meta={
            "name": name,
            "config_fingerprint": cfg.fingerprint(),
            "scene_dir": str(scene_dir),
        },
    )

    out.mkdir(parents=True, exist_ok=True)
    stem = name or f"{lat:.5f}_{lon:.5f}"
    result.save(out / f"{stem}.npz", dtype=cfg.save_dtype)
    # Keep the raw chip so figures can show imagery without re-reading the scene.
    np.savez_compressed(out / f"{stem}_chip.npz", chip=raw.astype(np.float32))
    write_run_manifest(
        out,
        cfg,
        model,
        {"locations": [{"name": name, "lat": lat, "lon": lon}]},
    )

    typer.echo(
        f"taps: {result.tap_labels}\n"
        f"grid: {result.grid_shape}  dim: {result.embed_dim}\n"
        f"wrote: {out / f'{stem}.npz'}"
    )


@app.command("sweep-position")
def sweep_position_cmd(
    lat: float = typer.Option(..., help="Latitude of the tracked ground patch."),
    lon: float = typer.Option(..., help="Longitude of the tracked ground patch."),
    scene_dir: Path = typer.Option(
        default_scene_dir(), help="Directory with *.SAFE scenes."
    ),
    config: Optional[Path] = typer.Option(None, help="YAML run config."),
    tap: str = typer.Option(
        "-1", help="Tap label (e.g. 'blk4') or index. Default: deepest."
    ),
    out: Path = typer.Option(Path("artifacts/sweep"), help="Output directory."),
    allow_path_fallback: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Track one ground patch as it moves across the chip.

    Slides the chip one patch at a time so the same ground sits at every
    horizontal position within it, and records that patch's vector each time.
    This isolates the effect of *context position* from the effect of content:
    the ground is identical across rows, only the surroundings the model could
    see have changed.
    """
    _setup_logging(verbose)
    cfg = _load_config(config)
    model = OlmoEarthModel.load(cfg.model, allow_path_fallback=allow_path_fallback)

    chip_px, patch_px = cfg.tile.chip_px, cfg.tile.patch_px
    grid = chip_px // patch_px

    # Region geometry: as tall as the chip, and wide enough for the chip to
    # slide grid-1 patches. The target pixel is placed so that chip k starts at
    # column patch_px*(grid-1-k), putting the target at patch column k.
    region_h, region_w = chip_px, chip_px + patch_px * (grid - 1)
    reader = RegionReader(SafeSceneSource(scene_dir=scene_dir))
    region = reader.read_region(
        lat, lon, region_h, region_w, chip_px // 2, patch_px * (grid - 1)
    )
    # Normalizing the whole region once is identical to normalizing each chip,
    # because both strategies use fixed constants -- see tiles.py.
    normalized = normalize_chip(region.array, cfg.tile.normalize)

    encoder = Encoder(model, cfg.tile, cfg.tap)
    tap_sel: str | int = tap if not _is_int(tap) else int(tap)
    patch_row = (chip_px // 2) // patch_px

    rows: list[np.ndarray] = []
    for k in range(grid):
        col0 = patch_px * (grid - 1 - k)
        sub = normalized[:, col0 : col0 + chip_px, :]
        result = encoder.encode(sub, latlon=(lat, lon))
        rows.append(result.grid(tap_sel)[patch_row, k, :])

    matrix = np.stack(rows, axis=0)

    # Neighbouring positions should be similar but not identical: same ground,
    # different surroundings. A cosine of exactly 1.0 everywhere would mean the
    # chip never actually moved.
    a, b = matrix[:-1], matrix[1:]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = (a * b).sum(-1) / np.maximum(denom, 1e-12)

    out.mkdir(parents=True, exist_ok=True)
    fig, spread = viz.plot_feature_matrix(
        matrix,
        row_label="patch position within chip",
        title=f"target patch vector by position in chip, tap {tap}",
    )
    viz.save(fig, str(out / "position_sweep.png"))
    np.savez_compressed(
        out / "position_sweep.npz",
        rows=matrix.astype(np.float32),
        column_range=spread,
        adjacent_cosine=cosine,
        positions=np.arange(grid),
        patch_row=patch_row,
        region_origin_rowcol=np.array(region.origin_rowcol),
        target_pixel_rowcol=np.array(region.target_pixel_rowcol),
        tap=str(tap),
    )
    write_run_manifest(out, cfg, model, {"sweep": {"lat": lat, "lon": lon, "tap": tap}})

    typer.echo(
        f"adjacent-position cosine: min {cosine.min():.4f} "
        f"max {cosine.max():.4f} mean {cosine.mean():.4f}\n"
        f"wrote {out / 'position_sweep.npz'}"
    )


@app.command("inspect-dimensions")
def inspect_dimensions_cmd(
    npz: Path = typer.Option(..., help="A .npz written by encode-location."),
    chip_npz: Optional[Path] = typer.Option(
        None, help="Matching *_chip.npz, to draw the RGB composite alongside."
    ),
    top_k: int = typer.Option(9, help="How many dimensions to plot."),
    tap: str = typer.Option("-1", help="Tap label or index."),
    out: Path = typer.Option(Path("artifacts/figures"), help="Output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Plot the most spatially variable dimensions of a saved result."""
    _setup_logging(verbose)
    from olmoearth_activations.encode import EmbeddingResult

    result = EmbeddingResult.load(npz)
    tap_sel: str | int = tap if not _is_int(tap) else int(tap)
    out.mkdir(parents=True, exist_ok=True)

    fig, stats = viz.plot_top_dimensions(result, k=top_k, tap=tap_sel)
    viz.save(fig, str(out / "top_dimensions.png"))
    stats.to_csv(out / "top_dimensions.csv", index=False)

    chip = None
    if chip_npz is not None:
        with np.load(chip_npz) as data:
            chip = data["chip"]

    best = int(stats.iloc[0]["dim"])
    viz.save(
        viz.plot_dimension_map(result, best, tap_sel, chip=chip),
        str(out / f"dim{best:03d}_map.png"),
    )
    viz.save(
        viz.plot_depth_comparison(result, best),
        str(out / f"dim{best:03d}_depth.png"),
    )
    analysis.depth_drift(result).to_csv(out / "depth_drift.csv", index=False)

    typer.echo(
        f"most variable dim at {tap}: {best}\n"
        f"taps: {result.tap_labels}\n"
        f"wrote figures to {out}"
    )


def _is_int(value: str) -> bool:
    """True if the string parses as an integer (including negatives)."""
    try:
        int(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    app()
