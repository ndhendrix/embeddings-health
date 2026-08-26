#!/usr/bin/env python
"""Prove the whole forward path works, with no satellite imagery required.

This is the first thing to run. It downloads a checkpoint, builds a synthetic
chip of plausible reflectance values, and encodes it -- exercising model
loading, sample construction, both tap mechanisms, the flat-to-grid reshape and
the save/load round trip. What it cannot tell you is whether your imagery is
being read correctly; for that you need a real scene or composite.

It also reports two things that were documented from source but never measured:
whether ``fast_pass`` changes the numbers, and how hook taps relate to
``token_exit`` taps. Both print actual values rather than assertions, because
the point is to find out.

Example:
    python scripts/smoke_test.py --config configs/default.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from olmoearth_activations import analysis
from olmoearth_activations.config import RunConfig, TapConfig
from olmoearth_activations.encode import Encoder, EmbeddingResult
from olmoearth_activations.loader import OlmoEarthModel

app = typer.Typer(add_completion=False)


def synthetic_chip(chip_px: int, n_bands: int, seed: int = 0) -> np.ndarray:
    """A chip of plausible Sentinel-2 digital numbers.

    Values sit in the 0-10000 range the L2A COGs use, with a per-band offset so
    the bands are distinguishable. Structure is added as a diagonal gradient so
    the token grid has something spatial to encode -- uniform noise would make
    every patch interchangeable and hide a reshape bug.

    Args:
        chip_px: Side length in pixels.
        n_bands: Channel count.
        seed: RNG seed.

    Returns:
        ``(chip_px, chip_px, n_bands)`` float32 array.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(200.0, 3000.0, size=(chip_px, chip_px, n_bands))
    ramp = np.add.outer(
        np.linspace(0.0, 1200.0, chip_px), np.linspace(0.0, 1200.0, chip_px)
    )
    offsets = np.arange(n_bands, dtype=np.float64) * 80.0
    return (noise + ramp[:, :, None] + offsets).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-patch cosine similarity between two (H, W, D) grids."""
    x, y = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.mean((x * y).sum(axis=1) / np.maximum(denom, 1e-12)))


@app.command()
def main(
    config: Optional[Path] = typer.Option(None, help="YAML run config."),
    out: Path = typer.Option(
        Path("artifacts/smoke"), help="Where to write the result."
    ),
    allow_path_fallback: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run every code path that does not need imagery, and report findings."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    cfg = RunConfig.from_yaml(config) if config else RunConfig()
    typer.echo(f"config fingerprint: {cfg.fingerprint()}")

    typer.echo("\n[1/6] loading model (downloads on first run)...")
    model = OlmoEarthModel.load(cfg.model, allow_path_fallback=allow_path_fallback)
    typer.echo(
        f"      {model.model_value}: depth={model.depth} "
        f"embed_dim={model.embed_dim} band_sets={model.num_band_sets}"
    )
    typer.echo(f"      blocks: {model.block_names[0]} .. {model.block_names[-1]}")
    typer.echo(f"      band order: {' '.join(model.band_order)}")

    chip = synthetic_chip(cfg.tile.chip_px, len(model.band_order))
    typer.echo(f"\n[2/6] synthetic chip {chip.shape}, raw DN range "
               f"{chip.min():.0f}-{chip.max():.0f}")

    typer.echo("\n[3/6] encoding with token_exit taps...")
    exit_res = Encoder(model, cfg.tile, TapConfig(method="token_exit")).encode(chip)
    typer.echo(
        f"      taps {exit_res.tap_labels}\n"
        f"      grid {exit_res.grid_shape}  dim {exit_res.embed_dim}\n"
        f"      activations {exit_res.activations.shape}"
    )

    typer.echo("\n[4/6] determinism + fast_pass...")
    again = Encoder(model, cfg.tile, TapConfig(method="token_exit")).encode(chip)
    identical = np.array_equal(exit_res.activations, again.activations)
    typer.echo(f"      two runs bitwise identical: {identical}")

    slow = Encoder(
        model, cfg.tile, TapConfig(method="token_exit", fast_pass=False)
    ).encode(chip)
    fp_diff = float(np.max(np.abs(exit_res.embeddings - slow.embeddings)))
    typer.echo(
        f"      fast_pass True vs False, max abs diff: {fp_diff:.3e} "
        f"({'identical' if fp_diff == 0.0 else 'DIFFERENT -- investigate'})"
    )

    typer.echo("\n[5/6] hooks vs token_exit (the documented-but-unmeasured bit)...")
    hook_res = Encoder(model, cfg.tile, TapConfig(method="hooks")).encode(chip)
    typer.echo(f"      hook taps {hook_res.tap_labels}")
    for depth in hook_res.tap_depths:
        label = f"blk{depth}"
        h, e = hook_res.grid(label), exit_res.grid(label)
        ratio = float(
            np.mean(np.linalg.norm(h.reshape(-1, h.shape[-1]), axis=1))
            / max(np.mean(np.linalg.norm(e.reshape(-1, e.shape[-1]), axis=1)), 1e-12)
        )
        typer.echo(
            f"      {label}: cosine={_cosine(h, e):.6f} "
            f"max_abs_diff={float(np.max(np.abs(h - e))):.4e} norm_ratio={ratio:.4f}"
        )
    typer.echo(
        "      (a cosine near 1 with norm_ratio far from 1 means the taps differ\n"
        "       by the final LayerNorm's rescaling, as expected)"
    )

    typer.echo("\n[6/6] analysis + round trip...")
    stats = analysis.most_variable_dims(exit_res, k=5)
    typer.echo("      most variable dims at deepest tap:")
    for row in stats.itertuples(index=False):
        typer.echo(f"        dim {int(row.dim):3d}  sd={row.std:.4g}  range={row.range:.4g}")
    typer.echo("      depth drift (consecutive-tap cosine):")
    for row in analysis.depth_drift(exit_res).itertuples(index=False):
        typer.echo(
            f"        {row.from_tap:>5s} -> {row.to_tap:<5s} mean={row.mean_cosine:.4f}"
        )

    out.mkdir(parents=True, exist_ok=True)
    path = exit_res.save(out / "smoke.npz", dtype=cfg.save_dtype)
    reloaded = EmbeddingResult.load(path)
    ok = np.allclose(reloaded.activations, exit_res.activations, rtol=1e-2, atol=1e-2)
    typer.echo(f"      saved {path}, reload matches: {ok}")

    typer.echo("\nSmoke test complete. The forward path works.")
    typer.echo(
        "Note: this says nothing about imagery reading -- no scene was involved."
    )


if __name__ == "__main__":
    app()
