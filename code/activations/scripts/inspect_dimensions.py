#!/usr/bin/env python
"""Plot the most spatially variable dimensions of a saved result. Thin wrapper.

Needs no model and no GPU -- it reads a ``.npz`` written by ``encode_location``.
Pass the matching ``*_chip.npz`` to get the true-colour composite beside the
heatmap, which is the only way to judge whether an activation pattern
corresponds to anything on the ground.

Example:
    python scripts/inspect_dimensions.py \
        --npz artifacts/dc/dc_downtown.npz \
        --chip-npz artifacts/dc/dc_downtown_chip.npz \
        --top-k 9 --out artifacts/dc/figures
"""

from __future__ import annotations

import typer

from olmoearth_activations.cli import inspect_dimensions_cmd

app = typer.Typer(add_completion=False)
app.command()(inspect_dimensions_cmd)

if __name__ == "__main__":
    app()
