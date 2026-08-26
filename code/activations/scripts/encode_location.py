#!/usr/bin/env python
"""Encode a single location. Thin wrapper around the CLI.

Scenes are read from the repo's ``data/scenes`` unless --scene-dir says
otherwise.

Example:
    python scripts/encode_location.py --lat 38.90324 --lon -77.036964 \
        --config configs/default.yaml \
        --out artifacts/dc --name dc_downtown
"""

from __future__ import annotations

import typer

from olmoearth_activations.cli import encode_location_cmd

app = typer.Typer(add_completion=False)
app.command()(encode_location_cmd)

if __name__ == "__main__":
    app()
