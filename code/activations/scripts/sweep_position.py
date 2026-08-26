#!/usr/bin/env python
"""Track one ground patch as it slides across the chip. Thin CLI wrapper.

This is the port of the prototype ``sweep_patch.py`` onto the library. The
prototype's result on a DC location is worth knowing before re-running it:
adjacent-position cosine similarity came out between 0.9948 and 0.9998, i.e.
context position had almost no effect at the deepest layer. That is a useful
null, so treat a repeat run as a confirmation rather than an open question.

Scenes are read from the repo's ``data/scenes`` unless --scene-dir says
otherwise.

Example:
    python scripts/sweep_position.py --lat 38.90324 --lon -77.036964 \
        --config configs/default.yaml \
        --out artifacts/sweep_dc
"""

from __future__ import annotations

import typer

from olmoearth_activations.cli import sweep_position_cmd

app = typer.Typer(add_completion=False)
app.command()(sweep_position_cmd)

if __name__ == "__main__":
    app()
