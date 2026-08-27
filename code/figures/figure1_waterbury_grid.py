"""
3×2 RGB embedding grid for Waterbury, CT.

Panels (row-major):
  [0,0] OlmoEarth Nano       [0,1] OlmoEarth v1.1-Base
  [1,0] Clay v1.5             [1,1] Prithvi-EO-2.0 tiny
  [2,0] Prithvi-EO-2.0 300M-TL  [2,1] (blank — AlphaEarth, TBD)

Each panel is a PCA(3) RGB projection of the raw embedding window.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from figure1_rgb_panel import make_rgb_panel

SCRATCH = os.environ["SCRATCH"]
# TIF_BASE may be overridden to L_SCRATCH by the sbatch wrapper for faster I/O.
TIF_BASE = os.environ.get("TIF_BASE", f"{SCRATCH}/embeddings-health")
OUT_DIR  = f"{SCRATCH}/embeddings-health/figures"

LAT, LON = 41.5582, -73.0515   # Waterbury, CT
BOX_KM = 24.0

PANELS = [
    {
        "tif": f"{TIF_BASE}/olmoearth_nano_embeddings/CT/olmoearth_Nano_CT_2022.tif",
        "label": "OlmoEarth Nano",
    },
    {
        "tif": f"{TIF_BASE}/olmoearth_embeddings/CT/olmoearth_v1_1-Base_CT_2022.tif",
        "label": "OlmoEarth v1.1-Base",
    },
    {
        "tif": f"{TIF_BASE}/clay_embeddings/CT/clay_v1.5_CT_2022.tif",
        "label": "Clay v1.5",
    },
    {
        "tif": f"{TIF_BASE}/prithvi_embeddings/tiny/CT/prithvi_tiny_CT_2022_raw.tif",
        "label": "Prithvi-EO-2.0 tiny",
    },
    {
        "tif": f"{TIF_BASE}/prithvi_embeddings/300M-TL/CT/prithvi_300M-TL_CT_2022_raw.tif",
        "label": "Prithvi-EO-2.0 300M-TL",
    },
]

OUT = Path(f"{OUT_DIR}/waterbury_grid.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

NROWS, NCOLS = 3, 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 4))

for idx, panel in enumerate(PANELS):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]
    tif = panel["tif"]
    print(f"[{idx+1}/5] {panel['label']} ...")
    rgb, explained = make_rgb_panel(tif, LAT, LON, BOX_KM)
    print(f"       shape={rgb.shape}  PCA variance={explained:.1%}")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(panel["label"], fontsize=10, pad=4)
    ax.set_axis_off()

# Blank panel for AlphaEarth
blank_ax = axes[2, 1]
blank_ax.set_facecolor("#1a1a1a")
blank_ax.text(
    0.5, 0.5, "AlphaEarth\n(coming soon)",
    ha="center", va="center",
    fontsize=10, color="#666666",
    transform=blank_ax.transAxes,
)
blank_ax.set_axis_off()

fig.suptitle("Waterbury, CT — 24 km × 24 km  |  PCA RGB projections  (2022)",
             fontsize=12, y=1.01)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}")
