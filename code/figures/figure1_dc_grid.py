"""
3×2 RGB embedding grid for Washington, DC.

Panels (row-major):
  [0,0] OlmoEarth Nano        [0,1] Clay v1.5
  [1,0] Prithvi-EO tiny-TL   [1,1] Prithvi-EO 300M-TL
  [2,0] (blank — AlphaEarth) [2,1] (blank — reserved)

Window: 24 km × 24 km centered on (38.9072°N, 77.0369°W).
Mosaic: DC + MD (EPSG:32618) + VA (EPSG:32617, reprojected on-the-fly).
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure1_rgb_panel import make_rgb_panel

SCRATCH = os.environ["SCRATCH"]
# STAGE is set by the sbatch wrapper to L_SCRATCH for staged files.
# Clay files are too large to stage; they are read directly from SCRATCH.
STAGE   = os.environ.get("STAGE_DIR", f"{SCRATCH}/embeddings-health")
SRC     = f"{SCRATCH}/embeddings-health"
OUT_DIR = f"{SCRATCH}/embeddings-health/figures"

LAT, LON = 38.9072, -77.0369
BOX_KM   = 24.0

PANELS = [
    {
        "tifs": [
            f"{STAGE}/olmoearth_nano_embeddings/DC/olmoearth_Nano_DC_2022.tif",
            f"{STAGE}/olmoearth_nano_embeddings/MD/olmoearth_Nano_MD_2022.tif",
            f"{STAGE}/olmoearth_nano_embeddings/VA/olmoearth_Nano_VA_2022.tif",
        ],
        "label": "OlmoEarth Nano",
    },
    {
        "tifs": [
            f"{SRC}/clay_embeddings/DC/clay_v1.5_DC_2022.tif",
            f"{SRC}/clay_embeddings/MD/clay_v1.5_MD_2022.tif",
            f"{SRC}/clay_embeddings/VA/clay_v1.5_VA_2022.tif",
        ],
        "label": "Clay v1.5",
    },
    {
        "tifs": [
            f"{STAGE}/prithvi_embeddings/tiny/DC/prithvi_tiny_DC_2022_raw.tif",
            f"{STAGE}/prithvi_embeddings/tiny/MD/prithvi_tiny_MD_2022_raw.tif",
            f"{STAGE}/prithvi_embeddings/tiny/VA/prithvi_tiny_VA_2022_raw.tif",
        ],
        "label": "Prithvi-EO tiny-TL",
    },
    {
        "tifs": [
            f"{STAGE}/prithvi_embeddings/300M-TL/DC/prithvi_300M-TL_DC_2022_raw.tif",
            f"{STAGE}/prithvi_embeddings/300M-TL/MD/prithvi_300M-TL_MD_2022_raw.tif",
            f"{STAGE}/prithvi_embeddings/300M-TL/VA/prithvi_300M-TL_VA_2022.tif",
        ],
        "label": "Prithvi-EO 300M-TL",
    },
]

OUT = Path(f"{OUT_DIR}/dc_grid.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

NROWS, NCOLS = 3, 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 4))

for idx, panel in enumerate(PANELS):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]
    print(f"[{idx+1}/{len(PANELS)}] {panel['label']} ...", flush=True)
    rgb, explained = make_rgb_panel(panel["tifs"], LAT, LON, BOX_KM)
    print(f"       shape={rgb.shape}  PCA variance={explained:.1%}", flush=True)

    # Save standalone panel
    slug = panel["label"].lower().replace(" ", "_").replace("-", "_").replace(".", "")
    solo = Path(f"{OUT_DIR}/dc_{slug}.png")
    fig_s, ax_s = plt.subplots(figsize=(6, 6))
    ax_s.imshow(rgb, interpolation="nearest")
    ax_s.set_title(f"Washington DC — {panel['label']}", fontsize=10, pad=6)
    ax_s.set_axis_off()
    fig_s.tight_layout(pad=0.5)
    fig_s.savefig(solo, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_s)
    print(f"       saved → {solo}", flush=True)

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(panel["label"], fontsize=10, pad=4)
    ax.set_axis_off()

# Blank panels for AlphaEarth and reserved
for blank_idx in [4, 5]:
    row, col = divmod(blank_idx, NCOLS)
    ax = axes[row, col]
    ax.set_facecolor("#1a1a1a")
    label = "AlphaEarth\n(coming soon)" if blank_idx == 4 else ""
    if label:
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=10, color="#666666", transform=ax.transAxes)
    ax.set_axis_off()

fig.suptitle("Washington, DC — 24 km × 24 km  |  PCA RGB projections  (2022)",
             fontsize=12, y=1.01)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}", flush=True)
