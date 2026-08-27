"""
3×2 RGB embedding grid for Palo Alto, CA.

Panels (row-major):
  [0,0] OlmoEarth Nano        [0,1] Clay v1.5
  [1,0] Prithvi-EO tiny-TL   [1,1] Prithvi-EO 300M-TL
  [2,0] (blank — AlphaEarth) [2,1] (blank — reserved)

Window: 24 km × 24 km centered on (37.4419°N, 122.1430°W).
Single CA file per model (EPSG:32610).
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure1_rgb_panel import make_rgb_panel

SCRATCH = os.environ["SCRATCH"]
STAGE   = os.environ.get("STAGE_DIR", f"{SCRATCH}/embeddings-health")
SRC     = f"{SCRATCH}/embeddings-health"
OUT_DIR = f"{SCRATCH}/embeddings-health/figures"

LAT, LON = 37.4419, -122.1430
BOX_KM   = 24.0

PANELS = [
    {
        "tifs": [f"{STAGE}/olmoearth_nano_embeddings/CA/olmoearth_Nano_CA_2022.tif"],
        "label": "OlmoEarth Nano",
    },
    {
        "tifs": [f"{SRC}/clay_embeddings/CA/clay_v1.5_CA_2022.tif"],
        "label": "Clay v1.5",
    },
    {
        "tifs": [f"{STAGE}/prithvi_embeddings/tiny/CA/prithvi_tiny_CA_2022_raw.tif"],
        "label": "Prithvi-EO tiny-TL",
    },
    {
        "tifs": [f"{STAGE}/prithvi_embeddings/300M-TL/CA/prithvi_300M-TL_CA_2022.tif"],
        "label": "Prithvi-EO 300M-TL",
    },
]

OUT = Path(f"{OUT_DIR}/palo_alto_grid.png")
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
    solo = Path(f"{OUT_DIR}/pa_{slug}.png")
    fig_s, ax_s = plt.subplots(figsize=(6, 6))
    ax_s.imshow(rgb, interpolation="nearest")
    ax_s.set_title(f"Palo Alto — {panel['label']}", fontsize=10, pad=6)
    ax_s.set_axis_off()
    fig_s.tight_layout(pad=0.5)
    fig_s.savefig(solo, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_s)
    print(f"       saved → {solo}", flush=True)

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(panel["label"], fontsize=10, pad=4)
    ax.set_axis_off()

# Blank panels
for blank_idx in [4, 5]:
    row, col = divmod(blank_idx, NCOLS)
    ax = axes[row, col]
    ax.set_facecolor("#1a1a1a")
    label = "AlphaEarth\n(coming soon)" if blank_idx == 4 else ""
    if label:
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=10, color="#666666", transform=ax.transAxes)
    ax.set_axis_off()

fig.suptitle("Palo Alto, CA — 24 km × 24 km  |  PCA RGB projections  (2022)",
             fontsize=12, y=1.01)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}", flush=True)
