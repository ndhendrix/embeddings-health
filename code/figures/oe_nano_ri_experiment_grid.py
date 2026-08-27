"""Providence PCA RGB review for the corrected OE-Nano RI experiment."""

import os
from pathlib import Path

import matplotlib.pyplot as plt

from figure1_rgb_panel import make_rgb_panel


LAT, LON = 41.8240, -71.4128
BOX_KM = 24.0
ROOT = Path(os.environ["OE_NANO_RI_ROOT"])
YEAR = os.environ.get("YEAR", "2022")
MODEL = "olmoearth-v1.2-nano"
STEM = f"{MODEL}_overlap-center50_RI_{YEAR}.tif"
OLD = Path(
    os.environ.get(
        "OLD_OE_NANO_PROVIDENCE",
        f"{os.environ['SCRATCH']}/embeddings-health/providence_overlap_v1/"
        f"{MODEL}/RI/{STEM}",
    )
)
PANELS = [
    (OLD, "Old OE-Nano\nunnormalized"),
    (ROOT / "reference" / MODEL / "RI" / STEM, "Corrected v2\nsingle task"),
    (ROOT / "rect2x2" / MODEL / "RI" / STEM, "Corrected v2\n2x2 merged"),
]


def main() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (raster, label) in zip(axes, PANELS):
        rgb, variance = make_rgb_panel(raster, LAT, LON, BOX_KM)
        axis.imshow(rgb, interpolation="nearest")
        axis.set_title(f"{label}\nPCA3 variance: {variance:.1%}")
        axis.set_axis_off()
    figure.suptitle("Providence, RI - 24 km x 24 km - OE-Nano artifact review")
    figure.tight_layout()
    output = ROOT / "figures" / "providence_oe_nano_old_vs_center50_v2.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
