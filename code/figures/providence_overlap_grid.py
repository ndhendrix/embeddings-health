"""PCA RGB panels for Providence overlap/center-crop embeddings."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from figure1_rgb_panel import make_rgb_panel, plot_panel

LAT,LON=41.8240,-71.4128; BOX_KM=24.0
ROOT=Path(os.environ.get("PROVIDENCE_ROOT",f"{os.environ['SCRATCH']}/embeddings-health/providence_overlap_v1"))
OUT=ROOT/"figures"; OUT.mkdir(parents=True,exist_ok=True)
panels=[
 ("olmoearth-v1.2-nano",ROOT/"olmoearth-v1.2-nano/RI/olmoearth-v1.2-nano_overlap-center50_RI_2022.tif","OLMoEarth v1.2 Nano"),
 ("olmoearth-v1.2-base",ROOT/"olmoearth-v1.2-base/RI/olmoearth-v1.2-base_overlap-center50_RI_2022.tif","OLMoEarth v1.2 Base"),
 ("clay-1.5",ROOT/"clay-1.5/RI/clay-1.5_overlap-center50_RI_2022.tif","Clay 1.5"),
]
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(key,tif,label) in zip(axes,panels):
    rgb,variance=make_rgb_panel(tif,LAT,LON,BOX_KM)
    plot_panel(rgb,f"{label} | PCA3 {variance:.1%}",OUT/f"providence_{key}_pca_rgb.png",dpi=200)
    ax.imshow(rgb); ax.set_title(f"{label}\nPCA3 variance: {variance:.1%}"); ax.set_axis_off()
fig.suptitle("Providence, RI — 24 km × 24 km — overlap/center-crop PCA RGB (2022)")
fig.tight_layout(); target=OUT/"providence_overlap_pca_rgb_grid.png"; fig.savefig(target,dpi=200,bbox_inches="tight"); plt.close(fig); print(target)
