"""
PCA-based RGB validation map for RI OlmoEarth embeddings.

For each tract GeoTIFF, computes the spatial mean to get one 192-dim vector per tract,
fits PCA across all tracts, maps the first 3 components to RGB, and plots a choropleth.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
TIFS_DIR = PROJECT_ROOT / "data" / "olmoearth" / "ri"
TRACTS_ZIP = PROJECT_ROOT / "data" / "tl_2022_44_tract.zip"
OUT_PNG = PROJECT_ROOT / "outputs" / "ri_olmoearth_rgb.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

tif_paths = sorted(TIFS_DIR.glob("*.tif"))
if not tif_paths:
    print(f"No GeoTIFFs found in {TIFS_DIR}. Run 02_collect_results.py first.")
    sys.exit(1)

print(f"Loading {len(tif_paths)} tract embeddings...")

geoids, vectors = [], []
for p in tif_paths:
    with rasterio.open(p) as src:
        data = src.read()          # (192, H, W), int8
        nodata = src.nodata        # -128
    mask = data[0] != nodata       # valid pixels (H, W)
    valid = data[:, mask].astype("float32")  # (192, n_valid)
    vectors.append(valid.mean(axis=1))       # spatial mean → (192,)
    geoids.append(p.stem)

X = np.stack(vectors)  # (n_tracts, 192)
print(f"  array shape: {X.shape}")

pca = PCA(n_components=3, random_state=0)
rgb_raw = pca.fit_transform(X)  # (n_tracts, 3)
print(f"  PCA explained variance: {pca.explained_variance_ratio_.round(3)}")

rgb = MinMaxScaler().fit_transform(rgb_raw)  # scale each channel to [0, 1]

tracts = gpd.read_file(f"zip://{TRACTS_ZIP}").to_crs("EPSG:4326")
tracts = tracts[tracts["GEOID"].isin(geoids)].copy()
tracts = tracts.set_index("GEOID")

tracts[["r", "g", "b"]] = rgb[
    [geoids.index(g) for g in tracts.index], :
]
tracts["color"] = [
    (row.r, row.g, row.b) for _, row in tracts[["r", "g", "b"]].iterrows()
]

fig, ax = plt.subplots(figsize=(10, 12))
tracts.plot(
    ax=ax,
    color=tracts["color"].tolist(),
    edgecolor="white",
    linewidth=0.3,
)
ax.set_axis_off()
ax.set_title(
    "Rhode Island census tracts\nOlmoEarth Studio embeddings — PCA RGB (2022)",
    fontsize=13,
    pad=12,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
