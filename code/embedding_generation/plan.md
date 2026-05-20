# Plan: OlmoEarth 1.1 and Prithvi-EO-2.0 Embedding Generation for CONUS

## Goal

Produce per-pixel (or near-pixel) geospatial embeddings for the contiguous United States using two open-weight foundation models — OlmoEarth 1.1 (Allen AI) and Prithvi-EO-2.0 (IBM/NASA) — formatted as multi-band Cloud-Optimized GeoTIFFs (COGs) matching the tile layout and census-tract aggregation conventions of the AlphaEarth Foundations data already in this project.

## Decisions

| Decision | Choice |
|---|---|
| OlmoEarth variant | Base (89M); architecture parameterized so Large (308M) is a single config change |
| Prithvi variant | 300M; 600M is a single config change |
| Year | 2022 only for now |
| Embedding dims | PCA-compressed to 64 (matching AlphaEarth); disable with `--no-pca` CLI flag |
| OlmoEarth temporal input | Single annual composite (no multi-temporal stacking for now) |
| Test geography | Rhode Island (RI) — ~1–2 MGRS tiles, ~244 census tracts; fast enough to run locally |

## Python environment constraint

The main project uses Python 3.14, but PyTorch, `olmoearth_pretrain`, and `terratorch` require Python 3.11–3.12. The embedding pipeline lives in its own sub-project (`code/embedding_generation/`) with a separate `pyproject.toml` pinned to Python 3.11. Run via `uv run --python 3.11` from that directory. The output CSVs are the handoff point back to the main project.

---

## Why this is non-trivial

AlphaEarth ships precomputed embeddings. OlmoEarth and Prithvi are open weights with no precomputed products, so we need to (1) source and preprocess Sentinel-2 imagery at scale, (2) run GPU inference across all of CONUS, and (3) reassemble patch-level model outputs into spatially coherent rasters. Each step has its own scale challenges.

---

## Target output format

| Property | AlphaEarth (target) | OlmoEarth (planned) | Prithvi (planned) |
|---|---|---|---|
| Format | COG GeoTIFF | COG GeoTIFF | COG GeoTIFF |
| Spatial resolution | 10 m/px | ~40 m/px (see note) | ~30 m/px |
| Embedding dims | 64 | 768 (Base) | TBD from config.json |
| Value encoding | 8-bit signed int | float32 (or 8-bit quantized) | float32 (or 8-bit quantized) |
| Tiling scheme | UTM 8192×8192 px | Same UTM zones | Same UTM zones |
| Temporal coverage | Annual (2017–2024) | Annual composite | Annual composite |

**Resolution note:** OlmoEarth uses ViT patch size = 4 pixels. If we feed 10 m Sentinel-2 imagery, each output token covers a 4×4 pixel = 40 m × 40 m area. We store embeddings at this 40 m resolution and resample to 10 m only if needed for census-tract aggregation (nearest neighbor). Prithvi was trained on 30 m HLS data, so output stays at 30 m. AlphaEarth's 10 m per-pixel advantage is a meaningful difference we should document.

The downstream product (census-tract-level CSV with `{DIM}_{MEAN,MEDIAN,MAX,MIN,STD}` columns) can be produced from any of these resolutions given that even at 40 m the average US census tract (~40 km²) still contains ~25,000 output pixels.

---

## Year(s) to produce

**2022** (confirmed). Extend to 2017–2022 once validated using the same code with a `--year` CLI flag.

---

## Step 1: Enumerate CONUS MGRS tiles

Sentinel-2 is organized in MGRS tiles (110 km × 110 km at 10 m). Roughly **470 MGRS tiles** cover CONUS. We get the tile list from the AWS sentinel-cogs STAC catalog by querying with a CONUS bounding box and extracting unique `mgrs:utm_zone` + `mgrs:latitude_band` + `mgrs:grid_square` combinations from any 2022 scene.

We then reproject those tile footprints to UTM zones to define our output GeoTIFF coordinate system — matching AlphaEarth's UTM-zone tiling.

**Output:** `conus_mgrs_tiles.csv` (~470 rows, with MGRS tile ID, UTM EPSG code, bounding box).

---

## Step 2: Sentinel-2 L2A ingestion and annual compositing

**Source:** AWS Open Data bucket `s3://sentinel-cogs` (public, no credentials needed), accessed via `pystac-client` and `odc-stac`.

**Process for each MGRS tile:**
1. Query the STAC API for all 2022 scenes intersecting the tile.
2. Load bands needed by each model (see below) plus the Scene Classification Layer (SCL) for cloud masking.
3. Apply SCL cloud mask: exclude SCL classes 0 (no data), 1 (saturated), 3 (cloud shadow), 8–10 (cloud probabilities), 11 (snow — optional).
4. Compute a **pixel-wise annual median** across all valid observations. This is the standard cloud-free composite strategy and aligns with AlphaEarth's annual temporal aggregation.
5. Write the composite as a temporary float32 GeoTIFF.

**Bands per model:**
- **OlmoEarth:** 12 bands — B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (all Sentinel-2 MSI bands except B10).
- **Prithvi:** 6 bands — B02 (Blue), B03 (Green), B04 (Red), B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2). Values must be scaled to surface reflectance (divide DN by 10000).

For **Prithvi** specifically, 3 timesteps are expected rather than a single composite. We define three seasonal windows:
- **Spring:** March–May median
- **Summer:** June–August median  
- **Fall:** September–November median

Winter is excluded to reduce snow/bare-soil artifacts; this matches phenological signal that Prithvi's HLS training emphasized.

---

## Step 3: Chip-based GPU inference

Both models are ViTs that operate on fixed-size image chips. The inference strategy:

1. Slide a window across the composited tile with **no overlap** (stride = chip size). Overlap could improve edge quality but would quadruple compute; for initial production, skip it.
2. For each chip:
   - **OlmoEarth:** Feed a single chip (224×224 px at 10 m) as a time-series or single-timestep input. Extract the spatial **patch token outputs** from the final encoder layer (shape: 56×56×768). This gives one 768-dim vector per 40 m × 40 m spatial location in the chip.
   - **Prithvi:** Feed 3 chips (one per season, 224×224 px at 10 m — but resampled to 30 m internally). Extract the spatial patch tokens from the encoder. The output embedding dimension will be confirmed from the model's config.json before implementation.
3. Write patch embeddings for each chip into an in-memory array indexed by spatial position.
4. Assemble chip outputs into a full-tile raster.

**Compute estimate (OlmoEarth Base, 89M params):**
- CONUS land area ≈ 7.7M km²
- Chip size: 224 px × 10 m = 2.24 km
- Non-overlapping chips: 7.7M / (2.24)² ≈ 1.5M chips
- Throughput on one A100 GPU: ~20 chips/sec (batch size 16–32)
- Single GPU: ~21 hours; **8× A100s: ~2.6 hours**
- Prithvi is similar but x3 for the seasonal pass; estimate ~8 hours on 8 GPUs.

The recommended platform is an **AWS p3.16xlarge** (8× V100, ~$24/hr) or **p4d.24xlarge** (8× A100). Total compute cost per year is likely **$50–200** depending on instance type and spot pricing.

---

## Step 4: Raster assembly and COG output

After inference, for each UTM output tile:
1. Stack the assembled embedding arrays into a multi-band raster (one band per embedding dimension).
2. Optionally quantize to 8-bit signed int using per-band linear scaling (matching AlphaEarth's encoding) to reduce storage ~4×.
3. Write as a Cloud-Optimized GeoTIFF with LZW compression and internal overview pyramids (1:2, 1:4, 1:8, 1:16 via mean aggregation with L2 renormalization to match AlphaEarth).
4. Name files consistently: `{model}_{utm_zone}_{year}.tif`.

**Storage estimate:**
- OlmoEarth (768 dims, float32, 40 m, CONUS): ~500 tiles × (8192/4)² × 768 × 4 bytes ≈ **5 TB** uncompressed; ~500 GB with 8-bit + LZW.
- Prithvi (TBD dims, 30 m): similar scale.

We should store final COGs in an **S3 bucket** with requester-pays or private access, not locally.

---

## Step 5: Census-tract aggregation (matching existing CSV format)

This step mirrors how AlphaEarth embeddings were already aggregated for this repo:

1. Load US census tract shapefiles (Tiger/Line, 2020 vintage, same as AlphaEarth data in repo).
2. For each census tract, extract all embedding pixels within its boundary using `rasterio.mask`.
3. Compute mean, median, max, min, std across pixels for each embedding dimension.
4. Output: one CSV per model per year with columns `GEOID, year, {DIM_MEAN, DIM_MEDIAN, DIM_MAX, DIM_MIN, DIM_STD}` — same schema as `alphaearth_embeddings.csv`.

Because OlmoEarth has 768 dimensions vs AlphaEarth's 64, the downstream analysis code will need to handle variable dimensionality. Column naming convention: `E{000–767}_{STAT}` for OlmoEarth and `P{000–NNN}_{STAT}` for Prithvi.

---

## Proposed directory structure

```
code/
  embedding_generation/
    plan.md                    ← this file
    00_enumerate_tiles.py      ← MGRS tile list for CONUS
    01_s2_composite.py         ← per-tile Sentinel-2 annual composite
    02_inference_olmoearth.py  ← OlmoEarth chip inference → COG
    02_inference_prithvi.py    ← Prithvi chip inference → COG
    03_tract_aggregation.py    ← COG → census-tract CSV
    utils/
      cog_writer.py
      cloud_mask.py
      stac_query.py
data/
  olmoearth_embeddings.csv    ← final output (same schema as alphaearth)
  prithvi_embeddings.csv
```

---

## Things to verify at implementation time

1. **Prithvi band ordering:** The paper specifies Blue/Green/Red/Narrow NIR/SWIR1/SWIR2 = B02, B03, B04, B8A, B11, B12. Confirm against `config.json` on HuggingFace before running.

2. **Prithvi timesteps:** Paper cites n_timesteps=3; one web source said 4. Confirm from `config.json`. Our plan uses 3 seasonal windows but this controls input tensor shape.

3. **Prithvi normalization scale:** Are mean/std values (e.g., 1087, 2248 for Blue) in raw S2 DN units (0–10000) or reflectance (0–1)? Determines whether we divide by 10000 before passing to model. Check model card.

4. **OlmoEarth chip size:** One source cited 128×128, not 224×224. Confirm from source or rslearn docs before running.

5. **OlmoEarth normalization:** `OlmoEarthNormalize` transform handles band-specific stats internally when calling `model.encode()`; confirm we don't need to pre-normalize.
