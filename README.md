The goal of this project is to determine whether spatial embeddings (DeepMind AlphaEarth Foundations to start with) can explain more variability in health outcomes than traditional socioeconomic and environmental factors, including neighborhood social risk data. We disagregate this into two research questions:

1. Can spatial embeddings explain more variability in health outcomes than traditional socioeconomic and environmental factors, including neighborhood social risk data within a given year?
2. Can spatial embeddings better predict changes in health outcomes than traditional socioeconomic and environmental factors, including neighborhood social risk data? For instance, can 2017 embeddings predict changes in health outcomes between 2017 and 2022 better than 2017 socioeconomic and environmental factors, including neighborhood social risk data?

As an exploratory aim, we look to mechanistic interpretability research to guide us in the search for vectors within the embeddings that may be associated with more specific health outcomes. Just as vectors in general purpose LLMs encode information about factors like political ideology, we hypothesize that vectors in spatial embeddings may encode information about elements of the built and natural environments that are known to influence specific health outcomes.

---

## Embedding generation pipeline

The pipeline lives in `code/embedding_generation/` and requires Python 3.11–3.12. All commands below are run from that directory with `uv run python`.

### Step 1 — Sentinel-2 composites (`composite.py`)

Produces annual (OlmoEarth) or seasonal (Prithvi) cloud-free median GeoTIFFs from Sentinel-2 L2A imagery via the public AWS Element84 STAC catalog. No credentials required.

```bash
# Single state, Prithvi model (spring/summer/fall composites at 30 m)
uv run python composite.py --model prithvi --state RI --year 2022 \
    --max-scenes-per-month 8 --output-dir outputs/composites

# All CONUS states
uv run python composite.py --model prithvi --all-states --year 2022 \
    --max-scenes-per-month 8 --output-dir outputs/composites
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `olmoearth` | `olmoearth` (annual, 12 bands, 10 m) or `prithvi` (seasonal, 6 bands, 30 m) |
| `--state` | `RI` | Two-letter state abbreviation |
| `--all-states` | off | Process all 48 contiguous US states |
| `--year` | `2022` | Composite year |
| `--max-cloud-cover` | `30` | Maximum scene-level cloud cover % |
| `--max-scenes-per-month` | `3` | Max scenes kept per calendar month (clearest first). Use 8+ for good spatial coverage |
| `--max-tile-km` | `200` | Tiles states wider than this (km) to avoid memory limits |
| `--force` | off | Reprocess existing outputs instead of skipping |
| `--output-dir` | `outputs/composites` | Directory for output GeoTIFFs |

### Step 2 — Embedding inference (`embed.py`)

Runs chip-based GPU inference over a composite GeoTIFF and writes a Cloud-Optimized GeoTIFF of patch embeddings. Supports OlmoEarth 1.1 (AllenAI) and Prithvi-EO-2.0 (IBM/NASA). Automatically uses CUDA, Apple Silicon MPS, or CPU.

```bash
# Prithvi tiny (default variant) — 192-dim embeddings at 480 m resolution
uv run python embed.py --model prithvi \
    --input outputs/composites/s2_spring_RI_2022_prithvi.tif \
            outputs/composites/s2_summer_RI_2022_prithvi.tif \
            outputs/composites/s2_fall_RI_2022_prithvi.tif \
    --output outputs/embeddings/prithvi_tiny_RI_2022.tif \
    --raw-output outputs/embeddings/prithvi_tiny_RI_2022_raw.tif

# OlmoEarth Base — 768-dim embeddings at 80 m resolution
uv run python embed.py --model olmoearth \
    --input outputs/composites/s2_annual_RI_2022_olmoearth.tif \
    --output outputs/embeddings/olmoearth_RI_2022.tif
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `olmoearth` or `prithvi` |
| `--variant` | `tiny` (Prithvi) / `Base` (OlmoEarth) | Prithvi: `tiny`, `300M`, `600M`. OlmoEarth: `Base`, `Large` |
| `--input` | required | One TIF for OlmoEarth; one-to-four for Prithvi (spring, summer, fall, winter). If fewer TIFs than the model's expected frame count are supplied, the last TIF is repeated with a warning |
| `--output` | required | Output COG path (PCA-compressed by default) |
| `--raw-output` | off | Also save the pre-PCA raw embeddings to this path. Useful for re-running PCA/UMAP without redoing GPU inference |
| `--no-pca` | off | Skip PCA; write raw embeddings directly to `--output` |
| `--pca-dims` | `64` | PCA output dimensionality |
| `--pca-model` | auto | Path to a pre-fitted `.pkl` PCA model. If absent, a new one is fitted and saved alongside the output |
| `--force` | off | Delete existing output and checkpoint files before starting |
| `--checkpoint-every` | `500` | Save a recovery checkpoint every N chips. Jobs interrupted mid-run resume automatically from the last checkpoint when restarted with the same `--output` path |
| `--batch-size` | `8` | Chips per GPU batch |
| `--year` | `2022` | Used for OlmoEarth temporal encoding |
| `--test-chips` | off | Process only the first N chips (debug/smoke-test) |

**Output resolution:**
- OlmoEarth: 80 m/pixel (8-pixel patch × 10 m input)
- Prithvi: 480 m/pixel (16-pixel patch × 30 m input)

### Step 3 — Census-tract aggregation (`aggregate.py`)

Aggregates the embedding COG to census-tract level statistics (mean, median, max, min, std across all pixels within each tract boundary).

```bash
uv run python aggregate.py \
    --embedding outputs/embeddings/prithvi_tiny_RI_2022.tif \
    --tracts data/census_tracts_2020.gpkg \
    --output data/prithvi_tiny_embeddings.csv \
    --model prithvi --year 2022
```

Output CSV columns follow the schema `{BAND}_{STAT}` (e.g. `PR0000_MEAN`, `PR0001_MEDIAN`), matching the AlphaEarth embeddings format already in this repo.

---

## Analysis pipeline

Scripts live in `code/analyses/`. Run all commands from the project root.

### Preparing embeddings for analysis (`prepare_embeddings.py`)

The aggregation pipeline (Step 3 above) produces one CSV per state. This script
concatenates them into a single file and joins `ALAND`/`AWATER` from
`data/alphaearth_embeddings.csv` using lazy column selection (only those three
columns are read from the 2.3 GB file).

```bash
# Prithvi tiny (run once; output already exists at data/prithvi_tiny_2022_all_tracts.csv)
.venv/bin/python code/analyses/prepare_embeddings.py \
    --input-dir data/prithvi_aggregated/tiny \
    --output data/prithvi_tiny_2022_all_tracts.csv

# Prithvi 300M-TL (run after the GPU cluster finishes all states)
.venv/bin/python code/analyses/prepare_embeddings.py \
    --input-dir data/prithvi_aggregated/300M-TL \
    --output data/prithvi_300M-TL_2022_all_tracts.csv
```

Output columns: `GEOID`, `year`, embedding features (`PC00_MEAN` … `PC63_STD`
for tiny; `PR0000_MEAN` … `PR5119_STD` for 300M-TL), `ALAND`, `AWATER`.

### Full analyses notebook (`analyses.ipynb`)

Open `code/analyses/analyses.ipynb`. **Cell 0** is the only cell you need to edit — uncomment the two lines for the embedding source you want (AlphaEarth or Prithvi tiny), then run all cells. `OUTPUTS_DIR` and `EMBEDDINGS_PATH` propagate through the rest of the notebook automatically.

```python
# Prithvi tiny (uncomment both lines):
# EMBEDDINGS_PATH = Path("../../data/prithvi_tiny_2022_all_tracts.csv")
# OUTPUTS_DIR     = Path("../../outputs/prithvi_tiny")
```

### Predictive dependency analysis (`predictive_dependency.py`)

Fits LightGBM models (GroupKFold by state) for 20 ACS target variables, builds
a 20×20 cross-prediction R² matrix, runs mediation decomposition for 5 focal
pairs, and writes a heatmap and PCA biplot. Column prefix and ALAND/AWATER
presence are detected automatically from the input file.

```bash
# AlphaEarth (original)
.venv/bin/python code/analyses/predictive_dependency.py

# Prithvi tiny
.venv/bin/python code/analyses/predictive_dependency.py \
    --embeddings data/prithvi_tiny_2022_all_tracts.csv \
    --outputs-dir outputs/prithvi_tiny

# Prithvi 300M-TL (after prepare step above)
.venv/bin/python code/analyses/predictive_dependency.py \
    --embeddings data/prithvi_300M-TL_2022_all_tracts.csv \
    --outputs-dir outputs/prithvi_300M-TL
```

Add `--outputs-dir` so each model's results land in a separate folder and don't
overwrite each other. Set `USE_CACHED = False` at the top of the script to
refit from scratch rather than loading cached OOF predictions.

---

## To do

- [ ] Add TESSERA embeddings
