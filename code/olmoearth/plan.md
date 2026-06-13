# OlmoEarth Studio Embedding Pull — Plan

## Goal

Pull 192-dimensional embeddings from the OlmoEarth Studio API at the state level,
PCA-reduce to 64 dimensions per pixel, and aggregate to census tract statistics
(mean / median / max / min / std per dimension). Output CSV matches the schema of
`data/alphaearth_embeddings.csv` for downstream model comparison.

---

## Architecture

One job per state (not per tract). The API handles internal tiling; the result is
a single COG covering the full state. Tract-level aggregation is done locally after
download.

```
code/olmoearth/
├── plan.md
├── .env                        (OLMOEARTH_API_KEY, OLMOEARTH_PROJECT_ID — gitignored)
├── config.json                 (project_id + model_id, written by 00_setup_model.py)
├── manifest.csv                (name, prediction_id, status — one row per state-year job)
├── 00_setup_model.py           (one-time: create Tiny model, write config.json)
├── 01_submit_jobs.py           (dissolve tracts → state boundary → submit one job)
├── 02_collect_results.py       (single-pass sweep: poll, download, mosaic, save COG)
├── 03_aggregate.py             (PCA fit → per-tract mask → stats → CSV)
├── 04_submit_conus.py          (CONUS batch: calls 01_submit_jobs.py for all 49 states)
├── 05_batch_aggregate.py       (CONUS batch: aggregate all state COGs, write CONUS CSV)
└── manifest_tract.csv          (archived: per-tract test run, RI 2022, 250 jobs)

data/olmoearth/
├── state_44_2022.tif           (state-level COG: 192 bands, int8, 40m, UTM)
└── ri/                         (archived: per-tract COGs from initial test run)
```

---

## Auth

`.env` (gitignored):
```
OLMOEARTH_API_KEY=...
OLMOEARTH_PROJECT_ID=673498a1-84ff-411e-88c3-535ef35fa44c
```

`config.json` written by `00_setup_model.py`:
```json
{ "project_id": "...", "model_id": "..." }
```

---

## Step 0 — One-time model setup (`00_setup_model.py`)

Already run. Model: OlmoEarth-v1-Tiny, `forty_meter`, 12 monthly periods, Sentinel-2 L2A.
Skip unless creating a new project.

---

## Step 1 — Submit (`01_submit_jobs.py`)

```bash
uv run python 01_submit_jobs.py                    # Rhode Island (default)
uv run python 01_submit_jobs.py --state 06 --year 2022  # California
```

- Downloads 2022 TIGER/Line tracts for the state if not cached.
- Dissolves tracts to derive the state boundary (MultiPolygon).
- Submits **one prediction job** with the full state boundary.
- Appends one row to `manifest.csv`.
- Idempotent: skips if the job name is already in the manifest.

---

## Step 2 — Collect (`02_collect_results.py`)

```bash
uv run python 02_collect_results.py
# Loop until all done:
until uv run python 02_collect_results.py | grep -q "All done"; do sleep 300; done
```

- Single-pass sweep: polls each submitted job once, downloads completed ones.
- If the ZIP contains multiple tiles, mosaics them with `rasterio.merge`.
- Writes `data/olmoearth/state_{fips}_{year}.tif` (192-band int8 COG).
- Updates manifest status to `done`.
- Resumable: re-running skips already-collected jobs.

---

## Step 3 — Aggregate (`03_aggregate.py`)

```bash
uv run python 03_aggregate.py                          # Rhode Island
uv run python 03_aggregate.py --state 06 --year 2022   # California
```

1. **PCA fit**: samples 2% of valid pixels from the state COG, fits `PCA(64)`.
2. **Per-tract aggregation**: for each tract, masks the COG to the tract boundary,
   applies the PCA transform, computes mean/median/max/min/std for each of 64 PCs.
3. **Output**: `data/olmoearth_studio_{fips}_{year}_embeddings.csv`
   — columns: `GEOID`, `year`, `PC00_MEAN`, ..., `PC63_STD` (322 columns total).

---

## Scaling to CONUS

```bash
# Submit all 49 CONUS states (48 contiguous + DC) — idempotent, safe to re-run
uv run python 04_submit_conus.py

# Poll until all jobs are downloaded
until uv run python 02_collect_results.py | grep -q "All done"; do sleep 300; done

# Aggregate each state COG and write one CONUS CSV
uv run python 05_batch_aggregate.py
# → data/olmoearth_studio_conus_2022_embeddings.csv
```

Total jobs: 49 (vs. ~85,000 with the per-tract approach).
RI job is already in manifest.csv and its CSV already exists; both scripts skip it.

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Job granularity | One per state | Matches intended API usage; ~50 emails vs. ~85,000 |
| Tile mosaicking | `rasterio.merge` in collect script | Handles variable tile count transparently |
| PCA fitting | Sample 2% of state COG pixels | Memory-safe; reproducible (fixed seed) |
| PCA application | Per-tract, on masked pixels | Avoids writing a full transformed state COG |
| Output format | CSV matching AlphaEarth schema | Drop-in for downstream models |
| Resumability | Manifest CSV checkpoint | Re-run collect and aggregate safely |
