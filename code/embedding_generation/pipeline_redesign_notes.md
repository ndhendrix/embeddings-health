# Embedding Pipeline Redesign Notes

These notes are design-level only. They do not prescribe an immediate rewrite;
they describe a cleaner target architecture for producing census-tract
embeddings from satellite imagery on Sherlock or a similar HPC cluster.

## Current Strengths

- The pipeline already separates the major stages: Sentinel-2 compositing,
  model inference, tile/state merging, PCA, and tract aggregation.
- Large-state compositing can run per tile, which is the right direction for
  memory control and Slurm parallelism.
- Embedding inference is checkpointed and can resume after walltime limits.
- Jobs stage large inputs to node-local storage when possible.
- Merge operations are increasingly resumable and avoid full-scene in-memory
  mosaics.

The main weaknesses are not algorithmic. They are mostly orchestration and
provenance issues: pipeline state is inferred from filenames, work discovery is
spread across shell scripts, and success means "a file exists" rather than "this
declared task completed with these exact inputs and parameters."

## Recommended Target Shape

Move to a manifest-driven DAG with explicit task records:

1. `discover`
   Build a versioned manifest of state/model/year/tile/season tasks.
2. `composite_tile`
   Produce one composite tile per state/tile/season or annual window.
3. `embed_tile`
   Produce one embedding tile directly from the composite tile.
4. `aggregate_tile_to_tract`
   Compute partial tract statistics from each embedding tile.
5. `reduce_tract_stats`
   Combine tile-level partial stats into final census-tract rows.

The major rethink is to avoid merging huge embedding rasters when the endpoint
is tract-level statistics. Full state embedding COGs can remain an optional
artifact for validation and visualization, not the required path to analysis.

## Manifest and Provenance

Create a small SQLite database or Parquet manifest under `$SCRATCH`, with one
row per task and stable content-derived IDs.

Suggested fields:

- `task_id`, `stage`, `state`, `year`, `model`, `variant`, `tile_id`, `season`
- `bbox_wgs84`, `crs`, `resolution`, `tile_grid_version`
- `input_paths`, `output_paths`
- `parameters_json`
- `code_git_sha`, `uv_lock_hash`, `model_repo`, `model_revision`
- `status`, `attempt`, `slurm_job_id`, `started_at`, `finished_at`
- `input_checksum`, `output_checksum`, `validation_json`

This makes retries deterministic and auditable. Submit scripts should query the
manifest for `status != complete`, not rediscover work from glob patterns.

## Efficiency Recommendations

### 1. Aggregate by tile instead of merging embeddings first

For the analysis endpoint, a state-level embedding COG is an expensive
intermediate. The more scalable path is:

- Build a spatial index of census tracts intersecting each tile.
- For each embedding tile, compute per-tract partial sufficient statistics.
- Reduce partials across tiles into final tract-level means, mins, maxes, and
  standard deviations.

Mean, min, max, and standard deviation are easy to reduce exactly from partials
using counts, sums, sums of squares, minima, and maxima. Median is not exactly
reducible unless all pixels are retained or a sketch is used. Options:

- Drop median for model-development runs.
- Use approximate quantile sketches such as t-digest.
- Keep exact median only in a slower final pass.

This avoids writing, reading, and merging multi-hundred-band state rasters for
large states. It also makes failed tiles cheap to retry.

### 2. Use a fixed projected tile grid

The current state-bbox tiling is pragmatic, but a reproducible product should
use a stable tile grid independent of state boundaries.

Recommended grid:

- Per UTM zone, fixed 100 km or 110 km tiles aligned to integer projected
  coordinates.
- Each tile has a durable `tile_id`.
- States are just sets of tile intersections.

Benefits:

- Adjacent states can share composite/inference tiles.
- No duplicate work at state borders.
- Easier caching across models and years.
- Cleaner restart semantics.

### 3. Cache STAC search results

STAC queries are a source of nondeterminism and latency. For each tile/window,
write the selected item IDs and asset hrefs to the manifest or a sidecar JSON.

Then compositing becomes:

- Query once in `discover` or `select_scenes`.
- Freeze the selected scenes.
- Reuse the same item list for retries, model variants, and reproducibility.

Include scene filtering parameters in the task hash: max cloud cover,
max-scenes-per-month, season/year window, collection, and STAC endpoint.

### 4. Composite once, feed multiple models

Where possible, create one canonical Sentinel-2 composite product and derive
model-specific inputs from it. For example, a 12-band 10 m annual or seasonal
tile can feed OlmoEarth and Clay directly, and can be resampled/band-selected
for Prithvi.

This trades extra storage for less repeated STAC loading and cloud masking,
which are expensive and network-sensitive on HPC.

Use two temporal forms of the canonical 12-band, 10 m product:

- **Annual 10 m composites** remain the default preservation master for
  OlmoEarth, Clay, and annual fine-tuning experiments.
- **Spring, summer, and fall 10 m composites** are retained for 2022 and other
  selected benchmark years. A Prithvi-compatible 30 m, six-band input can be
  derived from them by band selection followed by grid-aligned, area-weighted
  3x3 aggregation. Store the aggregation rule and valid-area threshold in the
  manifest.

The seasonal product is optional for future production years rather than an
unconditional expansion of the archive. Prithvi has performed less well than
the other models in this project and may not remain part of future work. The
10 m seasonal composites are still useful for phenology-aware fine-tuning, but
new years should be produced only when that use or a Prithvi comparison is
planned. Preserve the existing 30 m Prithvi seasonal inputs until their 10 m
replacements have been generated and validated.

Spatially aggregating a finished 10 m temporal median is not exactly equivalent
to loading observations at 30 m and then taking their temporal median. Treat the
derived 30 m raster as a versioned model input and validate it against the
current direct-30 m pipeline before replacing the existing inputs.

### 5. Separate raw arrays from COG publication

For intermediate HPC artifacts, a chunked array format may be faster and more
robust than GeoTIFF:

- Zarr for composites and embeddings while computing.
- COG only for publishable rasters or external GIS inspection.

Zarr advantages:

- Chunk-aligned writes and retries.
- Natural cloud/HPC object layout.
- Easier partial reads for tract aggregation.

COG remains valuable for final map artifacts, but it does not need to be the
primary internal exchange format.

### 6. Reduce environment setup cost

The current jobs may spend significant time building or resolving environments.
For production runs, prefer a prebuilt Apptainer/Singularity image or a frozen
shared uv environment created by a dedicated setup job.

At minimum:

- Build environment once per code/lockfile hash.
- Store the environment path in the manifest.
- Fail fast if the runtime hash differs from the task hash.

This improves reproducibility and reduces per-job walltime variance.

## Long-Term Archive Representation

The long-term archive should preserve the source information needed for future
fine-tuning, while keeping only a compact record of the current frozen
embeddings. Raw CONUS float32 embedding rasters are not the preservation master:
they can be regenerated from the composites, exact model snapshot, and frozen
workflow if a future project needs a different representation.

### Preservation masters

Keep lossless copies of:

- canonical annual 10 m Sentinel-2 composites;
- the selected-year seasonal 10 m composites described above;
- exact model weights, configuration, model revision, code commit, and runtime
  lockfiles or container;
- frozen STAC item selections and compositing parameters;
- tract tables, validation reports, census boundaries, and fitted PCA and
  quantizer artifacts.

Do not archive caches, checkpoints, temporary rasters, superseded repair trees,
or quarantined outputs once their canonical replacements have two verified
archive copies.

### Compact embedding record: national PCA64 followed by int8

For each model and workflow version, use this archive transform:

    raw float32 embedding
      -> one fixed national PCA transform
      -> 64 float32 principal components
      -> one fixed per-component int8 quantizer
      -> tiled GeoTIFF with lossless internal compression

The PCA must be national, not independently fitted by state or tile. Use one
versioned transform everywhere that embeddings need to remain comparable. The
current PCA code already samples states approximately in proportion to raster
area and fits 64 components with a fixed seed. For future longitudinal products,
either continue applying the frozen reference PCA or fit a separately versioned
multi-year PCA; never silently refit the transform for each year.

Reserve -128 for nodata. For valid pixels, calibrate one global scale per
principal component and map values to [-127, 127]. Do not use per-tile or
per-state scales, because those make the stored integer values incomparable and
complicate mosaicking. Store the PCA mean/components, quantizer method, scale,
clipping bounds, calibration sample checksum, and inverse-transform formula in
both GeoTIFF metadata and a sidecar JSON.

Benchmark a symmetric linear quantizer and a nonlinear companding quantizer on
held-out states. AlphaEarth is a useful precedent for 64-dimensional int8
embedding fields, but its published power-law quantizer is tailored to its
unit-normalized vectors and should not be assumed optimal for Clay or OlmoEarth
principal components.

The theoretical payload reductions relative to the current raw float32 vectors
are:

| Model | Raw dimensions | PCA64 float32 | PCA64 int8 |
|---|---:|---:|---:|
| Clay 1.5 | 1024 | 16x smaller | 64x smaller |
| OlmoEarth Base | 768 | 12x smaller | 48x smaller |
| OlmoEarth Nano | 128 | 2x smaller | 8x smaller |
| Prithvi tiny | 192 | 3x smaller | 12x smaller |
| Prithvi 300M | 1024 | 16x smaller | 64x smaller |

Actual GeoTIFF ratios will differ because of nodata, tiling, and lossless codec
behavior. Rewrite archive rasters cleanly so that every TIFF block is written
once; otherwise superseded partial-block payloads can dominate file size.

### Validation gate and fallback

PCA is expected to account for most of the information loss. The latest Clay
national PCA64 fit retained 86.7% of sampled variance, so the archive must record
that value and not describe PCA64 as lossless. Int8 should be evaluated as the
additional loss after PCA, not conflated with the PCA reduction.

Before deleting a raw model collection, validate PCA64-int8 on held-out states:

- clipping/saturation rate by principal component;
- component-wise error and correlation after dequantization;
- cosine similarity and RMSE after inverse-PCA reconstruction;
- differences in tract mean and standard-deviation features;
- change in held-out-state health-model performance;
- exact preservation of CRS, transform, coverage, and nodata mask.

Use PCA64-int8 as the archive representation if the additional int8 error is
negligible relative to the PCA error and downstream conclusions are unchanged.
If int8 fails that gate, fall back to PCA64-int16; if necessary, retain PCA64
float32, which still reduces a 1024-dimensional model by 16x.

Retain the national PCA calibration sample and one complete small-state raw
float32 raster per model as audit fixtures. These are inexpensive compared with
the CONUS collection and allow future readers to verify the transform and
quantizer without restoring or regenerating all raw embeddings.

## Scheduling Model

Use Slurm job arrays as workers over manifest task IDs:

- `submit_stage.py --stage composite_tile --limit 1000`
- `submit_stage.py --stage embed_tile --model prithvi --variant 300M-TL`
- `submit_stage.py --stage aggregate_tile_to_tract`
- `submit_stage.py --stage reduce_tract_stats`

Each worker receives a `task_id`, loads exactly one manifest row, claims it with
an atomic status update, writes outputs to temporary paths, validates them, then
marks the task complete.

This removes fragile array-index mappings like "state index times number of
models" and makes retrying failed work a manifest query.

## Validation Gates

Each stage should write a small validation record before being marked complete.

Composite validation:

- CRS, transform, dimensions, band count, dtype
- NaN fraction by band
- selected scene count
- valid-pixel fraction

Embedding validation:

- expected embedding dimension
- expected output resolution
- finite-pixel fraction
- per-band summary statistics
- model revision and code hash

Aggregation validation:

- number of tracts touched
- pixel counts per tract
- missing tract list
- feature column count

These records are small, but they make debugging failed states much easier.

## Reproducibility Priorities

The most important items to freeze are:

- Git commit SHA
- `uv.lock` hash
- model repository revision or exact snapshot hash
- STAC item IDs and asset hrefs
- tile grid version
- composite parameters
- PCA model checksum
- tract boundary file checksum

Without these, the same state/year/model label can silently mean different
inputs or different model weights.

## Practical Migration Path

1. Add a manifest generator without changing the computational code.
2. Modify submit scripts to read task IDs from the manifest instead of globbing.
3. Add validation JSON sidecars for composites and embeddings.
4. Add tile-level tract aggregation for mean/min/max/std.
5. Decide whether exact median is required.
6. Make state-level embedding COG merge optional.
7. Consider Zarr for internal artifacts once the manifest flow is stable.

The highest-impact change is tile-level tract aggregation. It directly targets
the endpoint of the project and avoids spending cluster time on very large
embedding mosaics that are not necessary for the regression analyses.
