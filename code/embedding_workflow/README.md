# Embedding workflow

This isolated workflow supports `olmoearth-v1.2-nano`, `olmoearth-v1.2-base`,
and `clay-1.5` on the existing state Sentinel-2 composites. Inference produces
independently valid row tiles. Merging is strict, atomic, and non-destructive:
it validates schemas, grid alignment, overlap, and complete coverage before
publishing. Aggregation uses the established tract aggregator and an optional
national PCA (never independently fitted tile PCA).

`aggregate_tiles.py` is the bounded-memory production path. Each tile emits
per-tract count, sum, sum-of-squares, minimum, and maximum arrays. The reducer
combines them into exact mean/minimum/maximum/population-standard-deviation
features without opening a state mosaic. Exact median is intentionally absent
because it cannot be reduced from fixed-size sufficient statistics.
New workflow embedding tiles are band-interleaved, allowing merge and tract
workers to read bounded dimension groups without decompressing every embedding
dimension for each request. Existing pixel-interleaved tiles remain readable.

Submit the exact real-grid Rhode Island merge test from the repository root:

```bash
bash code/embedding_workflow/slurm/submit_ri_tests.sh
```

This submits both a fast exact synthetic merge on the real RI grid and a
bounded-memory test over the actual Nano tile embeddings, including tile-vs-
merged tract-statistic equivalence.

Submit a state/model production run:

```bash
MODEL=olmoearth-v1.2-nano STATE=RI NUM_TILES=4 \
  bash code/embedding_workflow/slurm/submit_state.sh
```

The scripts follow `code/embedding_generation/slurm/README.md`: explicit
Sherlock modules, scratch caches, GPU arrays, resumable inference, BigTIFF,
and dependency-driven merge jobs.

## Overlap/center-crop experiment

`generate_overlap.py` uses a stride equal to half the input chip width and
retains only the central half-width in each axis. Thus OLMoEarth uses 128/64
chip/stride with the central 64 pixels, while Clay uses 256/128 with the
central 128 pixels. Row tiles read their necessary halo implicitly through
boundless windows but own only half-open center regions. Outputs are versioned
under `embedding_workflow_overlap_v1` and never overwrite baseline products.

OlmoEarth overlap products created by the corrected path are tagged
`overlap-center50-v2`. Before encoder inference, the canonical 12 Sentinel-2
L2A bands are NaN-imputed and normalized using the OlmoEarth pretraining
computed mean +/- two standard deviations. The earlier unnormalized OlmoEarth
outputs must not be reused.

Prepare or submit the Rhode Island Nano experiment that mirrors the Clay
single-task versus 2x2 rectangular test:

```bash
bash code/embedding_workflow/slurm/submit_olmoearth_nano_ri_experiment.sh
CONFIRM_OE_NANO_RI_SUBMIT=1 \
  bash code/embedding_workflow/slurm/submit_olmoearth_nano_ri_experiment.sh
```

The default dry run submits nothing. A confirmed run writes to
`$SCRATCH/embeddings-health/olmoearth_nano_ri_experiment_v2`, checks raster and
tract-statistic equivalence, and reports embedding discontinuities at retained
chip boundaries relative to ordinary adjacent output pixels. It also renders a
Providence PCA RGB panel comparing the old unnormalized raster with the new
single-task and merged products.

## OlmoEarth Nano production waves

Plan the corrected overlap workflow across selected state composites without
submitting jobs:

```bash
SCRATCH=/scratch/users/nhendrix \
STATES="AL AR AZ" DRY_RUN=1 \
  bash code/embedding_workflow/slurm/submit_olmoearth_nano_overlap_resubmit.sh
```

Set `CONFIRM_OE_NANO_SUBMIT=1` to submit. The launcher defaults to batch 32 in
the model registry, rectangular tiles of at most roughly 20,000 retained
blocks, and 16 concurrent GPU tasks. Each wave must finish before its immutable
task manifest is rescanned, preventing active tasks from being duplicated or
remapped. State rasters are not merged; tract sufficient statistics are
computed tilewise and reduced exactly after all validated tiles exist.
Use `EXCLUDE_STATES` to defer a state whose source composite is under repair;
submit that state later with its repaired `COMPOSITE_DIR`.
## OlmoEarth Base production

RI experiments selected batch 16 and lossless 512x512 ZSTD source tiles. The
source layout reduced preload from 252.7 to 80.3 seconds and a
production-shaped tile from 45:12 to 18:51 (2.40x). The retiled source was
bit-exact over all 856,941,360 RI values, and Base embeddings were bit-exact at
fixed batch size. Batch-dependent GPU floating-point variation had maximum
absolute difference 0.000277. Center-crop seam-to-interior mean ratios were
1.0149 horizontally and 1.0102 vertically.

Prepare all sources and start bounded, resumable Base production waves:

```bash
SCRATCH=/scratch/users/nhendrix \
  bash code/embedding_workflow/slurm/submit_olmoearth_base_conus.sh
SCRATCH=/scratch/users/nhendrix CONFIRM_OE_BASE_CONUS_SUBMIT=1 \
  bash code/embedding_workflow/slurm/submit_olmoearth_base_conus.sh
```

The launcher losslessly retiles source composites before GPU work, uses the
canonical 128-pixel chips with stride 64 and retained central 64 pixels,
defaults to batch 16 and rectangular tiles of at most roughly 4,000 retained
blocks, and does not create state mosaics. Validated tiles are aggregated
through exact tilewise sufficient statistics. Repaired-source QA status is
recorded in the immutable retiling manifest; WY is included by an explicit
low-population-density coverage exception.
