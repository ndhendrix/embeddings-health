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
