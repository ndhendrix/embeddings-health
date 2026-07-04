# Sub-tiling the embedding pipeline (Base / Nano / Clay)

## Problem

28 of ~44 states are stuck in a Slurm TIMEOUT retry loop across the OlmoEarth
Base (`oe-embed`), OlmoEarth Nano (`oe-nano-embed`), and Clay (`clay-embed`)
array jobs (see `[[scratch-staging-leak]]` memory for the disk-leak side of
this, already fixed separately). Two distinct causes, both rooted in
processing a whole state as one un-splittable unit of work:

1. **Base model is compute-bound.** Real throughput (tqdm counts *batches*,
   not chips — confirmed via `initial=n_skip // batch_size` in `embed.py`) is
   ~24s per 32-chip batch, i.e. ~0.75s/chip. Large states need tens of hours
   of continuous inference — more than any walltime tier can provide in one
   sitting, and chip-level checkpointing only helps *within* a job, not
   across the many job resubmissions actually required.
2. **Nano/Clay are setup-bound.** Composites are 100s of GB; before the
   staging fix, copying one to disk (plus allocating the disk-backed output
   memmap) ate nearly the entire walltime, leaving ~10 minutes/retry for
   actual inference — near-zero net progress per attempt.

The staging-leak fix (2026-07-04, see memory) stopped the disk bleed but does
not fix either root cause above — states still can't finish.

## Goal

Let every state finish within existing walltime tiers (no ability to request
longer walltime on this cluster) by splitting each state's embedding work
into independently-schedulable, independently-resumable tiles, mirroring the
tiling pattern `composite.py` already uses for composite generation
(`split_bbox_into_tiles` → per-tile job → `_merge_tiles`).

**Scope:** Base, Nano, and Clay pipelines (all three key off the same
`s2_annual_<STATE>_<YEAR>_olmoearth.tif` composite format and share the same
sbatch structure). Prithvi is out of scope (separate file naming, mostly
already complete, not observed in the stuck-state list). Fixing the stuck
TX/CA *composite* merge (separate bug in `composite.py`) is explicitly out of
scope for this effort.

## Architecture

- `embed.py` gains `--tile-index i --num-tiles n` (default `0`/`1` = today's
  behavior, unchanged for states that don't need splitting).
- Given `num_tiles`, `embed.py` derives which rows/cols of the chip grid this
  task owns purely from the composite's own pixel dimensions + chip size —
  no external geometry/CRS math needed, unlike `composite.py`'s bbox tiling.
- Chips are read via plain rasterio **windowed reads directly against the
  composite file on `$SCRATCH`** — no local copy, no staging directory, no
  `$L_SCRATCH`. Composites are already COG-structured (tiled, block-organized
  GeoTIFFs), so windowed reads of one tile's pixels don't require touching
  the whole file. This removes the setup-copy bottleneck (fixes Nano/Clay)
  and, combined with bounded per-tile chip counts, lets Base finish within a
  walltime tier (fixes Base) — and removes the staging-leak risk for this
  pipeline altogether, since there's nothing large left to copy.
- A merge step (adapted from `composite.py`'s `_merge_tiles`) mosaics a
  state's finished tile outputs into the one final per-state file
  `aggregate.py` already expects. No downstream (aggregate/figures) changes.

## Components

1. **`embed.py`**
   - New args: `--tile-index` (int, default 0), `--num-tiles` (int, default 1).
   - New helper `tile_row_col_bounds(n_rows_chips, n_cols_chips, num_tiles) ->
     (row_start, row_end, col_start, col_end)`: splits the chip grid (not the
     geographic bbox) into `num_tiles` roughly-square, roughly-equal chunks.
     Simpler than `composite.py`'s version since it operates on integer chip
     indices, not projected coordinates.
   - `iter_chips` takes an optional bounds argument restricting which
     (row_off, col_off) it yields.
   - Output filename gets a `_tile###` suffix when `num_tiles > 1`; the tile
     raster is a standalone valid georeferenced file (correct geotransform
     offset for its sub-window), matching how composite tiles work today.

2. **`plan_tiles(composite_path, target_chips_per_tile) -> int`** (new
   helper, likely added to `embed.py` or a small shared module) — opens the
   composite header only (no pixel data read) to get dimensions, computes
   total chip count, returns `ceil(total_chips / target_chips_per_tile)`.
   One `target_chips_per_tile` constant per pipeline (see "Starting tile
   sizes" below) — states whose total chip count is already under the target
   get `num_tiles=1`, i.e. no behavior change for the ~16 states that
   complete fine today.

3. **Submit scripts** (per pipeline) — instead of one array task per state,
   build a flat list of `(state, tile_index, num_tiles)` triples across all
   states via `plan_tiles()`, submit one array over that flat list, then a
   dependent merge array (`--dependency=afterany:<embed_array_id>`) with one
   task per state that has `num_tiles > 1`.

4. **`run_*_state_array.sbatch` scripts** — staging block removed entirely
   for the tiled path (no `$L_SCRATCH`/`$SCRATCH/staging` decision needed —
   nothing to stage). Task resolves its `(state, tile_index, num_tiles)` from
   the array index same as today's state resolution, then calls `embed.py`
   with `--input "$COMPOSITE" --tile-index "$TILE_INDEX" --num-tiles
   "$NUM_TILES"`. Checkpoint dir stays `checkpoints/<pipeline>/<state>/` (no
   new subdirectory level); individual checkpoint/output filenames get a
   `_tile<N>` suffix (matching `composite.py`'s existing flat
   `_tile###.tif` convention exactly) so tiles don't clobber each other:
   `<basename>_tile<N>.tif`, `<basename>_tile<N>.ckpt.n`, etc. The
   `--signal=B:TERM@600` / background+`wait` cleanup-trap fix stays (still
   useful for the checkpoint/output files), but `cleanup_stage` becomes a
   no-op / can be deleted since there's no staging dir anymore.

5. **New merge sbatch** (one per pipeline, or one generic script
   parameterized by pipeline) — adapts `composite.py`'s `_merge_tiles`:
   verifies all of a state's tile outputs exist (else exits with a warning,
   matching `composite.py`'s `merge_only` behavior), mosaics via windowed
   writes into `<pipeline>_embeddings/<state>/<file>.tif`, deletes tile files
   after a successful merge, with its own resumable `.merge_ckpt` (same
   mechanism `composite.py` already uses).

## Data flow (example: NV under Base, `num_tiles=6`)

1. Submit script's `plan_tiles()` returns 6 → appends `(NV,0,6)...(NV,5,6)` to
   the flat task list.
2. Six independent array tasks run (any node/time, no shared state). Each
   opens the NV composite read-only from `$SCRATCH`, computes its own bounds
   from `--tile-index`/`--num-tiles`, iterates only its chips, checkpoints
   periodically to `checkpoints/oe-embed/NV/olmoearth_v1_1-Base_NV_2022_tile0.ckpt.n`,
   writes `checkpoints/oe-embed/NV/olmoearth_v1_1-Base_NV_2022_tile0.tif`
   there too until merge picks it up.
3. On timeout/failure: existing trap-based cleanup runs (now simpler — no
   staging to clean). Chip-level checkpoint means retry resumes mid-tile with
   a small remaining chip count, not from scratch across a multi-hour setup.
4. Once all 6 tile outputs exist, the merge task for NV runs, mosaics into
   the final file, deletes the 6 tile files. If the merge itself times out
   (plausible for the largest states), it resumes via `.merge_ckpt`.
5. A tile that's still too slow for its walltime tier just needs a smaller
   `target_chips_per_tile` re-plan for that one state — not a full-state redo.

## Starting tile sizes (to validate/tune during testing, not final)

| Pipeline | Observed/assumed rate | Walltime tier | Target chips/tile (starting point) |
|---|---|---|---|
| Base (`oe-embed`) | ~0.75s/chip (measured, NY) | 4h | ~7,000 (≈90 min inference + margin) |
| Nano (`oe-nano-embed`) | ~50 chips/sec (conservative; setup bottleneck removal likely raises this) | 2h | ~150,000 (≈60 min inference + margin) |
| Clay (`clay-embed`) | Not measured yet | 4h | ~20,000 (placeholder — measure during srun testing, adjust before batch submission) |

## Testing plan

1. Unit-level: `tile_row_col_bounds()` and `plan_tiles()` are pure functions
   — test tile partitions have no gaps/overlaps and small states get
   `num_tiles=1`.
2. `srun` smoke test on a small already-completed state (e.g. DC or RI)
   forced to `--num-tiles 3`: verify each tile writes a correctly-bounded
   sub-raster, and that merging reproduces today's single-shot output
   (compare against existing DC/RI final embeddings).
3. `srun` end-to-end test on one real currently-stuck large state (Nano
   first — fastest to iterate): a few tiles, confirm each finishes well
   within walltime, confirm kill-and-resume works per-tile, confirm merge
   produces a valid final file.
4. Only after 2–3 pass, submit one pipeline's real batch array at a time
   (Nano → Clay → Base), not all three simultaneously.

## Out of scope

- TX/CA's stuck composite merge (separate bug in `composite.py`).
- Prithvi pipeline (not observed in the stuck-state list, mostly complete).
- Batched-inference/GPU-utilization tuning (bigger `--batch-size`, mixed
  precision, etc.) — complementary future optimization, doesn't by itself
  fix either root cause (a faster per-chip rate still can't make a
  single-job-per-state finish in bounded walltime for the largest states,
  and doesn't touch the setup-bound Nano/Clay case at all). Worth revisiting
  once tiling is in place and each tile's bottleneck is purely GPU compute.
