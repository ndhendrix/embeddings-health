# Reassembling overlap embedding tiles into statewide GeoTIFFs

This document describes how to reconstruct a statewide embedding GeoTIFF from
the independently generated `overlap-center50-v1` tiles. Statewide GeoTIFFs are
not produced by the current workflow, but the tiles are designed to support
lossless future assembly.

## What must be preserved

Keep all of the following for each state:

- Every `*_tileNNN.tif` file
- The corresponding `*_tileNNN.validation.json` files
- This repository, or at minimum `code/embedding_workflow/merge.py`

The GeoTIFFs themselves contain the information required for assembly: CRS,
affine transform, bounds, resolution, band count, nodata value, model
provenance, source composite, workflow version, and tile-grid dimensions. The
validation JSON files make it easier to audit completeness and provenance
without opening the large rasters.

Tile numbers do not determine spatial placement. `merge.py` places each tile
using its embedded georeferencing, so input order is immaterial. Tile numbers
are still useful for detecting missing files.

## Requirements

Use the repository environment containing `rasterio` and its GDAL
dependencies. On Sherlock, the existing CPU environment can be used:

```bash
export SCRATCH=/scratch/users/$USER
export REPO_DIR=$HOME/embeddings-health
export UV_PROJECT_ENVIRONMENT=$SCRATCH/embeddings-health/cache/venv-3.11-cpu
cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
```

No GPU is required. The merge is an I/O-heavy CPU operation.

For large states, use a node with at least 24 GB RAM, ample temporary and
destination storage, and an eight-hour or longer wall-time allocation. The
result is a 1,024-band float32 BigTIFF and may be very large even with DEFLATE
compression. Check available space before starting.

## 1. Select one state and verify completeness

Set the product root, model, state, and year:

```bash
ROOT="$SCRATCH/embeddings-health/embedding_workflow_overlap_v1"
MODEL="clay-1.5"
STATE="CT"
YEAR="2022"
DIR="$ROOT/$MODEL/$STATE"
STEM="${MODEL}_overlap-center50_${STATE}_${YEAR}"
```

Build a naturally sorted list of tiles:

```bash
mapfile -t TILES < <(
  find "$DIR" -maxdepth 1 -type f -name "${STEM}_tile[0-9][0-9][0-9].tif" |
  sort
)
printf 'Found %d tiles\n' "${#TILES[@]}"
```

Read the expected grid from a validation sidecar:

```bash
GRID=$(jq -r '.tags.tile_grid' "${TILES[0]%.tif}.validation.json")
ROWS=${GRID%x*}
COLS=${GRID#*x}
EXPECTED=$((ROWS * COLS))
printf 'Grid: %s; expected: %d; found: %d\n' \
  "$GRID" "$EXPECTED" "${#TILES[@]}"
test "${#TILES[@]}" -eq "$EXPECTED"
```

Also verify that every raster has a validation sidecar:

```bash
for tile in "${TILES[@]}"; do
  test -s "${tile%.tif}.validation.json" ||
    { echo "Missing validation: ${tile%.tif}.validation.json" >&2; exit 1; }
done
```

The strict merge performs the authoritative spatial checks. It refuses to
write a final output if tiles have gaps, overlaps, inconsistent CRS, band
count, data type or resolution, or if they are not aligned to a common grid.

States whose grid is `1x1` may have been written directly as
`${STEM}.tif`, without a `_tile000` suffix. Such a file is already the
statewide product and does not need merging.

## 2. Merge

Choose an output path that does not match the tile filename pattern:

```bash
OUTPUT="$DIR/${STEM}.tif"
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"

uv run --python 3.11 python \
  ../embedding_workflow/merge.py \
  --tiles "${TILES[@]}" \
  --output "$OUTPUT"
```

`merge.py` writes to a hidden partial file in the destination directory and
atomically renames it only after a successful merge. Existing source tiles are
never modified or deleted. If a job is interrupted, remove the hidden
`.${STEM}.tif.partial` file before retrying; do not remove any tile.

The default merge reads 16 bands at a time. If memory is constrained, reduce
the chunk:

```bash
uv run --python 3.11 python \
  ../embedding_workflow/merge.py \
  --band-chunk 8 \
  --tiles "${TILES[@]}" \
  --output "$OUTPUT"
```

## 3. Validate the statewide result

Run the same structural and numerical validator used for individual tiles:

```bash
uv run --python 3.11 python \
  ../embedding_workflow/validate_overlap.py \
  --raster "$OUTPUT" \
  --model "$MODEL"
```

This creates `${OUTPUT%.tif}.validation.json`. Retain it with the statewide
GeoTIFF.

As a final audit, confirm that:

- The merge and validation commands exited successfully.
- Both `$OUTPUT` and `${OUTPUT%.tif}.validation.json` are nonempty.
- The statewide bounds equal the union of the tile bounds.
- Model revision, workflow, year, resolution, band count, and source composite
  are consistent with the tile sidecars.

Do not delete the source tiles immediately after merging. Keep them until the
statewide file has been copied to durable storage, checksummed, and tested by
the intended downstream reader.

## Slurm template

The current `merge_state.sbatch` belongs to an older filename convention and
should not be used unchanged for `overlap-center50` products. A future Slurm
job can run the commands above on the `normal` partition with approximately:

```text
CPUs:       2
Memory:     24–32 GB
Time:       8–24 hours, depending on state and storage throughput
GPU:        none
GDAL cache: 512 MB
```

Always perform the expected-tile-count check before submission. A dependency
on inference jobs is convenient but is not a substitute for checking the
files, because jobs can be retried or outputs can be moved independently.

## Format and provenance notes

The current Clay products have these expected properties:

```text
model:                    clay-1.5
model revision:           70200ebcccdf67bf2a0cb9984c77ddee26c10ed2
workflow:                 overlap-center50-v1
bands:                    1024 float32 embedding dimensions
output resolution:        80 m
nodata:                   NaN
ownership:                half-open, non-overlapping rectangles
source edge handling:     boundless NaN-imputed context
```

Treat the embedded tags as authoritative if a later workflow version or model
differs from these values. Do not combine tiles with different models, model
revisions, workflow versions, years, source composites, band schemas, CRS, or
resolution. The merge checks the spatial schema, while the validation
sidecars and embedded tags provide the provenance checks.

