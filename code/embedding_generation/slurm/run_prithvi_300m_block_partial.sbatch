#!/bin/bash
#SBATCH --job-name=prithvi-300m-block
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${MANIFEST:?MANIFEST must point to the block-task TSV}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PYTHON="${PYTHON:-$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python}"
line_number=$((SLURM_ARRAY_TASK_ID + 1))
line=$(sed -n "${line_number}p" "$MANIFEST")
[[ -n "$line" ]] || {
  echo "ERROR: no manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
}

IFS=$'\t' read -r STATE EMBEDDING TRACTS OUTPUT BLOCK_ROW BLOCK_COL <<< "$line"

if [[ -s "$OUTPUT" ]]; then
  echo "SKIP existing atomic partial: $OUTPUT"
  exit 0
fi

echo "State: $STATE"
echo "Block: row=$BLOCK_ROW col=$BLOCK_COL"
echo "Input: $EMBEDDING"
echo "Output: $OUTPUT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
"$PYTHON" "$REPO_DIR/code/embedding_generation/aggregate_raster_blocks.py" \
  --embedding "$EMBEDDING" \
  --tracts "$TRACTS" \
  --output "$OUTPUT" \
  --block-row "$BLOCK_ROW" \
  --block-col "$BLOCK_COL" \
  --expected-dim 1024 \
  --band-chunk 64

echo "Complete: $OUTPUT"
