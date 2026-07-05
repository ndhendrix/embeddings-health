#!/bin/bash
#SBATCH --job-name=oe-nano-merge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_nano_merge_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_nano_merge_%A_%a.err

# Merge tile embeddings into the final Nano embedding for one state.
# STATE_LIST and NUM_TILES_LIST (both colon-separated, same order/length)
# are set by submit_olmoearth_nano_embed_all_states.sh; SLURM_ARRAY_TASK_ID
# indexes into both. Only states with NUM_TILES > 1 are included in this
# array — single-tile states are already handled by the tile task itself.
#
# This job is submitted with --dependency=afterany:<tile-array-id> so it
# runs after all tile jobs complete. embed.py --merge-only exits cleanly
# (a SKIP message, not an error) if any tiles are still missing — the
# resubmit chain will retry those tiles and re-run this merge.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-Nano}"

: "${SCRATCH:?SCRATCH is not set}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_nano_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
: "${NUM_TILES_LIST:?NUM_TILES_LIST must be set to a colon-separated list matching STATE_LIST}"

IFS=: read -ra STATES <<< "$STATE_LIST"
IFS=: read -ra NUM_TILES_ARR <<< "$NUM_TILES_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"
NUM_TILES="${NUM_TILES_ARR[$TASK_ID]}"

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"
mkdir -p "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  [[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH" || { echo "ERROR: uv not found." >&2; exit 1; }
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

echo "Node:  $(hostname -s)"
echo "State: $STATE  (merge-only, $NUM_TILES tiles)"
echo ""

cd "$EMBED_DIR"
uv run --python 3.11 python embed.py \
  --model      olmoearth \
  --variant    "$VARIANT" \
  --output     "$CKPT_DIR/${OUTPUT_BASENAME}.tif" \
  --num-tiles  "$NUM_TILES" \
  --year       "$YEAR" \
  --merge-only

MERGED="$CKPT_DIR/${OUTPUT_BASENAME}.tif"
if [[ -s "$MERGED" ]]; then
  mv -f "$MERGED" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  echo "=== Merge complete: $STATE ==="
  ls -lh "$FINAL_DIR/${OUTPUT_BASENAME}.tif"
else
  echo "=== Merge skipped or failed for $STATE — tiles not yet complete ==="
fi
