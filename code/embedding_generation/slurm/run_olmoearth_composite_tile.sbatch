#!/bin/bash
#SBATCH --job-name=oe-tile
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_tile_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_tile_%A_%a.err

# Per-tile composite job. SLURM_ARRAY_TASK_ID indexes into TILE_TASK_FILE,
# a newline-delimited file with lines of the form:
#   STATE TILE_IDX
# Created by submit_olmoearth_composite_parallel.sh.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
OUT_DIR="${OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${TILE_TASK_FILE:?TILE_TASK_FILE must be set to the task list file path}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "ERROR: uv not found." >&2; exit 1
  fi
fi

# Read this task's (STATE, TILE_IDX) from the task file (1-indexed by Slurm)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TILE_TASK_FILE")
STATE=$(echo "$LINE" | awk '{print $1}')
TILE_IDX=$(echo "$LINE" | awk '{print $2}')

echo "Node:       $(hostname -s)"
echo "Task ID:    $SLURM_ARRAY_TASK_ID  →  $STATE tile $TILE_IDX"
echo "Output dir: $OUT_DIR"
echo ""

cd "$EMBED_DIR"
uv run --python 3.11 python composite.py \
  --state          "$STATE" \
  --model          olmoearth \
  --year           "$YEAR" \
  --output-dir     "$OUT_DIR" \
  --max-tile-km    100 \
  --max-cloud-cover "${MAX_CLOUD_COVER:-30}" \
  --max-scenes-per-month "${MAX_SCENES_PER_MONTH:-2}" \
  --tile-index     "$TILE_IDX"

echo ""
echo "=== Done: $STATE tile $TILE_IDX ==="
TILE_PATH="$OUT_DIR/s2_annual_${STATE}_${YEAR}_olmoearth_tile$(printf '%03d' "$TILE_IDX").tif"
[[ -s "$TILE_PATH" ]] && ls -lh "$TILE_PATH" || echo "WARNING: tile file not found: $TILE_PATH"
