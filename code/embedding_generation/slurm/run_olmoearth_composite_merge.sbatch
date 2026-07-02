#!/bin/bash
#SBATCH --job-name=oe-merge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_merge_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_merge_%A_%a.err

# Merge tile TIFs into the final composite for one state.
# STATE_LIST (colon-separated) is set by the submit script;
# SLURM_ARRAY_TASK_ID indexes into it.
# This job is submitted with --dependency=afterany:<tile-array-id> so it runs
# after all tile jobs complete. composite.py --merge-only exits cleanly if any
# tiles are still missing (they'll be retried by the fallback resubmit chain).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
OUT_DIR="${OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"

IFS=: read -ra STATES <<< "$STATE_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export GDAL_CACHEMAX=4096
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

echo "Node:  $(hostname -s)"
echo "State: $STATE  (merge-only)"
echo ""

cd "$EMBED_DIR"
PYTHONUNBUFFERED=1 uv run --python 3.11 python composite.py \
  --state        "$STATE" \
  --model        olmoearth \
  --year         "$YEAR" \
  --output-dir   "$OUT_DIR" \
  --max-tile-km  100 \
  --merge-only

echo ""
FINAL="$OUT_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
if [[ -s "$FINAL" ]]; then
  echo "=== Merge complete: $STATE ==="
  ls -lh "$FINAL"
else
  echo "=== Merge skipped or failed for $STATE — tiles not yet complete ==="
fi
