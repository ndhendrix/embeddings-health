#!/bin/bash
#SBATCH --job-name=oe-composite
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_composite_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_composite_%A_%a.err

# STATE_LIST must be set by the submit script (colon-separated sorted state abbrevs).
# SLURM_ARRAY_TASK_ID indexes into it.
#
# Partial progress is preserved automatically: composite.py skips tile TIFs that
# already exist and skips states whose final TIF is already present. Re-submitting
# the same array task safely resumes where the previous run left off.

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

if (( TASK_ID < 0 || TASK_ID >= ${#STATES[@]} )); then
  echo "ERROR: TASK_ID=$TASK_ID out of range (${#STATES[@]} states)" >&2
  exit 1
fi

STATE="${STATES[$TASK_ID]}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
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
    echo "ERROR: uv not found. Run an embedding job first to bootstrap it, or set UV_INSTALL_DIR." >&2
    exit 1
  fi
fi

echo "uv: $(command -v uv)  $(uv --version)"
echo "gcc: $(gcc --version | head -n 1)"
echo "=== OlmoEarth Sentinel-2 Composite ==="
echo "State:      $STATE"
echo "Year:       $YEAR"
echo "Output dir: $OUT_DIR"
echo "Task ID:    $TASK_ID"
echo ""

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

cd "$EMBED_DIR"

uv run --python 3.11 python composite.py \
  --state        "$STATE" \
  --model        olmoearth \
  --year         "$YEAR" \
  --output-dir   "$OUT_DIR" \
  --max-tile-km  100 \
  $FORCE_FLAG

echo ""
echo "=== Done ==="
EXPECTED="$OUT_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
if [[ -s "$EXPECTED" ]]; then
  ls -lh "$EXPECTED"
else
  echo "WARNING: expected output not found — composite may need another run: $EXPECTED"
fi
