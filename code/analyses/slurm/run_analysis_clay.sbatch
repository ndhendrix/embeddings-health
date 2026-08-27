#!/bin/bash
#SBATCH --job-name=emb-health-clay
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=code/analyses/slurm/logs/clay_%j.out
#SBATCH --error=code/analyses/slurm/logs/clay_%j.err

# Run from project root:
#   sbatch code/analyses/slurm/run_analysis_clay.sbatch
# Mirrors run_analysis_prithvi_tiny.sbatch.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
ANALYSIS_MODEL="${ANALYSIS_MODEL:-clay}"
AGG_DIR="${AGG_DIR:-$SCRATCH/embeddings-health/clay_aggregated}"
COMBINED="${COMBINED:-$AGG_DIR/clay_v1.5_2022_all_tracts.csv}"
AREA_SOURCE="$SCRATCH/embeddings-health/alphaearth/alphaearth_embeddings.csv"

: "${SCRATCH:?SCRATCH env var is not set}"

mkdir -p "$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
export PYTHONUNBUFFERED=1
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
    echo "ERROR: uv not found. Run an embedding job first or set UV_INSTALL_DIR." >&2
    exit 1
  fi
fi

echo "=== Clay analysis ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "MODEL       : $ANALYSIS_MODEL"
echo "AGG_DIR     : $AGG_DIR"
echo "OUTPUTS     : $SCRATCH/embeddings-health/outputs/$ANALYSIS_MODEL"
echo "venv        : $UV_PROJECT_ENVIRONMENT"
echo ""

cd "$REPO_DIR/code/analyses"

# Combine per-state CSVs into a single national file (skipped if already done).
if [[ ! -s "$COMBINED" ]]; then
  echo "--- Combining per-state Clay CSVs ---"
  uv run --python 3.11 python prepare_embeddings.py \
    --input-dir  "$AGG_DIR" \
    --output     "$COMBINED" \
    --no-area
  echo "Combined file: $COMBINED"
else
  echo "Combined file already exists: $COMBINED"
fi

echo ""
echo "--- Running analysis ---"
uv run --python 3.11 python analyses_sherlock.py --model "$ANALYSIS_MODEL"

echo ""
echo "=== Done ==="
