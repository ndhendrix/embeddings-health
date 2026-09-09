#!/bin/bash
#SBATCH --job-name=export-acs-folds
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=code/analyses/slurm/logs/acs_folds_%j.out
#SBATCH --error=code/analyses/slurm/logs/acs_folds_%j.err

# Run from project root:
#   sbatch code/analyses/slurm/export_acs_folds.sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

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

echo "=== Export ACS folds (no fitting) ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "OUTPUTS     : $SCRATCH/embeddings-health/diagnostics/acs_folds_${SLURM_JOB_ID}.json"
echo "venv        : $UV_PROJECT_ENVIRONMENT"
echo ""

cd "$REPO_DIR/code/analyses"

uv run --python 3.11 python diagnostics/export_acs_folds.py \
  --output "$SCRATCH/embeddings-health/diagnostics/acs_folds_${SLURM_JOB_ID}.json"

echo ""
echo "=== Done ==="
