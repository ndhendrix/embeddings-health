#!/bin/bash
#SBATCH --job-name=prithvi-tiny-acs
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=code/analyses/slurm/logs/prithvi_tiny_acs_%j.out
#SBATCH --error=code/analyses/slurm/logs/prithvi_tiny_acs_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH env var is not set}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
COMBINED="$SCRATCH/embeddings-health/prithvi_aggregated/prithvi_tiny_2022_all_tracts.csv"

[[ -s "$COMBINED" ]] || { echo "ERROR: missing combined embeddings: $COMBINED" >&2; exit 1; }
mkdir -p "$CACHE_ROOT" "$REPO_DIR/code/analyses/slurm/logs"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-analysis"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load devel
module load gcc/14.2.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"

echo "=== Prithvi Tiny ACS-only analysis ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "EMBEDDINGS  : $COMBINED"
echo "OUTPUTS     : $SCRATCH/embeddings-health/outputs/prithvi_tiny"

cd "$REPO_DIR/code/analyses"
uv run --python 3.11 python analyses_sherlock.py \
  --model prithvi_tiny \
  --stage acs \
  --no-cache

echo "=== Done ==="
