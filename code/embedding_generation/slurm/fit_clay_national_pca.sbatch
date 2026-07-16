#!/bin/bash
#SBATCH --job-name=fit-clay-national-pca
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/fit_clay_national_pca_%j.out
#SBATCH --error=code/embedding_generation/slurm/logs/fit_clay_national_pca_%j.err

# Fit a single national PCA across all complete Clay state embeddings, mirroring
# fit_national_pca.sbatch (Prithvi) but for Clay's flat, single-variant layout.
# Run once after all/most Clay embedding jobs finish, then run
# submit_clay_aggregate_all_states.sh.
#
# Usage:
#   sbatch fit_clay_national_pca.sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
EMBED_OUT_DIR="${EMBED_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
NATIONAL_PCA_DIR="${NATIONAL_PCA_DIR:-$SCRATCH/embeddings-health/clay_aggregated/national_pca}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

OUTPUT_PKL="$NATIONAL_PCA_DIR/clay_v1.5_national_pca.pkl"

mkdir -p "$NATIONAL_PCA_DIR" "$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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

echo "=== Fit Clay national PCA ==="
echo "Year:      $YEAR"
echo "Embed dir: $EMBED_OUT_DIR"
echo "Output:    $OUTPUT_PKL"
echo ""

cd "$EMBED_DIR"

uv run --python 3.11 python fit_national_pca.py \
  --model          clay \
  --year           "$YEAR" \
  --embed-dir      "$EMBED_OUT_DIR" \
  --output         "$OUTPUT_PKL" \
  --n-components   64 \
  --total-samples  500000

echo ""
echo "=== Done ==="
echo "PCA model saved to: $OUTPUT_PKL"
echo "Next: bash submit_clay_aggregate_all_states.sh"
