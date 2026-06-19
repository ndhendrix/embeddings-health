#!/bin/bash
#SBATCH --job-name=fit-national-pca
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/fit_national_pca_%j.out
#SBATCH --error=code/embedding_generation/slurm/logs/fit_national_pca_%j.err

# Fit a single national PCA for one model variant across all complete state
# raw embeddings.  Run this once per variant after all embedding jobs finish,
# then run submit_aggregate_all_states.sh to use the result.
#
# Usage:
#   sbatch --export=ALL,VARIANT=tiny fit_national_pca.sbatch
#   sbatch --export=ALL,VARIANT=300M-TL fit_national_pca.sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:?Set VARIANT=tiny or VARIANT=300M-TL before submitting}"

: "${SCRATCH:?SCRATCH is not set}"
EMBED_OUT_DIR="${EMBED_OUT_DIR:-$SCRATCH/embeddings-health/prithvi_embeddings}"
NATIONAL_PCA_DIR="${NATIONAL_PCA_DIR:-$SCRATCH/embeddings-health/prithvi_aggregated/national_pca}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_PKL="$NATIONAL_PCA_DIR/prithvi_${SAFE_VARIANT}_national_pca.pkl"

mkdir -p "$NATIONAL_PCA_DIR" "$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
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

echo "=== Fit national PCA ==="
echo "Variant:   $VARIANT"
echo "Year:      $YEAR"
echo "Embed dir: $EMBED_OUT_DIR/$VARIANT/"
echo "Output:    $OUTPUT_PKL"
echo ""

cd "$EMBED_DIR"

uv run --python 3.11 python fit_national_pca.py \
  --variant        "$VARIANT" \
  --year           "$YEAR" \
  --embed-dir      "$EMBED_OUT_DIR" \
  --output         "$OUTPUT_PKL" \
  --n-components   64 \
  --total-samples  500000

echo ""
echo "=== Done ==="
echo "PCA model saved to: $OUTPUT_PKL"
echo "Next: bash submit_aggregate_all_states.sh"
