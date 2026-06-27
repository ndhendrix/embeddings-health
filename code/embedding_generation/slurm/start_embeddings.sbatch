#!/bin/bash
#SBATCH --job-name=start-embeddings
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Trigger script: submitted with --dependency=afterany:<resubmit-job-id>
# Runs all three embed submit scripts once the composite pipeline settles.
# Each submit script skips states without a complete composite and has its
# own resubmit chain, so it is safe to fire even if a stray state is still
# being retried by the composite resubmit chain.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
SCRIPT_DIR="$REPO_DIR/code/embedding_generation/slurm"

: "${SCRATCH:?SCRATCH is not set}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
mkdir -p "$LOG_DIR"

export REPO_DIR SCRATCH LOG_DIR

# GDAL is needed by the embed submit scripts for chip-count estimation
if command -v module >/dev/null 2>&1; then
  module load devel 2>/dev/null || true
  module load physics gdal/3.10.2 2>/dev/null || true
fi

echo "=== Embedding submission trigger ==="
echo "Date: $(date)"
echo "Node: $(hostname -s)"
echo ""

echo "--- OlmoEarth Base ---"
bash "$SCRIPT_DIR/submit_olmoearth_embed_all_states.sh"
echo ""

echo "--- OlmoEarth Nano ---"
bash "$SCRIPT_DIR/submit_olmoearth_nano_embed_all_states.sh"
echo ""

echo "--- Clay ---"
bash "$SCRIPT_DIR/submit_clay_embed_all_states.sh"
echo ""

echo "=== All three embedding pipelines submitted ==="
