#!/bin/bash
#SBATCH --job-name=overlap-pca64-partials
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${CHUNK_DIR:?CHUNK_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

printf -v chunk_name 'chunk_%02d.txt' "$SLURM_ARRAY_TASK_ID"
export TASK_FILE="$CHUNK_DIR/$chunk_name"
[[ -s "$TASK_FILE" ]] || { echo "ERROR: missing or empty chunk: $TASK_FILE" >&2; exit 1; }
export WORKERS="${WORKERS:-4}"
export BAND_CHUNK="${BAND_CHUNK:-64}"

exec bash "$REPO_DIR/code/embedding_workflow/slurm/aggregate_missing_partials_driver.sbatch"
