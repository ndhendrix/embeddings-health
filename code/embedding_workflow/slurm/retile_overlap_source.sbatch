#!/bin/bash
#SBATCH --job-name=oe-base-retile
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${RETILE_TASK_FILE:?RETILE_TASK_FILE is required}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

IFS=$'\t' read -r STATE INPUT OUTPUT SOURCE_QA <<< "$(sed -n "$((INDEX + 1))p" "$RETILE_TASK_FILE")"
: "${STATE:?empty retile task}"
: "${INPUT:?empty retile input}"
: "${OUTPUT:?empty retile output}"
echo "state=$STATE source_qa=$SOURCE_QA input=$INPUT output=$OUTPUT"

module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
export GDAL_CACHEMAX=2048

cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/retile_source.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --block-size 512
