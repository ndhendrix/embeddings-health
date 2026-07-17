#!/bin/bash
#SBATCH --job-name=embed-merge
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${MODEL:?MODEL is required}"
: "${STATE:?STATE is required}"
: "${NUM_TILES:?NUM_TILES is required}"
YEAR="${YEAR:-2022}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
OUTPUT="$OUTPUT_ROOT/$MODEL/$STATE/${MODEL}_${STATE}_${YEAR}.tif"
module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
tiles=()
for ((i=0;i<NUM_TILES;i++)); do
  printf -v suffix "_tile%03d.tif" "$i"; tile="${OUTPUT%.tif}${suffix}"
  [[ -s "$tile" ]] || { echo "missing tile: $tile" >&2; exit 1; }; tiles+=("$tile")
done
uv run --python 3.11 python ../embedding_workflow/merge.py --tiles "${tiles[@]}" --output "$OUTPUT"
