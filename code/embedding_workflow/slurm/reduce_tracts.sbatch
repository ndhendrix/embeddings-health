#!/bin/bash
#SBATCH --job-name=tract-reduce
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?}"; : "${MODEL:?}"; : "${STATE:?}"; : "${NUM_TILES:?}"
YEAR="${YEAR:-2022}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"; CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
DIR="$OUTPUT_ROOT/$MODEL/$STATE"; PRODUCT_STEM="${PRODUCT_STEM:-${MODEL}_${STATE}_${YEAR}}"; partials=()
for ((i=0;i<NUM_TILES;i++)); do p="$DIR/partials/$PRODUCT_STEM/tile$(printf '%03d' "$i").npz"; [[ -s "$p" ]] || exit 1; partials+=("$p"); done
prefix=OE; [[ "$MODEL" == clay-* ]] && prefix=CL
module load devel; module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/aggregate_tiles.py reduce --partials "${partials[@]}" --output "$DIR/${PRODUCT_STEM}_tracts.csv" --year "$YEAR" --prefix "$prefix"
