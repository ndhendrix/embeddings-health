#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL="${MODEL:?MODEL is required}"; STATE=RI; NUM_TILES="${NUM_TILES:-2}"; YEAR="${YEAR:-2022}"; export REPO_DIR MODEL STATE NUM_TILES YEAR
mkdir -p "$SCRIPT_DIR/logs"
tiles=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --array="0-$((NUM_TILES-1))" --output="$SCRIPT_DIR/logs/${MODEL}_overlap_RI_%A_%a.out" --error="$SCRIPT_DIR/logs/${MODEL}_overlap_RI_%A_%a.err" "$SCRIPT_DIR/run_overlap_tile.sbatch"); tiles="${tiles%%;*}"
ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"; export OUTPUT_ROOT="$ROOT"; output="$ROOT/$MODEL/RI/${MODEL}_overlap-center50_RI_${YEAR}.tif"
merge=$(cd "$REPO_DIR" && MODEL="$MODEL" STATE=RI NUM_TILES="$NUM_TILES" OUTPUT_ROOT="$ROOT" sbatch --parsable --export=ALL --dependency="afterok:$tiles" --output="$SCRIPT_DIR/logs/${MODEL}_overlap_RI_merge_%j.out" --error="$SCRIPT_DIR/logs/${MODEL}_overlap_RI_merge_%j.err" --wrap="module load devel; module load gcc/14.2.0; export UV_PROJECT_ENVIRONMENT='$SCRATCH/embeddings-health/cache/venv-3.11-cpu'; cd '$REPO_DIR/code/embedding_generation'; tiles=(); for ((i=0;i<$NUM_TILES;i++)); do printf -v s '_tile%03d.tif' \$i; tiles+=(\"${output%.tif}\$s\"); done; uv run --python 3.11 python ../embedding_workflow/merge.py --tiles \"\${tiles[@]}\" --output '$output'")
echo "Submitted overlap $MODEL RI: tiles=$tiles merge=${merge%%;*}"
