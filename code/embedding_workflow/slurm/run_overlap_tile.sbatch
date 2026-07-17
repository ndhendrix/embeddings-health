#!/bin/bash
#SBATCH --job-name=overlap-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@600
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?}"; : "${MODEL:?}"
YEAR="${YEAR:-2022}"; INDEX="${SLURM_ARRAY_TASK_ID:-0}"; CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
if [[ -n "${TASK_FILE:-}" ]]; then
 read -r STATE TILE_INDEX NUM_TILES TASK_GRID_ROWS TASK_GRID_COLS <<< "$(sed -n "$((INDEX+1))p" "$TASK_FILE")"; INDEX="$TILE_INDEX"
 [[ -n "${TASK_GRID_ROWS:-}" ]] && GRID_ROWS="$TASK_GRID_ROWS"
 [[ -n "${TASK_GRID_COLS:-}" ]] && GRID_COLS="$TASK_GRID_COLS"
fi
: "${STATE:?}"; : "${NUM_TILES:?}"
INPUT="${INPUT:-${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}/s2_annual_${STATE}_${YEAR}_olmoearth.tif}"
ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"; OUTPUT="$ROOT/$MODEL/$STATE/${MODEL}_overlap-center50_${STATE}_${YEAR}.tif"
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"
TILE_OUTPUT="$OUTPUT"; (( NUM_TILES > 1 )) && TILE_OUTPUT="${OUTPUT%.tif}_tile$(printf '%03d' "$INDEX").tif"
if [[ -s "$TILE_OUTPUT" && -s "${TILE_OUTPUT%.tif}.validation.json" ]]; then echo "validated output exists: $TILE_OUTPUT"; exit 0; fi
module load devel; module load gcc/14.2.0; module load rust/1.90.0; module load cuda/12.6.1 || true
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}" HF_HOME="$CACHE_ROOT/huggingface" EMBEDDING_COG_INTERLEAVE=band
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
if [[ "$MODEL" == olmoearth-* ]] && ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain"; then
 VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps git+https://github.com/allenai/olmoearth_pretrain
 VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install "einops>=0.7" huggingface_hub universal-pathlib
fi
extra=(); [[ -n "${TEST_BLOCKS:-}" ]] && extra+=(--test-blocks "$TEST_BLOCKS"); [[ -n "${GRID_ROWS:-}" ]] && extra+=(--grid-rows "$GRID_ROWS" --grid-cols "${GRID_COLS:-1}")
export CODE_GIT_SHA="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
uv run --python 3.11 python ../embedding_workflow/generate_overlap.py --model "$MODEL" --input "$INPUT" --output "$OUTPUT" --tile-index "$INDEX" --num-tiles "$NUM_TILES" --year "$YEAR" ${extra[@]:-}
partial=(); [[ -n "${TEST_BLOCKS:-}" ]] && partial+=(--allow-partial)
uv run --python 3.11 python ../embedding_workflow/validate_overlap.py --raster "$TILE_OUTPUT" --model "$MODEL" ${partial[@]:-}
