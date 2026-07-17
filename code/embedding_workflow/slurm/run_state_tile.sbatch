#!/bin/bash
#SBATCH --job-name=embed-tile
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${MODEL:?MODEL is required}"
: "${STATE:?STATE is required}"
: "${NUM_TILES:?NUM_TILES is required}"
YEAR="${YEAR:-2022}"; INDEX="${SLURM_ARRAY_TASK_ID:-0}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
INPUT="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
OUTPUT="$OUTPUT_ROOT/$MODEL/$STATE/${MODEL}_${STATE}_${YEAR}.tif"
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"
module load devel
module load gcc/14.2.0
module load rust/1.90.0
module load cuda/12.6.1 || true
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
if [[ "$MODEL" == olmoearth-* ]] && ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain"; then
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install "einops>=0.7" huggingface_hub universal-pathlib
fi
extra=(); [[ -n "${TEST_CHIPS:-}" ]] && extra+=(--test-chips "$TEST_CHIPS")
uv run --python 3.11 python ../embedding_workflow/generate.py --model "$MODEL" --input "$INPUT" --output "$OUTPUT" --tile-index "$INDEX" --num-tiles "$NUM_TILES" --year "$YEAR" "${extra[@]}"
