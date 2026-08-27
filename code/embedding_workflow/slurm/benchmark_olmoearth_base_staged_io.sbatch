#!/bin/bash
#SBATCH --job-name=oe-base-stage
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=code/embedding_workflow/slurm/logs/oe_base_stage_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/oe_base_stage_%j.err
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
INPUT="${INPUT:-$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_RI_2022_olmoearth.tif}"
OUTPUT="${OUTPUT:-$SCRATCH/embeddings-health/olmoearth_base_ri_experiment_v2/batch_sweep_staged.json}"
YEAR="${YEAR:-2022}"
LOCAL_DIR="$(mktemp -d "/tmp/${USER}-oe-base-stage-${SLURM_JOB_ID}-XXXXXX")"
trap 'rm -rf "$LOCAL_DIR"' EXIT
LOCAL_INPUT="$LOCAL_DIR/$(basename "$INPUT")"

mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT" "$REPO_DIR/code/embedding_workflow/slurm/logs"
copy_started=$(date +%s)
cp "$INPUT" "$LOCAL_INPUT"
copy_seconds=$(($(date +%s) - copy_started))
echo "staged $INPUT to $LOCAL_INPUT in ${copy_seconds}s"

module load devel
module load gcc/14.2.0
module load rust/1.90.0
module load cuda/12.6.1 || true
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"

cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain"; then
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install "einops>=0.7" huggingface_hub universal-pathlib
fi

uv run --python 3.11 python ../embedding_workflow/benchmark_olmoearth_batch.py \
  --model olmoearth-v1.2-base \
  --input "$LOCAL_INPUT" \
  --output "$OUTPUT" \
  --year "$YEAR" \
  --num-chips 1024 \
  --repeats 2 \
  --batch-sizes 16
