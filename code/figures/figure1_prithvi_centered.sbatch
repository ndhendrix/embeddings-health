#!/bin/bash
#SBATCH --job-name=fig1-prithvi-pa-center
#SBATCH -p gpu,owners
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=code/figures/logs/figure1_prithvi_pa_centered_%A_%a.out
#SBATCH --error=code/figures/logs/figure1_prithvi_pa_centered_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/nhendrix/embeddings-health}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/nhendrix/embeddings-health}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH_ROOT/cache}"
VARIANTS=(tiny 300M-TL)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p "$REPO_DIR/code/figures/logs" "$SCRATCH_ROOT/figures/figure1_prithvi_centered"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
  module load rust/1.90.0
  module load cuda/12.6.1 || true
fi

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR/code/embedding_generation"
uv run --python 3.11 python "$REPO_DIR/code/figures/figure1_prithvi_centered.py" \
  --variant "$VARIANT" \
  --data-root "$SCRATCH_ROOT/data/prithvi_data" \
  --output-root "$SCRATCH_ROOT/figures/figure1_prithvi_centered"
