#!/bin/bash
#SBATCH --job-name=fig1-prithvi-pa-recon
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-1
#SBATCH --output=code/figures/logs/figure1_prithvi_pa_recon_%A_%a.out
#SBATCH --error=code/figures/logs/figure1_prithvi_pa_recon_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/nhendrix/embeddings-health}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/nhendrix/embeddings-health}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH_ROOT/cache}"
VARIANTS=(tiny 300M-TL)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
INPUT="$SCRATCH_ROOT/figures/figure1_prithvi_reconstructed/palo_alto_annual_30m.tif"
OUTPUT_ROOT="$SCRATCH_ROOT/figures/figure1_prithvi_reconstructed"

mkdir -p "$REPO_DIR/code/figures/logs" "$OUTPUT_ROOT"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR/code/embedding_generation"
uv run --python 3.11 python "$REPO_DIR/code/figures/figure1_prithvi_centered.py" \
  --variant "$VARIANT" \
  --input "$INPUT" \
  --output-root "$OUTPUT_ROOT"
