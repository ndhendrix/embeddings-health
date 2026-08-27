#!/bin/bash
#SBATCH --job-name=fig1-pca3
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/home/users/nhendrix/embeddings-health/code/figures/logs/figure1_panels_%A_%a.out
#SBATCH --error=/home/users/nhendrix/embeddings-health/code/figures/logs/figure1_panels_%A_%a.err

set -euo pipefail

PROJECT="/home/users/nhendrix/embeddings-health"
PYTHON="/scratch/users/nhendrix/embeddings-health/cache/venv-3.11-cpu/bin/python"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Task index: $TASK_INDEX"

"$PYTHON" -u code/figures/figure1_generate_panels.py --task-index "$TASK_INDEX"

echo "Finished: $(date --iso-8601=seconds)"
