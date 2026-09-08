#!/bin/bash
#SBATCH --job-name=prithvi-300m-shard
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${PREPARED_DIR:?PREPARED_DIR must be set}"
: "${STAGE:?STAGE must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
ROOT="$SCRATCH/embeddings-health"
PYTHON="${PYTHON:-$ROOT/cache/venv-3.11-cpu/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$REPO_DIR/code/analyses/analyses_sharded.py" run \
  --prepared-dir "$PREPARED_DIR" \
  --run-root "$RUN_ROOT" \
  --stage "$STAGE" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --threads "${SLURM_CPUS_PER_TASK:-1}"
