#!/bin/bash
#SBATCH --job-name=prithvi-300m-final
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${PREPARED_DIR:?PREPARED_DIR must be set}"
: "${OUTPUTS_DIR:?OUTPUTS_DIR must be set}"

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
ROOT="$SCRATCH/embeddings-health"
PYTHON="${PYTHON:-$ROOT/cache/venv-3.11-cpu/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

"$PYTHON" "$REPO_DIR/code/analyses/analyses_sharded.py" finalize \
  --prepared-dir "$PREPARED_DIR" \
  --run-root "$RUN_ROOT" \
  --outputs-dir "$OUTPUTS_DIR"
