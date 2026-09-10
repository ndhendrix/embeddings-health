#!/usr/bin/env bash
#SBATCH --job-name=prithvi-income-folds
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=prithvi-income-folds-%j.log
set -euo pipefail
: "${SCRATCH:?SCRATCH is required}"
REPRO_REPO="${REPRO_REPO:-$SCRATCH/embeddings-health-reproduction}"
DATA_DIR="${DATA_DIR:-$SCRATCH/embeddings-health-reproduction-data}"
module load devel gcc/14.2.0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$REPRO_REPO/reproduction/.cache/matplotlib"
"$REPRO_REPO/reproduction/.venv/bin/python" \
  "$REPRO_REPO/code/analyses/diagnostics/prithvi_300m_income_folds.py" \
  --reproduction "$REPRO_REPO/reproduction" --data-dir "$DATA_DIR" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --output "$SCRATCH/prithvi-300m-income-folds-${SLURM_JOB_ID}"
