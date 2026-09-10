#!/usr/bin/env bash
#SBATCH --job-name=tiny-acs-diagnostic
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=tiny-acs-diagnostic-%j.log
set -euo pipefail
: "${SCRATCH:?SCRATCH is required}"
REPRO_REPO="${REPRO_REPO:-$SCRATCH/embeddings-health-reproduction}"
ORIGINAL_REPO="${ORIGINAL_REPO:-$HOME/embeddings-health}"
ORIGINAL_DATA="${ORIGINAL_DATA:-$SCRATCH/embeddings-health}"
DATA_DIR="${DATA_DIR:-$SCRATCH/embeddings-health-reproduction-data}"
EMBEDDINGS="${EMBEDDINGS:-$ORIGINAL_DATA/prithvi_aggregated_full/tiny}"
module load devel gcc/14.2.0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$REPRO_REPO/reproduction/.cache/matplotlib"
"$REPRO_REPO/reproduction/.venv/bin/python" \
  "$REPRO_REPO/code/analyses/diagnostics/prithvi_tiny_acs.py" \
  --reproduction "$REPRO_REPO/reproduction" --original-repo "$ORIGINAL_REPO" \
  --original-data "$ORIGINAL_DATA" --embeddings "$EMBEDDINGS" \
  --data-dir "$DATA_DIR" --threads "$SLURM_CPUS_PER_TASK" \
  --output "$SCRATCH/prithvi-tiny-acs-diagnostic-${SLURM_JOB_ID}"
