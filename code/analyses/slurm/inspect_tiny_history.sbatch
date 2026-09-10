#!/usr/bin/env bash
#SBATCH --job-name=tiny-history
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=tiny-history-%j.log
set -euo pipefail
: "${SCRATCH:?SCRATCH is required}"
REPRO_REPO="${REPRO_REPO:-$SCRATCH/embeddings-health-reproduction}"
ORIGINAL_REPO="${ORIGINAL_REPO:-$HOME/embeddings-health}"
ORIGINAL_DATA="${ORIGINAL_DATA:-$SCRATCH/embeddings-health}"
"$REPRO_REPO/reproduction/.venv/bin/python" \
  "$REPRO_REPO/code/analyses/diagnostics/tiny_history.py" \
  --original-repo "$ORIGINAL_REPO" --scratch-root "$ORIGINAL_DATA" \
  --output "$SCRATCH/tiny-history-${SLURM_JOB_ID}.json"
