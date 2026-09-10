#!/usr/bin/env bash
#SBATCH --job-name=prithvi-lineage
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=prithvi-lineage-%j.log
set -euo pipefail
: "${SCRATCH:?SCRATCH is required}"
REPRO_REPO="${REPRO_REPO:-$SCRATCH/embeddings-health-reproduction}"
ORIGINAL_REPO="${ORIGINAL_REPO:-$HOME/embeddings-health}"
PREPARED_DIR="${PREPARED_DIR:-$SCRATCH/embeddings-health/cache/prithvi_300m_tl_full_prepared}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/embeddings-health/cache/prithvi_300m_tl_analysis_runs/analysis_20260906T083000_71493}"
module load devel gcc/14.2.0
"$REPRO_REPO/reproduction/.venv/bin/python" \
  "$REPRO_REPO/code/analyses/diagnostics/prithvi_full.py" \
  --prepared-dir "$PREPARED_DIR" --run-root "$RUN_ROOT" \
  --original-code "$ORIGINAL_REPO/code" \
  --reproduction "$REPRO_REPO/reproduction" \
  --output "$SCRATCH/prithvi-full-diagnostic-${SLURM_JOB_ID}.json"
