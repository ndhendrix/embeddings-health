#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ROOT="$SCRATCH/embeddings-health"
RUN_ID="${RUN_ID:-analysis_$(date +%Y%m%dT%H%M%S)_$$}"
RUN_ROOT="${RUN_ROOT:-$ROOT/cache/prithvi_300m_tl_analysis_runs/$RUN_ID}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/cache/prithvi_300m_tl_full_prepared}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$ROOT/outputs/prithvi_300m_tl_full}"
PYTHON="${PYTHON:-$ROOT/cache/venv-3.11-cpu/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"
export REPO_DIR RUN_ROOT PREPARED_DIR OUTPUTS_DIR PYTHON LOG_DIR

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Run ID:       $RUN_ID"
  echo "Run root:     $RUN_ROOT"
  echo "Prepared dir: $PREPARED_DIR"
  echo "Outputs:      $OUTPUTS_DIR"
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

prep_job=$(sbatch --parsable \
  --output="$LOG_DIR/prithvi_300m_prep_${RUN_ID}_%j.out" \
  --error="$LOG_DIR/prithvi_300m_prep_${RUN_ID}_%j.err" \
  --export=ALL \
  "$SCRIPT_DIR/run_prithvi_300m_full_prepare.sbatch" | cut -d';' -f1)

dispatch_job=$(sbatch --parsable \
  --dependency="afterok:$prep_job" \
  --output="$LOG_DIR/prithvi_300m_dispatch_${RUN_ID}_%j.out" \
  --error="$LOG_DIR/prithvi_300m_dispatch_${RUN_ID}_%j.err" \
  --export=ALL \
  "$SCRIPT_DIR/run_prithvi_300m_full_dispatch.sbatch" | cut -d';' -f1)

{
  echo "RUN_ID=$RUN_ID"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "PREPARED_DIR=$PREPARED_DIR"
  echo "OUTPUTS_DIR=$OUTPUTS_DIR"
  echo "PREP_JOB=$prep_job"
  echo "DISPATCH_JOB=$dispatch_job"
} > "$RUN_ROOT/jobs.txt"

echo "Run ID:     $RUN_ID"
echo "Preparation:$prep_job"
echo "Dispatcher: $dispatch_job (afterok:$prep_job)"
echo "Job record: $RUN_ROOT/jobs.txt"
echo "Resume with: RUN_ID=$RUN_ID SCRATCH=$SCRATCH bash $0"
