#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${SCRATCH:?SCRATCH must be set}"

ROOT="$SCRATCH/embeddings-health"
EMBED_ROOT="${EMBED_ROOT:-$ROOT/prithvi_embeddings/300M-TL}"
AGG_OUT_DIR="${AGG_OUT_DIR:-$ROOT/prithvi_aggregated_full/300M-TL}"
TRACT_DIR="${TRACT_DIR:-$ROOT/data/census_tracts}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
MAX_CONCURRENT="${MAX_CONCURRENT:-48}"
YEAR="${YEAR:-2022}"

RUN_ID="block_$(date +%Y%m%dT%H%M%S)_$$"
PLAN_ROOT="$CACHE_DIR/prithvi_300m_block_plans/$RUN_ID"
PARTIAL_ROOT="$ROOT/prithvi_aggregated_full/block_partials/$RUN_ID"
MANIFEST="$PLAN_ROOT/tasks.tsv"
STATE_FILE="$PLAN_ROOT/states.txt"
PYTHON="${PYTHON:-$CACHE_DIR/venv-3.11-cpu/bin/python}"

mkdir -p "$PLAN_ROOT" "$PARTIAL_ROOT" "$AGG_OUT_DIR" "$LOG_DIR"
"$PYTHON" "$REPO_DIR/code/embedding_generation/plan_prithvi_block_tasks.py" \
  --embedding-root "$EMBED_ROOT" \
  --aggregate-root "$AGG_OUT_DIR" \
  --tract-root "$TRACT_DIR" \
  --partial-root "$PARTIAL_ROOT" \
  --manifest "$MANIFEST" \
  --states-output "$STATE_FILE" \
  --year "$YEAR"

TASKS=$(wc -l < "$MANIFEST")
STATES=$(wc -l < "$STATE_FILE")
if (( TASKS == 0 || STATES == 0 )); then
  echo "All 300M-TL states are already complete; nothing to submit."
  exit 0
fi

echo "Run ID:       $RUN_ID"
echo "States:       $STATES"
echo "Block tasks:  $TASKS"
echo "Concurrency:  $MAX_CONCURRENT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

export REPO_DIR SCRATCH MANIFEST STATE_FILE PARTIAL_ROOT AGG_OUT_DIR TRACT_DIR YEAR PYTHON
PARTIAL_JOB=$(sbatch --parsable \
  --array="0-$((TASKS - 1))%$MAX_CONCURRENT" \
  --output="$LOG_DIR/prithvi_300m_block_${RUN_ID}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_block_${RUN_ID}_%A_%a.err" \
  --export=ALL \
  "$SCRIPT_DIR/run_prithvi_300m_block_partial.sbatch" | cut -d';' -f1)

REDUCE_JOB=$(sbatch --parsable \
  --dependency="afterok:$PARTIAL_JOB" \
  --array="0-$((STATES - 1))%12" \
  --output="$LOG_DIR/prithvi_300m_reduce_${RUN_ID}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_reduce_${RUN_ID}_%A_%a.err" \
  --export=ALL \
  "$SCRIPT_DIR/run_prithvi_300m_block_reduce.sbatch" | cut -d';' -f1)

JOBS_FILE="$PLAN_ROOT/jobs.txt"
{
  echo "RUN_ID=$RUN_ID"
  echo "PARTIAL_JOB=$PARTIAL_JOB"
  echo "REDUCE_JOB=$REDUCE_JOB"
  echo "MANIFEST=$MANIFEST"
  echo "STATE_FILE=$STATE_FILE"
  echo "PARTIAL_ROOT=$PARTIAL_ROOT"
} > "$JOBS_FILE"

echo "Partial array: $PARTIAL_JOB"
echo "Reduce array:  $REDUCE_JOB (afterok:$PARTIAL_JOB)"
echo "Job record:    $JOBS_FILE"
