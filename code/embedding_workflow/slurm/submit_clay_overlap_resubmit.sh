#!/bin/bash
# Submit Clay overlap inference in bounded GPU waves and resubmit until done.
#
# This is designed for accounts/QOS settings that cap submitted GPU array
# elements. Each invocation scans real outputs, submits at most
# MAX_SUBMIT_PER_RUN missing tiles, then schedules this script to run again
# after that wave reaches a terminal state. Existing validated tiles are
# skipped by the planner/worker, so it is safe to rerun manually as well.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${SCRATCH:?Set SCRATCH before submitting.}"

YEAR="${YEAR:-2022}"
MODEL="${MODEL:-clay-1.5}"
STATE_SET="${STATE_SET:-eastern}"
EASTERN_STATES="CT DC DE FL GA IN KY MA MD ME MI NC NH NJ NY OH PA RI SC TN VA VT WV"
if [[ "$STATE_SET" == eastern ]]; then STATES="${STATES:-$EASTERN_STATES}"; fi

OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
CACHE="$CACHE_ROOT/clay_overlap"
LOG_DIR="${LOG_DIR:-$REPO_DIR/code/embedding_workflow/slurm/logs}"
mkdir -p "$CACHE" "$LOG_DIR"

TARGET_BLOCKS_PER_TILE="${TARGET_BLOCKS_PER_TILE:-4000}"
MAX_SUBMIT_PER_RUN="${MAX_SUBMIT_PER_RUN:-100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
GPU_REQUEST_LIMIT="${GPU_REQUEST_LIMIT:-100}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-10}"
REFILL_POLL_SECONDS="${REFILL_POLL_SECONDS:-60}"
CONTROLLER_RESERVE="${CONTROLLER_RESERVE:-1}"
TILING="${TILING:-rectangular}"
PY="${PY:-$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python}"
CONTROLLER_TIME="${CONTROLLER_TIME:-24:00:00}"

if [[ -n "${WATCH_JOB_ID:-}" ]]; then
  echo "Watching GPU work after wave $WATCH_JOB_ID; refill threshold=$REFILL_THRESHOLD"
  while true; do
    ACTIVE=$(squeue -r --noheader --user="$USER" --name=overlap-embed --format='%i' | wc -l)
    TOTAL_SUBMITTED=$(squeue -r --noheader --user="$USER" --format='%i' | wc -l)
    AVAILABLE=$((GPU_REQUEST_LIMIT - TOTAL_SUBMITTED - CONTROLLER_RESERVE))
    if (( ACTIVE <= REFILL_THRESHOLD && AVAILABLE >= 1 )); then break; fi
    echo "Active overlap tasks: $ACTIVE; total submitted jobs: $TOTAL_SUBMITTED; refill capacity: $AVAILABLE; checking again in ${REFILL_POLL_SECONDS}s"
    sleep "$REFILL_POLL_SECONDS"
  done
  echo "Refilling with up to $AVAILABLE tasks; $ACTIVE older overlap tasks and $TOTAL_SUBMITTED total jobs remain active"
  unset WATCH_JOB_ID
else
  TOTAL_SUBMITTED=$(squeue -r --noheader --user="$USER" --format='%i' | wc -l)
  AVAILABLE=$((GPU_REQUEST_LIMIT - TOTAL_SUBMITTED - CONTROLLER_RESERVE))
  if (( AVAILABLE < 1 )); then
    echo "No submission capacity: $TOTAL_SUBMITTED jobs are active and $CONTROLLER_RESERVE slot is reserved for the controller." >&2
    exit 1
  fi
fi
if (( MAX_SUBMIT_PER_RUN > AVAILABLE )); then MAX_SUBMIT_PER_RUN=$AVAILABLE; fi

LABEL="$STATE_SET"
if [[ -n "${STATES:-}" ]]; then LABEL="${LABEL}_$(tr ' ' '_' <<< "$STATES" | cut -c1-80)"; fi
TASK_FILE="$CACHE/tasks_${YEAR}_${LABEL}.txt"
MANIFEST="$CACHE/manifest_${YEAR}_${LABEL}.json"

planner=("$PY" "$REPO_DIR/code/embedding_workflow/plan_clay_tasks.py"
  --composites "$COMPOSITE_DIR"
  --output-root "$OUTPUT_ROOT"
  --task-file "$TASK_FILE"
  --manifest "$MANIFEST"
  --year "$YEAR"
  --target-blocks "$TARGET_BLOCKS_PER_TILE"
  --max-tasks "$MAX_SUBMIT_PER_RUN"
  --tiling "$TILING")
if [[ -n "${STATES:-}" ]]; then read -r -a state_args <<< "$STATES"; planner+=(--states "${state_args[@]}"); fi

"${planner[@]}"
COUNT=$(wc -l < "$TASK_FILE")
echo "Planned this wave: $COUNT Clay inference tasks"
echo "Manifest: $MANIFEST"
echo "Task file: $TASK_FILE"

if (( COUNT == 0 )); then
  echo "No missing Clay inference tiles remain for STATE_SET=$STATE_SET."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == 1 || "${CONFIRM_CLAY_SUBMIT:-0}" != 1 ]]; then
  echo "DRY RUN (set CONFIRM_CLAY_SUBMIT=1 to submit)"
  sed -n '1,20p' "$TASK_FILE"
  exit 0
fi

export REPO_DIR SCRATCH YEAR MODEL OUTPUT_ROOT OVERLAP_OUTPUT_ROOT="$OUTPUT_ROOT" COMPOSITE_DIR CACHE_ROOT TASK_FILE
INFER=$(cd "$REPO_DIR" && sbatch \
  --parsable \
  --export=ALL \
  --array="0-$((COUNT-1))%$MAX_CONCURRENT" \
  --output="$LOG_DIR/clay_overlap_%A_%a.out" \
  --error="$LOG_DIR/clay_overlap_%A_%a.err" \
  "$SCRIPT_DIR/run_overlap_tile.sbatch")
INFER="${INFER%%;*}"
echo "Submitted Clay inference wave: job=$INFER tasks=$COUNT concurrent=$MAX_CONCURRENT"

SELF="$SCRIPT_DIR/submit_clay_overlap_resubmit.sh"
RESUBMIT=$(WATCH_JOB_ID="$INFER" sbatch \
  --parsable \
  --job-name=clay-overlap-resubmit \
  --partition=normal \
  --time="$CONTROLLER_TIME" \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/clay_overlap_resubmit_%j.out" \
  --error="$LOG_DIR/clay_overlap_resubmit_%j.err" \
  --export=ALL \
  --wrap="bash '$SELF'")
RESUBMIT="${RESUBMIT%%;*}"
echo "Scheduled rolling refill controller: $RESUBMIT watching $INFER"
