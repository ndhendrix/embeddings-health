#!/bin/bash
# Submit corrected OlmoEarth Nano overlap inference in bounded rolling waves.
#
# Each invocation scans real outputs, submits a bounded wave of missing tiles,
# then schedules a lightweight controller to refill once the active Nano array
# count drops below REFILL_THRESHOLD and the user's total submitted job count
# leaves room. Existing validated tiles are skipped by the planner/worker, so
# manual reruns are safe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${SCRATCH:?Set SCRATCH before submitting.}"

YEAR="${YEAR:-2022}"
MODEL="olmoearth-v1.2-nano"
STATE_SET="${STATE_SET:-conus}"
CONUS_STATES="AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY"
STATES="${STATES:-$CONUS_STATES}"
EXCLUDE_STATES="${EXCLUDE_STATES:-}"
OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
CACHE="$CACHE_ROOT/olmoearth_nano_overlap"
LOG_DIR="${LOG_DIR:-$REPO_DIR/code/embedding_workflow/slurm/logs}"
mkdir -p "$CACHE" "$LOG_DIR"

TARGET_BLOCKS_PER_TILE="${TARGET_BLOCKS_PER_TILE:-20000}"
MAX_SUBMIT_PER_RUN="${MAX_SUBMIT_PER_RUN:-60}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
GPU_REQUEST_LIMIT="${GPU_REQUEST_LIMIT:-100}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-10}"
REFILL_POLL_SECONDS="${REFILL_POLL_SECONDS:-60}"
CONTROLLER_RESERVE="${CONTROLLER_RESERVE:-2}"
TILING="${TILING:-rectangular}"
PY="${PY:-$CACHE_ROOT/venv-3.11-cpu/bin/python}"
INFER_JOB_NAME="oe-nano-overlap"
CONTROLLER_TIME="${CONTROLLER_TIME:-24:00:00}"
RUN_ID="${SLURM_JOB_ID:-manual}_$(date +%Y%m%dT%H%M%S)_$$"
TASK_FILE="$CACHE/tasks_${YEAR}_${STATE_SET}_${RUN_ID}.txt"
ALL_TASK_FILE="$CACHE/all_tasks_${YEAR}_${STATE_SET}_${RUN_ID}.txt"
STATE_FILE="$CACHE/states_${YEAR}_${STATE_SET}_${RUN_ID}.txt"
MANIFEST="$CACHE/manifest_${YEAR}_${STATE_SET}_${RUN_ID}.json"

if [[ "${DRY_RUN:-0}" == 1 || "${CONFIRM_OE_NANO_SUBMIT:-0}" != 1 ]]; then
  available="$MAX_SUBMIT_PER_RUN"
else
  if [[ -n "${WATCH_JOB_ID:-}" ]]; then
    echo "Watching Nano GPU work after wave $WATCH_JOB_ID; refill threshold=$REFILL_THRESHOLD"
    while true; do
      active=$(squeue -r --noheader --user="$USER" --name="$INFER_JOB_NAME" --format='%i' | wc -l)
      total_submitted=$(squeue -r --noheader --user="$USER" --format='%i' | wc -l)
      available=$((GPU_REQUEST_LIMIT - total_submitted - CONTROLLER_RESERVE))
      if (( active <= REFILL_THRESHOLD && available >= 1 )); then
        break
      fi
      echo "Active Nano tasks: $active; total submitted jobs: $total_submitted; refill capacity: $available; checking again in ${REFILL_POLL_SECONDS}s"
      sleep "$REFILL_POLL_SECONDS"
    done
    echo "Refilling with up to $available tasks; $active older Nano tasks and $total_submitted total jobs remain active"
    unset WATCH_JOB_ID
  else
    total_submitted=$(squeue -r --noheader --user="$USER" --format='%i' | wc -l)
    available=$((GPU_REQUEST_LIMIT - total_submitted - CONTROLLER_RESERVE))
  fi
  if (( available < 1 )); then
    echo "No submission capacity: $total_submitted jobs are active and $CONTROLLER_RESERVE slots are reserved for controllers/reducers." >&2
    exit 1
  fi
fi
wave_limit="$MAX_SUBMIT_PER_RUN"
(( wave_limit > available )) && wave_limit="$available"

read -r -a state_args <<< "$STATES"
planner=("$PY" "$REPO_DIR/code/embedding_workflow/plan_overlap_tasks.py"
  --model "$MODEL"
  --composites "$COMPOSITE_DIR"
  --output-root "$OUTPUT_ROOT"
  --task-file "$TASK_FILE"
  --all-task-file "$ALL_TASK_FILE"
  --state-file "$STATE_FILE"
  --manifest "$MANIFEST"
  --year "$YEAR"
  --target-blocks "$TARGET_BLOCKS_PER_TILE"
  --max-tasks "$wave_limit"
  --tiling "$TILING"
  --states "${state_args[@]}")
if [[ -n "$EXCLUDE_STATES" ]]; then
  read -r -a exclude_state_args <<< "$EXCLUDE_STATES"
  planner+=(--exclude-states "${exclude_state_args[@]}")
fi
"${planner[@]}"

count=$(wc -l < "$TASK_FILE")
total_missing=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_missing_tiles"])' "$MANIFEST")
echo "Planned Nano wave: $count tasks ($total_missing missing overall); manifest: $MANIFEST"

if [[ "${DRY_RUN:-0}" == 1 || "${CONFIRM_OE_NANO_SUBMIT:-0}" != 1 ]]; then
  echo "DRY RUN (set CONFIRM_OE_NANO_SUBMIT=1 to submit)"
  sed -n '1,20p' "$TASK_FILE"
  exit 0
fi

export REPO_DIR SCRATCH YEAR MODEL OUTPUT_ROOT OVERLAP_OUTPUT_ROOT="$OUTPUT_ROOT" COMPOSITE_DIR CACHE_ROOT
export PRODUCT_MODE=overlap
export STATE_SET STATES EXCLUDE_STATES TARGET_BLOCKS_PER_TILE MAX_SUBMIT_PER_RUN MAX_CONCURRENT GPU_REQUEST_LIMIT REFILL_THRESHOLD REFILL_POLL_SECONDS CONTROLLER_RESERVE TILING

if (( count == 0 )); then
  reduced_missing=0
  while read -r state num_tiles grid_rows grid_cols; do
    stem="${MODEL}_overlap-center50_${state}_${YEAR}"
    csv="$OUTPUT_ROOT/$MODEL/$state/${stem}_tracts.csv"
    [[ -s "$csv" && -s "${csv%.csv}.validation.json" ]] || ((reduced_missing += 1))
  done < "$STATE_FILE"
  if (( reduced_missing == 0 )); then
    echo "All Nano tiles and state tract reductions are complete for STATE_SET=$STATE_SET."
    exit 0
  fi

  final_partials=$(TASK_FILE="$ALL_TASK_FILE" sbatch \
    --parsable \
    --job-name=oe-nano-final-partials \
    --export=ALL \
    --output="$LOG_DIR/oe_nano_final_partials_%j.out" \
    --error="$LOG_DIR/oe_nano_final_partials_%j.err" \
    "$SCRIPT_DIR/aggregate_missing_partials_driver.sbatch")
  final_partials="${final_partials%%;*}"
  reduction=$(STATE_FILE="$STATE_FILE" sbatch \
    --parsable \
    --job-name=oe-nano-reduce \
    --dependency="afterok:$final_partials" \
    --export=ALL \
    --output="$LOG_DIR/oe_nano_reduce_%j.out" \
    --error="$LOG_DIR/oe_nano_reduce_%j.err" \
    "$SCRIPT_DIR/reduce_complete_states_driver.sbatch")
  reduction="${reduction%%;*}"
  echo "Inference complete; submitted final partials=$final_partials reduction=$reduction"
  exit 0
fi

infer=$(TASK_FILE="$TASK_FILE" sbatch \
  --parsable \
  --job-name="$INFER_JOB_NAME" \
  --export=ALL \
  --array="0-$((count - 1))%$MAX_CONCURRENT" \
  --output="$LOG_DIR/oe_nano_overlap_%A_%a.out" \
  --error="$LOG_DIR/oe_nano_overlap_%A_%a.err" \
  "$SCRIPT_DIR/run_overlap_tile.sbatch")
infer="${infer%%;*}"

partials=$(TASK_FILE="$TASK_FILE" sbatch \
  --parsable \
  --job-name=oe-nano-partials \
  --dependency="afterany:$infer" \
  --export=ALL \
  --output="$LOG_DIR/oe_nano_partials_%j.out" \
  --error="$LOG_DIR/oe_nano_partials_%j.err" \
  "$SCRIPT_DIR/aggregate_missing_partials_driver.sbatch")
partials="${partials%%;*}"

continuation=$(sbatch \
  --parsable \
  --job-name=oe-nano-overlap-resubmit \
  --partition=normal \
  --time="$CONTROLLER_TIME" \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/oe_nano_overlap_resubmit_%j.out" \
  --error="$LOG_DIR/oe_nano_overlap_resubmit_%j.err" \
  --export=ALL,WATCH_JOB_ID="$infer" \
  --wrap="bash '$SCRIPT_DIR/submit_olmoearth_nano_overlap_resubmit.sh'")
continuation="${continuation%%;*}"
echo "Submitted Nano inference=$infer partials=$partials continuation=$continuation"
