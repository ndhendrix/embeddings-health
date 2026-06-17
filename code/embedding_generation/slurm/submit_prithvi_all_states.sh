#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
DATA_DIR="${DATA_DIR:-$SCRATCH/embeddings-health/data/prithvi_data}"

if [[ -z "${FINAL_OUT_DIR:-}" ]]; then
  : "${SCRATCH:?Set SCRATCH or FINAL_OUT_DIR before submitting.}"
  FINAL_OUT_DIR="$SCRATCH/embeddings-health/prithvi_embeddings"
fi

SCRIPT="$SCRIPT_DIR/run_prithvi_state_array.sbatch"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_VARIANTS=("tiny" "300M-TL")
NUM_MODELS="${#MODEL_VARIANTS[@]}"
SBATCH_TIME="${SBATCH_TIME:-1:00:00}"
SBATCH_MEM="${SBATCH_MEM:-128G}"

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "s2_spring_*_${YEAR}_prithvi.tif" -exec basename {} \; \
    | sed -E "s/^s2_spring_(.*)_${YEAR}_prithvi\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no spring composites found in $DATA_DIR for year $YEAR" >&2
  exit 1
fi

# Find task IDs that don't yet have a completed final output
INCOMPLETE_TASK_IDS=()
for i in "${!STATES[@]}"; do
  STATE="${STATES[$i]}"
  for j in "${!MODEL_VARIANTS[@]}"; do
    VARIANT="${MODEL_VARIANTS[$j]}"
    SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
    FINAL_TIF="$FINAL_OUT_DIR/$VARIANT/$STATE/prithvi_${SAFE_VARIANT}_${STATE}_${YEAR}.tif"
    if [[ ! -s "$FINAL_TIF" ]]; then
      INCOMPLETE_TASK_IDS+=("$(( i * NUM_MODELS + j ))")
    fi
  done
done

NUM_STATES="${#STATES[@]}"
NUM_INCOMPLETE="${#INCOMPLETE_TASK_IDS[@]}"
NUM_COMPLETE=$(( NUM_STATES * NUM_MODELS - NUM_INCOMPLETE ))

echo "Repo:          $REPO_DIR"
echo "Data:          $DATA_DIR"
echo "Final outputs: $FINAL_OUT_DIR"
echo "Year:          $YEAR"
echo "States (${NUM_STATES}): ${STATES[*]}"
echo "Models:        ${MODEL_VARIANTS[*]}"
echo "Complete:      $NUM_COMPLETE / $(( NUM_STATES * NUM_MODELS ))"
echo "Slurm time:    $SBATCH_TIME  mem: $SBATCH_MEM"

if (( NUM_INCOMPLETE == 0 )); then
  echo "All state/model pairs complete. Nothing to submit."
  exit 0
fi

TASK_ARRAY=$(IFS=,; echo "${INCOMPLETE_TASK_IDS[*]}")
echo "Submitting ${NUM_INCOMPLETE} incomplete tasks: $TASK_ARRAY"

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"

export REPO_DIR DATA_DIR FINAL_OUT_DIR YEAR

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo "Command: sbatch --export=ALL --time=$SBATCH_TIME --mem=$SBATCH_MEM --array=$TASK_ARRAY $SCRIPT"
  exit 0
fi

cd "$REPO_DIR"
JOB_ID=$(sbatch --export=ALL --time="$SBATCH_TIME" --mem="$SBATCH_MEM" \
  --array="$TASK_ARRAY" --parsable "$SCRIPT" | cut -d';' -f1)
echo "Submitted job array $JOB_ID"

# After the array finishes, run this script again — it will skip completed
# pairs and exit cleanly once everything is done. To stop the chain early,
# run: scancel <resubmit_job_id>
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency=afterany:"$JOB_ID" \
  --job-name=prithvi-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/resubmit_%j.out" \
  --error="$LOG_DIR/resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID will run after $JOB_ID (cancel with: scancel $RESUBMIT_ID)"
